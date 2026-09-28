"""
Eva V2 - LiveKit agent worker.

Handles TWO kinds of rooms:
  1. Plain browser/dev rooms (no job metadata) - original Eva V2 behavior.
  2. VoiceLink calls, bridged in via livekit_bridge.py - job metadata carries
     {agent, lead, meeting, call_id, callback_url}, mirroring what
     app.py's EvaSession used to build from the same fields.

Run:
    python agent.py console
    python agent.py dev
    python agent.py start
"""

import inspect
import json
import logging
import os
import re
import time
from typing import AsyncIterable

import httpx
import requests
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    ModelSettings,
    RoomOutputOptions,
    cli,
    inference,
)
from livekit.plugins import deepgram, groq, openai, sarvam, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eva-agent")

AGENT_NAME = os.environ.get("AGENT_NAME", "eva-agent")

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "sarvam").strip().lower()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

# ---------------- Cloudflare Workers AI (LLM) ----------------
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_MODEL = os.environ.get("CLOUDFLARE_MODEL", "openai/gpt-6-sol")
CLOUDFLARE_TEMPERATURE = float(os.environ.get("CLOUDFLARE_TEMPERATURE", "0.4"))

# ---------------- Sarvam (LLM) ----------------
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
SARVAM_LLM_MODEL = os.environ.get("SARVAM_LLM_MODEL", "sarvam-105b-conversations")
SARVAM_LLM_TEMPERATURE = float(os.environ.get("SARVAM_LLM_TEMPERATURE", "0.4"))
SARVAM_LLM_MAX_TOKENS = int(os.environ.get("SARVAM_LLM_MAX_TOKENS", "240"))

# ---------------- Sarvam (TTS) ----------------
SARVAM_TTS_MODEL = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v3")
SARVAM_TTS_TEMPERATURE = float(os.environ.get("SARVAM_TTS_TEMPERATURE", "0.6"))

EVA_API_SECRET = os.environ.get("EVA_API_SECRET", "")
PRAVAAH_API_BASE_URL = os.environ.get("PRAVAAH_API_BASE_URL", "").rstrip("/")

GLOBAL_TTS_MODEL = os.environ.get("GLOBAL_TTS_MODEL", "inworld/inworld-tts-2")
VOICE_MALE = os.environ.get("EVA_VOICE_MALE", "shubh")
VOICE_FEMALE = os.environ.get("EVA_VOICE_FEMALE", "priya")
TTS_SPEED = float(os.environ.get("EVA_TTS_SPEED", "1.0"))

SENTENCE_END_RE = re.compile(r"([.!?।\n])")
HINDI_RE = re.compile(r"[\u0900-\u097F]")
BOOK_MEETING_RE = re.compile(r"BOOK_MEETING:\s*(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})")
LANG_NAMES = {"en": "English", "hi": "Hindi"}
SUPPORTED_LANGUAGES = {"en", "hi"}
SPEAKABLE_RE = re.compile(r"[A-Za-z0-9\u0900-\u097F]")
# Hard ceiling on spoken chars per reply (~15 chars/sec of audio => 200 chars ~ 13s)
MAX_SPOKEN_CHARS = int(os.environ.get("EVA_MAX_SPOKEN_CHARS", "200"))


def _detect_lang(text: str) -> str:
    return "hi" if HINDI_RE.search(text) else "en"


def _sarvam_lang_code(lang: str) -> str:
    """Internal 'en'/'hi' -> Sarvam's BCP-47 target_language_code."""
    return "hi-IN" if lang == "hi" else "en-IN"


def render_call_vars(text: str, lead: dict) -> str:
    for key in ("name", "business_name", "email", "phone", "website", "description"):
        text = text.replace("{{%s}}" % key, str((lead or {}).get(key, "") or ""))
    return text


def book_meeting_via_pravaah(meeting_ctx: dict, lead: dict, call_id: str, requested_iso: str):
    """Ported as-is from app.py's EvaSession._trigger_meeting_booking helper."""
    if not meeting_ctx or not EVA_API_SECRET:
        return False, "Sorry, I'm not able to book meetings right now."
    url = meeting_ctx.get("booking_webhook_url") or (
        f"{PRAVAAH_API_BASE_URL}/api/eva-webhook/book-meeting" if PRAVAAH_API_BASE_URL else None
    )
    if not url:
        return False, "Sorry, I'm not able to book meetings right now."
    try:
        resp = requests.post(
            url,
            headers={"X-Eva-Secret": EVA_API_SECRET, "Content-Type": "application/json"},
            json={
                "owner_id": meeting_ctx.get("owner_id", ""),
                "lead_id": meeting_ctx.get("lead_id", ""),
                "lead_name": lead.get("name", ""),
                "lead_phone": lead.get("phone", ""),
                "call_id": call_id or "",
                "agent_id": meeting_ctx.get("agent_id", ""),
                "requested_datetime": requested_iso,
            },
            timeout=15,
        )
        data = resp.json()
        if resp.status_code == 201 and data.get("meeting"):
            when = data["meeting"].get("scheduled_at", requested_iso)
            return True, f"You're all set — your meeting is booked for {when} UTC. I've sent the details over WhatsApp."
        alts = data.get("alternatives") or []
        if alts:
            return False, f"That time isn't available. Would {' or '.join(alts[:2])} (UTC) work instead?"
        return False, data.get("error") or "That time isn't available — could you share another date and time?"
    except Exception as e:
        logger.error(f"booking webhook failed: {e}")
        return False, "Sorry, I had trouble booking that — could we try again?"


class EvaAgent(Agent):
    def __init__(self, instructions: str, global_tts: inference.TTS,
                 meeting: dict, lead: dict, call_id: str, opening_line: str = "",
                 forced_lang: str = None, persona_name: str = ""):
        super().__init__(instructions=instructions)
        self._global_tts = global_tts
        self._meeting = meeting or {}
        self._lead = lead or {}
        self._call_id = call_id
        self._opening_line = opening_line
        self._forced_lang = forced_lang
        self._persona_name = persona_name

    async def on_enter(self) -> None:
        logger.info("on_enter: call_id=%r persona_name=%r greeting=%r",
                    self._call_id, self._persona_name, self._opening_line)
        try:
            if self._opening_line:
                # session.say() bypasses tts_node()/_speak() entirely, so
                # without this the opening line always played in whatever
                # language the TTS was initialized with (English) - ignoring
                # the agent's configured/forced language for every call.
                lang = self._forced_lang or _detect_lang(self._opening_line)
                try:
                    self._global_tts.update_options(target_language_code=_sarvam_lang_code(lang), pace=TTS_SPEED)
                except Exception:
                    logger.warning("sarvam TTS update_options failed on opening line, using TTS defaults")
                await self.session.say(self._opening_line, allow_interruptions=True)
            else:
                # No opening_line configured on this agent - fall back to an
                # LLM-generated greeting, but pin down exactly what it's
                # allowed to say so it can't invent a name (e.g. "Eva").
                name_hint = (
                    f"Introduce yourself as {self._persona_name}."
                    if self._persona_name else
                    "Do not state any name for yourself."
                )
                await self.session.generate_reply(
                    instructions=(
                        "Greet the caller warmly in ONE short sentence and ask how you "
                        f"can help today. {name_hint}"
                    )
                )
        except Exception:
            logger.exception("on_enter failed for call_id=%r - greeting was not spoken", self._call_id)

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        buffer = ""
        spoken_chars = 0
        capped = False

        async for chunk in text:
            buffer += chunk
            parts = SENTENCE_END_RE.split(buffer)
            complete, i = "", 0
            while i + 1 < len(parts):
                complete += parts[i] + parts[i + 1]
                i += 2
            buffer = parts[i] if i < len(parts) else ""

            sentence = complete.strip()
            if not sentence:
                continue
            has_tag = bool(BOOK_MEETING_RE.search(sentence))
            sentence = await self._handle_booking_tag(sentence)
            # skip empty / punctuation-only text (Sarvam 400s on it), and drop
            # anything past the cap - except a booking confirmation.
            if not sentence or not SPEAKABLE_RE.search(sentence) or (capped and not has_tag):
                continue
            async for frame in self._speak(sentence):
                yield frame
            spoken_chars += len(sentence)
            if spoken_chars >= MAX_SPOKEN_CHARS:
                capped = True

        tail = buffer.strip()
        if tail:
            has_tag = bool(BOOK_MEETING_RE.search(tail))
            tail = await self._handle_booking_tag(tail)
            if tail and SPEAKABLE_RE.search(tail) and (not capped or has_tag):
                async for frame in self._speak(tail):
                    yield frame

    async def _handle_booking_tag(self, sentence: str) -> str:
        """Strips a BOOK_MEETING: tag out of the spoken text and fires the
        webhook, same convention app.py's LLM loop used."""
        bm = BOOK_MEETING_RE.search(sentence)
        if not bm:
            return sentence
        requested_iso = f"{bm.group(1)}T{bm.group(2)}:00"
        confirmed, message = book_meeting_via_pravaah(self._meeting, self._lead, self._call_id, requested_iso)
        logger.info(f"[{self._call_id or 'test'}] booking requested={requested_iso} confirmed={confirmed}")
        remainder = BOOK_MEETING_RE.sub("", sentence).strip()
        # Speak the booking result instead of the raw tag.
        return (remainder + " " + message).strip() if remainder else message

    async def _speak(self, sentence: str):
        # Prefer the call's configured language over script-detection: once
        # Hindi replies are written in Hinglish (Roman script) there's no
        # Devanagari left for _detect_lang to key off, so it would silently
        # fall back to "en" every time and use the wrong voice/pronunciation.
        lang = self._forced_lang or _detect_lang(sentence)
        try:
            self._global_tts.update_options(target_language_code=_sarvam_lang_code(lang), pace=TTS_SPEED)
        except Exception:
            logger.warning("sarvam TTS update_options failed for %r, using TTS defaults", sentence[:30])
        logger.info("speaking [%s]: %s", LANG_NAMES.get(lang, lang), sentence[:60])
        async for audio in self._global_tts.synthesize(sentence):
            yield audio.frame



def _build_instructions(agent_cfg: dict, lead: dict, meeting: dict):
    """Mirrors app.py's EvaSession.__init__ prompt-building. Returns
    (instructions, forced_lang, persona_name) - the caller needs forced_lang
    separately to drive TTS language selection, and persona_name so the
    LLM-generated fallback greeting (when no opening_line is set) can't
    invent a name like "Eva"."""
    persona_name = (agent_cfg.get("name") or "").strip()
    custom_prompt = re.split(r"You can also book meetings on the account owner",
                             (agent_cfg.get("system_prompt") or ""))[0].strip()
    base = custom_prompt or (
        "You are a helpful, warm, concise voice assistant taking this call "
        "on behalf of the business. Keep replies short and conversational "
        "(1-3 sentences) since they will be spoken aloud."
    )
    # The agent's configured name (set in the Pravaah dashboard) is NOT
    # always repeated inside the owner's custom system prompt, so without
    # this the model has no idea what it's actually called and guesses -
    # commonly landing on "Eva". Always tell it explicitly.
    if persona_name:
        base += (
            f"\n\nYour name is {persona_name}. If the caller asks your name, "
            f"tell them your name is {persona_name} - never any other name."
        )
    if lead:
        base += (
            f"\n\nYou are speaking with {lead.get('name', 'the lead')} from "
            f"{lead.get('business_name', 'their business')}. Use their name naturally, don't overuse it."
        )
    forced_lang = agent_cfg.get("language") if agent_cfg.get("language") in SUPPORTED_LANGUAGES else None
    if forced_lang and forced_lang != "en":
        base += (
            f"\nAlways reply in casual, natural {LANG_NAMES.get(forced_lang, forced_lang)} written "
            "in Roman/English letters (Hinglish) - the way people actually type it day to day. "
            "NEVER use Devanagari or any native script, unless the user explicitly writes in English."
        )
    elif forced_lang == "en":
        base += "\nAlways reply in English only."
    else:
        base += (
            "\nLanguage rule: follow the persona's language style above. Default to casual "
            "Hinglish (Hindi written in Roman/English letters, never Devanagari). "
            "Only switch to full English if the caller clearly speaks only English."
        )
    # Same gender-agreement rule as app.py's EvaSession - keeps karta/karti
    # consistent with agent_cfg.gender (the same field that picks the voice
    # a few lines up in entrypoint()). Default matches the voice-selection
    # default (female) when gender isn't set.
    gender_forms = (
        "masculine (e.g. main karta hoon, main bol raha hoon, main tha)"
        if agent_cfg.get("gender") == "male" else
        "feminine (e.g. main karti hoon, main bol rahi hoon, main thi)"
    )
    base += (
        f"\nGRAMMATICAL GENDER: whenever you speak Hindi or Hinglish, always refer to "
        f"yourself using {gender_forms} verb forms. Stay consistent for the entire call - never switch."
    )
    base += "\nNever reply using only emojis or symbols with no words."
    name_rule = (
        f"your name is {persona_name} - always use exactly that name."
        if persona_name else
        "you have not been given a specific name - never invent one; if asked who you "
        "are, just say something like 'I'm calling from the team here' instead of "
        "stating a name."
    )
    base += (
        "\nSTAY IN CHARACTER (always follow, no exceptions): you are the persona "
        f"defined above, nothing else. IDENTITY RULE: {name_rule} Never say you are "
        "an AI, a bot, a language model, or an assistant. If asked what you are, who "
        "you are, or where you're calling/speaking from, answer briefly and naturally "
        "the way a real staff member would - never mention a cloud server, data center, "
        "or any technical/infrastructure detail about yourself."
    )
    base += (
        "\n\nSPEAKING LENGTH RULE (always follow, no exceptions): this is a live "
        "phone call. Each reply is ONE short sentence, at most two. Target 100-150 "
        "characters, never above 200. Never read the example lines in the script word "
        "for word - compress them to the shortest natural version. Acknowledge in 2-3 "
        "words, then ask ONE question. Never say 'Sir/Ma'am' out loud as a pair: use the "
        "caller's name with 'ji' (e.g. 'Bhuvi ji'), or just 'ji'. Keep any closing "
        "summary to one sentence."
    )
    base += (
        "\n\nSpeak the way a real person talks on a phone call - use contractions "
        "and everyday words, keep a warm relaxed tone, and avoid stiff, scripted, "
        "or overly formal phrasing. Don't sound robotic."
    )
    if meeting:
        base += (
            "\n\nYou can book a meeting for this lead. Meetings are "
            f"{meeting.get('duration_minutes', 30)} minutes long. "
            f"Available windows: {meeting.get('availability_text', '')}. "
            "Never ask for the lead's name/phone again - only ask for their preferred "
            "meeting date and time. Once confirmed, output EXACTLY one line: "
            "BOOK_MEETING: YYYY-MM-DD HH:MM (24-hour clock, UTC). Do not say this "
            "line out loud or explain it - it's processed automatically."
        )
    return base, forced_lang, persona_name

def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer()
server.setup_fnc = prewarm


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    logger.info(f"=== ENTRYPOINT CALLED for room {ctx.room.name}, job metadata: {ctx.job.metadata!r} ===")
    ctx.log_context_fields = {"room": ctx.room.name}

    # --- parse per-call context, if any (set by livekit_bridge.py for
    # VoiceLink calls; absent for plain browser/dev rooms) ---
    call_ctx = {}
    if ctx.job.metadata:
        try:
            call_ctx = json.loads(ctx.job.metadata)
        except Exception:
            logger.warning("job metadata was not valid JSON, ignoring: %r", ctx.job.metadata)

    agent_cfg = call_ctx.get("agent") or {}
    lead = call_ctx.get("lead") or {}
    meeting = call_ctx.get("meeting") or {}
    call_id = call_ctx.get("call_id")
    callback_url = call_ctx.get("callback_url")

    # DEBUG: confirms exactly what agent config this call actually received.
    # If agent_name here isn't the one you configured in the Pravaah
    # dashboard, the bug is upstream (Pravaah/Eva/dispatch), not in the LLM.
    logger.info(
        "call_ctx received: call_id=%r agent_name=%r opening_line=%r language=%r gender=%r has_system_prompt=%s",
        call_id,
        agent_cfg.get("name"),
        agent_cfg.get("opening_line"),
        agent_cfg.get("language"),
        agent_cfg.get("gender"),
        bool((agent_cfg.get("system_prompt") or "").strip()),
    )
    # logger.info from the job subprocess wasn't showing up in `docker logs`
    # last debug session - plain print() to stdout always gets through.
    print(f"[EVA-DEBUG] call_id={call_id} agent_name={agent_cfg.get('name')!r} "
          f"opening_line={agent_cfg.get('opening_line')!r} "
          f"has_system_prompt={bool((agent_cfg.get('system_prompt') or '').strip())}", flush=True)

    # FULL agent/lead/meeting payload exactly as this job received it, so
    # the entire Pravaah-sent config is visible in one place instead of
    # guessing which field is wrong.
    logger.info("call_ctx FULL agent for call_id=%r: %s", call_id, json.dumps(agent_cfg, default=str))
    logger.info("call_ctx FULL lead for call_id=%r: %s", call_id, json.dumps(lead, default=str))
    logger.info("call_ctx FULL meeting for call_id=%r: %s", call_id, json.dumps(meeting, default=str))
    print(f"[EVA-DEBUG] FULL agent for call_id={call_id}: {json.dumps(agent_cfg, default=str)}", flush=True)
    print(f"[EVA-DEBUG] FULL lead for call_id={call_id}: {json.dumps(lead, default=str)}", flush=True)
    print(f"[EVA-DEBUG] FULL meeting for call_id={call_id}: {json.dumps(meeting, default=str)}", flush=True)

    voice = VOICE_MALE if agent_cfg.get("gender") == "male" else VOICE_FEMALE
    global_tts = sarvam.TTS(
        target_language_code="en-IN",   # overridden per-sentence in EvaAgent._speak()/on_enter()
        speaker=voice,
        model=SARVAM_TTS_MODEL,
        pace=TTS_SPEED,
        temperature=SARVAM_TTS_TEMPERATURE,
    )
    try:
        global_tts.prewarm()   # open the Sarvam WebSocket now, not on the first sentence
    except Exception:
        pass
 
    if LLM_PROVIDER == "sarvam" and SARVAM_API_KEY:
        # Sarvam accepts "Authorization: Bearer <key>", so LiveKit's OpenAI
        # plugin works against its /v1 endpoint. max_tokens + reasoning_effort
        # go through extra_body; the inspect filter drops any kwarg your
        # installed livekit-agents version doesn't support.
        _llm_kwargs = dict(
            model=SARVAM_LLM_MODEL,
            api_key=SARVAM_API_KEY,
            base_url="https://api.sarvam.ai/v1",
            temperature=SARVAM_LLM_TEMPERATURE,
            extra_body={"max_tokens": SARVAM_LLM_MAX_TOKENS, "reasoning_effort": None},
        )
        _llm_valid = set(inspect.signature(openai.LLM.__init__).parameters)
        if "extra_body" not in _llm_valid:
            logger.warning("openai.LLM has no extra_body in this version - max_tokens/reasoning_effort NOT applied; "
                           "only the prompt limits reply length.")
        llm = openai.LLM(**{k: v for k, v in _llm_kwargs.items() if k in _llm_valid})
    elif LLM_PROVIDER == "cloudflare" and CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN:
        llm = openai.LLM(
            model=CLOUDFLARE_MODEL,
            api_key=CLOUDFLARE_API_TOKEN,
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1",
            temperature=CLOUDFLARE_TEMPERATURE,
        )
    elif LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
        logger.warning("LLM_PROVIDER=gemini requested but not wired up in agent.py yet — using Groq.")
        llm = groq.LLM(model=GROQ_MODEL, api_key=GROQ_API_KEY)
    else:
        llm = groq.LLM(model=GROQ_MODEL, api_key=GROQ_API_KEY)

    # Tuned for lower latency + more natural barge-in. min_endpointing_delay/
    # max_endpointing_delay/preemptive_generation/resume_false_interruption/
    # false_interruption_timeout are recent livekit-agents additions - the
    # inspect-based filter below drops any that your installed version
    # doesn't recognize, so this can't crash on an older build.
    _session_kwargs = dict(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=llm,
        tts=global_tts,
        turn_detection=MultilingualModel(),
        allow_interruptions=True,
        min_interruption_duration=0.4,
        min_endpointing_delay=0.2,
        max_endpointing_delay=3.0,
        preemptive_generation=True,       # start generating before the user's turn is fully finalized
        resume_false_interruption=True,   # resume speaking if a "barge-in" turns out to be noise
        false_interruption_timeout=1.5,
    )
    _valid_params = set(inspect.signature(AgentSession.__init__).parameters)
    _session_kwargs = {k: v for k, v in _session_kwargs.items() if k in _valid_params}
    session = AgentSession(**_session_kwargs)

    session.on("error", lambda ev: logger.error("SESSION ERROR: %r", getattr(ev, "error", ev)))
    session.on("agent_state_changed",
               lambda ev: logger.info("agent state: %s -> %s", ev.old_state, ev.new_state))
    session.on("user_input_transcribed",
               lambda ev: logger.info("USER SAID (final=%s): %s", ev.is_final, ev.transcript))

    transcript = []

    def _on_item_added(ev):
        # VERIFY against your installed livekit-agents version: event name
        # and payload shape for conversation items have changed across
        # releases. Check `session.on(...)` options if this doesn't fire.
        item = getattr(ev, "item", None)
        if not item:
            return
        role = "lead" if getattr(item, "role", "") == "user" else "agent"
        text = getattr(item, "text_content", None) or str(item)
        transcript.append({"role": role, "text": text, "ts": time.time()})

    session.on("conversation_item_added", _on_item_added)

    async def _finish_and_callback():
        for usage in session.usage.model_usage:
            logger.info("usage %s/%s: %s", usage.provider, usage.model, usage)
        if not (callback_url and call_id and EVA_API_SECRET):
            return
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    callback_url,
                    headers={"X-Eva-Secret": EVA_API_SECRET, "Content-Type": "application/json"},
                    json={
                        "call_id": call_id,
                        "status": "completed" if transcript else "no_response",
                        "hangup_reason": "completed",
                        "duration_secs": 0,  # TODO: track actual call duration
                        "transcript": transcript,
                    },
                    timeout=15,
                )
        except Exception as e:
            logger.error(f"callback POST failed for {call_id}: {e}")

    ctx.add_shutdown_callback(_finish_and_callback)

    instructions, forced_lang, persona_name = _build_instructions(agent_cfg, lead, meeting)
    opening_line = render_call_vars(agent_cfg.get("opening_line") or "", lead)
    opening_line = re.sub(r'[“”„«»"]', "", opening_line).strip()   # drop curly/straight double quotes
    logger.info("resolved persona_name=%r opening_line=%r (empty opening_line falls back to LLM greeting)",
                persona_name, opening_line)

    # Language used for TTS pronunciation. If Pravaah sends language="auto",
    # use Hindi (hi-IN), which reads Roman-script Hinglish naturally.
    # Speaker/voice is fixed at TTS creation, so the voice stays constant all call.
    tts_lang = forced_lang or "hi"

    agent = EvaAgent(
        instructions=instructions,
        global_tts=global_tts,
        meeting=meeting,
        lead=lead,
        call_id=call_id,
        opening_line=opening_line,
        forced_lang=tts_lang,
        persona_name=persona_name,
    )

    await session.start(
        agent=agent,
        room=ctx.room,
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(server)
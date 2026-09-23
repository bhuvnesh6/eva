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
from livekit.plugins import deepgram, groq, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eva-agent")

AGENT_NAME = os.environ.get("AGENT_NAME", "eva-agent")

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

EVA_API_SECRET = os.environ.get("EVA_API_SECRET", "")
PRAVAAH_API_BASE_URL = os.environ.get("PRAVAAH_API_BASE_URL", "").rstrip("/")

GLOBAL_TTS_MODEL = os.environ.get("GLOBAL_TTS_MODEL", "inworld/inworld-tts-2")
VOICE_MALE = os.environ.get("EVA_VOICE_MALE", "Manoj")
VOICE_FEMALE = os.environ.get("EVA_VOICE_FEMALE", "Riya")

SENTENCE_END_RE = re.compile(r"([.!?।\n])")
HINDI_RE = re.compile(r"[\u0900-\u097F]")
BOOK_MEETING_RE = re.compile(r"BOOK_MEETING:\s*(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})")
LANG_NAMES = {"en": "English", "hi": "Hindi"}
SUPPORTED_LANGUAGES = {"en", "hi"}


def _detect_lang(text: str) -> str:
    return "hi" if HINDI_RE.search(text) else "en"


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
                 meeting: dict, lead: dict, call_id: str, opening_line: str = ""):
        super().__init__(instructions=instructions)
        self._global_tts = global_tts
        self._meeting = meeting or {}
        self._lead = lead or {}
        self._call_id = call_id
        self._opening_line = opening_line

    async def on_enter(self) -> None:
        logger.info("on_enter: greeting=%r", self._opening_line)
        if self._opening_line:
            await self.session.say(self._opening_line, allow_interruptions=True)
        else:
            await self.session.generate_reply(
                instructions="Greet the caller warmly and ask how you can help today."
            )

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        buffer = ""
        async for chunk in text:
            buffer += chunk
            parts = SENTENCE_END_RE.split(buffer)
            complete, i = "", 0
            while i + 1 < len(parts):
                complete += parts[i] + parts[i + 1]
                i += 2
            buffer = parts[i] if i < len(parts) else ""

            sentence = complete.strip()
            if sentence:
                sentence = await self._handle_booking_tag(sentence)
                if sentence:
                    async for frame in self._speak(sentence):
                        yield frame

        tail = buffer.strip()
        if tail:
            tail = await self._handle_booking_tag(tail)
            if tail:
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
        lang = _detect_lang(sentence)
        self._global_tts.update_options(language=lang)
        logger.info("speaking [%s]: %s", LANG_NAMES.get(lang, lang), sentence[:60])
        async for audio in self._global_tts.synthesize(sentence):
            yield audio.frame


def _build_instructions(agent_cfg: dict, lead: dict, meeting: dict) -> str:
    """Mirrors app.py's EvaSession.__init__ prompt-building, minus the
    speaking-length rule wording tweaks - kept close to the original."""
    custom_prompt = (agent_cfg.get("system_prompt") or "").strip()
    base = custom_prompt or (
        "You are Eva, a helpful, concise, warm voice assistant. "
        "Keep replies short and conversational (1-3 sentences) since they "
        "will be spoken aloud."
    )
    if lead:
        base += (
            f"\n\nYou are speaking with {lead.get('name', 'the lead')} from "
            f"{lead.get('business_name', 'their business')}. Use their name naturally, don't overuse it."
        )
    forced_lang = agent_cfg.get("language") if agent_cfg.get("language") in SUPPORTED_LANGUAGES else None
    if forced_lang and forced_lang != "en":
        base += (
            f"\nAlways reply in {LANG_NAMES.get(forced_lang, forced_lang)}, written in its own "
            "native script (not romanized/Latin script), unless the user explicitly writes in English."
        )
    elif forced_lang == "en":
        base += "\nAlways reply in English only."
    else:
        base += (
            "\nLanguage rule: default to English. If the user is clearly speaking "
            "Hindi, reply in Hindi using Devanagari script (not romanized). "
            "If their message is in English, unclear, or mixed, reply in English."
        )
    base += "\nNever reply using only emojis or symbols with no words."
    base += (
        "\n\nSPEAKING LENGTH RULE (always follow, no exceptions): this is a live "
        "phone/voice conversation. Normally answer in ONE short sentence. At most "
        "2-3 short sentences for a normal question."
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
    return base


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
            logger.warning("job metadata was not valid JSON, ignoring")

    agent_cfg = call_ctx.get("agent") or {}
    lead = call_ctx.get("lead") or {}
    meeting = call_ctx.get("meeting") or {}
    call_id = call_ctx.get("call_id")
    callback_url = call_ctx.get("callback_url")

    voice = VOICE_MALE if agent_cfg.get("gender") == "male" else VOICE_FEMALE
    global_tts = inference.TTS(model=GLOBAL_TTS_MODEL, voice=voice, language="en")

    if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
        # NOTE: livekit.plugins doesn't ship a Gemini LLM plugin in the base
        # install used elsewhere in this project - if you need Gemini here
        # too, either add livekit-plugins-google or keep Groq for the
        # VoiceLink path only. Left as Groq below until confirmed.
        logger.warning("LLM_PROVIDER=gemini requested but not wired up in agent.py yet — using Groq.")
    llm = groq.LLM(model=GROQ_MODEL, api_key=GROQ_API_KEY)

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=llm,
        tts=global_tts,
        turn_detection=MultilingualModel(),
        allow_interruptions=True,
        min_interruption_duration=0.5,
    )

    session.on("error", lambda ev: logger.error("SESSION ERROR: %r", getattr(ev, "error", ev)))
    session.on("agent_state_changed",
               lambda ev: logger.info("agent state: %s -> %s", ev.old_state, ev.new_state))

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

    instructions = _build_instructions(agent_cfg, lead, meeting)
    opening_line = render_call_vars(agent_cfg.get("opening_line") or "", lead).strip().strip('"').strip()
    agent = EvaAgent(instructions=instructions, global_tts=global_tts,
                      meeting=meeting, lead=lead, call_id=call_id, opening_line=opening_line)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(server)
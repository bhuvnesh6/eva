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
import asyncio
import inspect
import json
import logging
import os
import re
import time
from typing import AsyncIterable

import io
import wave
import audioop   # py3.13+: pip install audioop-lts (same as Eva app.py)
import httpx
import requests
from dotenv import load_dotenv
from livekit import rtc

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
SARVAM_LLM_MAX_TOKENS = int(os.environ.get("SARVAM_LLM_MAX_TOKENS", "140"))

# ---------------- Sarvam (TTS) ----------------
SARVAM_TTS_MODEL = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v3")
SARVAM_TTS_TEMPERATURE = float(os.environ.get("SARVAM_TTS_TEMPERATURE", "0.6"))

EVA_API_SECRET = os.environ.get("EVA_API_SECRET", "")
PRAVAAH_API_BASE_URL = os.environ.get("PRAVAAH_API_BASE_URL", "").rstrip("/")

GLOBAL_TTS_MODEL = os.environ.get("GLOBAL_TTS_MODEL", "inworld/inworld-tts-2")
VOICE_MALE = os.environ.get("EVA_VOICE_MALE", "shubh")
VOICE_FEMALE = os.environ.get("EVA_VOICE_FEMALE", "priya")
TTS_SPEED = max(0.8, min(1.4, float(os.environ.get("EVA_TTS_SPEED", "1.4"))))

SENTENCE_END_RE = re.compile(r"([.!?।\n])")
HINDI_RE = re.compile(r"[\u0900-\u097F]")
BOOK_MEETING_RE = re.compile(r"BOOK_MEETING:\s*(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})")
END_CALL_RE = re.compile(r"\[?\s*END_CALL\s*\]?")
END_CALL_GRACE_SECS = float(os.environ.get("EVA_END_CALL_GRACE_SECS", "0.5"))
# Safety net: if the caller clearly says goodbye in a SHORT sentence, hang up after the agent's
# reply even when the LLM forgot to add END_CALL. ("tata" is left out on purpose - it's also a company.)
BYE_RE = re.compile(
    r"(?<![a-z])(bye|good ?bye|bye ?bye|alvida|that'?s all|that is all|phone rakh\w*)(?![a-z])"
    r"|बाय|अलविदा|रखता हूँ|रखती हूँ|रखता हूं|रखती हूं",
    re.I,
)
BYE_MAX_WORDS = 8
LANGUAGES = {
    "en":  {"name": "English",    "tts": "en-IN", "script": None},
    "hi":  {"name": "Hindi",      "tts": "hi-IN", "script": None},   # Hinglish (Roman) - unchanged behaviour
    "hr":  {"name": "Haryanvi",   "tts": "hi-IN", "script": "Devanagari", "dialect": True},
    "raj": {"name": "Rajasthani", "tts": "hi-IN", "script": "Devanagari", "dialect": True},
    "bho": {"name": "Bhojpuri",   "tts": "hi-IN", "script": "Devanagari", "dialect": True},
    "pa":  {"name": "Punjabi",    "tts": "pa-IN", "script": "Gurmukhi"},
    "gu":  {"name": "Gujarati",   "tts": "gu-IN", "script": "Gujarati"},
    "mr":  {"name": "Marathi",    "tts": "mr-IN", "script": "Devanagari"},
    "bn":  {"name": "Bengali",    "tts": "bn-IN", "script": "Bengali"},
    "ta":  {"name": "Tamil",      "tts": "ta-IN", "script": "Tamil"},
    "te":  {"name": "Telugu",     "tts": "te-IN", "script": "Telugu"},
    "kn":  {"name": "Kannada",    "tts": "kn-IN", "script": "Kannada"},
    "ml":  {"name": "Malayalam",  "tts": "ml-IN", "script": "Malayalam"},
    "od":  {"name": "Odia",       "tts": "od-IN", "script": "Odia"},
}
LANG_NAMES = {k: v["name"] for k, v in LANGUAGES.items()}
SUPPORTED_LANGUAGES = set(LANGUAGES)
SPEAKABLE_RE = re.compile(r"[A-Za-z0-9\u0900-\u0D7F]")   # Latin + all major Indic scripts
SCRIPT_LANG_RES = [
    (re.compile(r"[\u0900-\u097F]"), "hi"),
    (re.compile(r"[\u0980-\u09FF]"), "bn"),
    (re.compile(r"[\u0A00-\u0A7F]"), "pa"),
    (re.compile(r"[\u0A80-\u0AFF]"), "gu"),
    (re.compile(r"[\u0B00-\u0B7F]"), "od"),
    (re.compile(r"[\u0B80-\u0BFF]"), "ta"),
    (re.compile(r"[\u0C00-\u0C7F]"), "te"),
    (re.compile(r"[\u0C80-\u0CFF]"), "kn"),
    (re.compile(r"[\u0D00-\u0D7F]"), "ml"),
]
SARVAM_STT_MODEL = os.environ.get("SARVAM_STT_MODEL", "saarika:v2.5")
# "deepgram" (default) = auto-language calls use Deepgram multi. "sarvam" = auto calls use Sarvam
# language auto-detect, which understands all Indian languages.
AUTO_STT = os.environ.get("EVA_AUTO_STT", "deepgram").strip().lower()
# "1" = English/Hindi agents also use Sarvam STT with language auto-detect, so they can follow
# a caller who asks to switch to Tamil/Gujarati/etc. mid-call.
ALLOW_SWITCH_STT = os.environ.get("EVA_ALLOW_LANG_SWITCH_STT", "0") == "1"
# Hard ceiling on spoken chars per reply (~15 chars/sec of audio => 200 chars ~ 13s)
MAX_SPOKEN_CHARS = int(os.environ.get("EVA_MAX_SPOKEN_CHARS", "200"))

# ---------------- Latency tuning ----------------
# How long the agent waits after the caller stops before replying, worst case.
# Was hard-coded 3.0 -> this was the biggest source of the 4-7s delay.
MIN_ENDPOINTING_DELAY = float(os.environ.get("EVA_MIN_ENDPOINTING_DELAY", "0.2"))
MAX_ENDPOINTING_DELAY = float(os.environ.get("EVA_MAX_ENDPOINTING_DELAY", "1.2"))
# "multilingual" = smart turn-detector model (default). "vad" = plain silence-based,
# fastest, but may cut in on slow talkers. Try "vad" if still too slow.
TURN_DETECTION_MODE = os.environ.get("EVA_TURN_DETECTION", "multilingual").strip().lower()
# First spoken chunk may be flushed at a comma once it has this many chars,
# so TTS starts before the LLM finishes the whole first sentence.
FIRST_CLAUSE_RE = re.compile(r"[,;](?=\s)")
MIN_FIRST_CHUNK_CHARS = int(os.environ.get("EVA_MIN_FIRST_CHUNK_CHARS", "18"))

# Where this worker can reach Eva's Flask app (app.py) to hand over the final
# transcript. Same container -> http://127.0.0.1:8420 ; otherwise PUBLIC_BASE_URL.
EVA_INTERNAL_BASE_URL = (
    os.environ.get("EVA_INTERNAL_BASE_URL")
    or os.environ.get("PUBLIC_BASE_URL")
    or ("http://127.0.0.1:" + os.environ.get("PORT", "8420"))
).rstrip("/")

def _detect_lang(text: str) -> str:
    """Script-based: Devanagari -> hi, Gujarati script -> gu, Gurmukhi -> pa, ... else en."""
    for rx, code in SCRIPT_LANG_RES:
        if rx.search(text or ""):
            return code
    return "en"


def _pick_tts_lang(text: str, forced: str = None) -> str:
    """If the sentence is written in a specific Indic script, use that language's voice.
    Otherwise use the call's configured language (so Haryanvi/Marathi in Devanagari
    keep their own setting instead of collapsing to plain Hindi)."""
    detected = _detect_lang(text)
    if detected not in ("en", "hi"):
        return detected
    return forced or detected


def _reply_language_rule(lang: str) -> str:
    cfg = LANGUAGES.get(lang)
    if not cfg or lang == "en":
        return "Always reply in English only."
    name = cfg["name"]
    if not cfg["script"]:
        return (f"Always reply in casual, natural {name} written in Roman/English letters (Hinglish) - the way "
                "people actually type it day to day. NEVER use Devanagari or any native script, unless the user explicitly writes in English.")
    rule = (f"Always reply in casual, natural spoken {name}, written in {cfg['script']} script "
            "(never Roman letters), exactly how people talk on a phone call.")
    if cfg.get("dialect"):
        rule += f" Use real {name} words, grammar and tone - not standard textbook Hindi."
    return rule


def _build_stt(language):
    """Regional languages -> Sarvam STT (understands Indian languages). English/Hindi/auto -> Deepgram
    as before. If Sarvam STT can't be created, falls back to Deepgram so calls never fail."""
    lang = language if language in SUPPORTED_LANGUAGES else None
    regional = lang is not None and lang not in ("en", "hi")
    auto_like = lang is None or lang in ("en", "hi")
    if SARVAM_API_KEY and (regional or (auto_like and (AUTO_STT == "sarvam" or ALLOW_SWITCH_STT))):
        code = LANGUAGES[lang]["tts"] if regional else "unknown"
        try:
            logger.info("STT: using Sarvam (%s, model=%s)", code, SARVAM_STT_MODEL)
            return sarvam.STT(language=code, model=SARVAM_STT_MODEL)
        except Exception:
            logger.exception("sarvam.STT failed to initialise - falling back to Deepgram")
    return deepgram.STT(model="nova-3", language="multi")


# ---------------- Pre-recorded opening line ----------------
_OPENING_AUDIO_CACHE = {}   # (url, rate) -> mono PCM16 bytes


async def _prepare_opening_audio(url: str, target_rate: int):
    """Downloads the agent's pre-generated opening-line WAV (Cloudinary) and
    converts it to mono PCM16 at the TTS output rate. Started as a background
    task at the top of the call so it's ready before the greeting is needed."""
    if not url:
        return None
    key = (url, target_rate)
    cached = _OPENING_AUDIO_CACHE.get(key)
    if cached:
        return cached
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, timeout=8)
            r.raise_for_status()
        with wave.open(io.BytesIO(r.content), "rb") as wf:
            channels, width, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        if width != 2:
            frames = audioop.lin2lin(frames, width, 2)
        if channels > 1:
            frames = audioop.tomono(frames, 2, 0.5, 0.5)
        if rate != target_rate:
            frames, _ = audioop.ratecv(frames, 2, 1, rate, target_rate, None)
        if len(_OPENING_AUDIO_CACHE) > 50:
            _OPENING_AUDIO_CACHE.clear()
        _OPENING_AUDIO_CACHE[key] = frames
        return frames
    except Exception:
        logger.exception("could not load pre-recorded opening audio: %s", url)
        return None


async def _pcm_frames(pcm: bytes, rate: int, chunk_ms: int = 20):
    """Async generator of rtc.AudioFrame for session.say(audio=...)."""
    step = max(2, (rate * chunk_ms // 1000) * 2)
    for i in range(0, len(pcm), step):
        chunk = pcm[i:i + step]
        if len(chunk) % 2:
            chunk = chunk[:-1]
        if not chunk:
            continue
        yield rtc.AudioFrame(
            data=chunk, sample_rate=rate, num_channels=1,
            samples_per_channel=len(chunk) // 2,
        )


def _sarvam_lang_code(lang: str) -> str:
    """Internal language key -> Sarvam's BCP-47 target_language_code."""
    return (LANGUAGES.get(lang) or LANGUAGES["en"])["tts"]


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
                 forced_lang: str = None, persona_name: str = "",
                 opening_audio_task=None, opening_audio_text: str = "",
                 opening_audio_rate: int = 22050,
                 greeting_played: bool = False, greeting_text: str = ""):
        super().__init__(instructions=instructions)
        self._greeting_played = greeting_played     # inbound: Eva already played the greeting on the line
        self._greeting_text = greeting_text
        self._end_call_requested = False
        self._global_tts = global_tts
        self._meeting = meeting or {}
        self._lead = lead or {}
        self._call_id = call_id
        self._opening_line = opening_line
        self._forced_lang = forced_lang
        self._persona_name = persona_name
        self._opening_audio_task = opening_audio_task
        self._opening_audio_text = opening_audio_text
        self._opening_audio_rate = opening_audio_rate

    async def _remember_played_greeting(self) -> None:
        """Inbound: Eva already played the greeting to the caller. Don't speak it again,
        just add it to the chat context so the LLM knows it was said."""
        text = self._greeting_text or self._opening_audio_text or self._opening_line
        logger.info("greeting already played by Eva for call_id=%r (%r)", self._call_id, text)
        if not text:
            return
        try:
            chat_ctx = self.chat_ctx.copy()
            chat_ctx.add_message(role="assistant", content=text)
            await self.update_chat_ctx(chat_ctx)
        except Exception:
            logger.exception("could not add the played greeting to chat context")

    def _strip_end_call(self, text: str) -> str:
        """Removes the END_CALL token (never spoken) and remembers that the caller is hanging up."""
        if END_CALL_RE.search(text):
            self._end_call_requested = True
            text = END_CALL_RE.sub("", text).strip()
        return text


    async def _play_prerecorded_opening(self) -> bool:
        """Plays the pre-generated opening audio instantly. Interruptible like any
        other speech (user barge-in stops it and the normal AI flow continues).
        The text is added to chat context so the LLM knows it was already said.
        Returns False if unavailable so the caller falls back to live TTS."""
        if self._opening_audio_task is None:
            return False
        try:
            pcm = await asyncio.wait_for(asyncio.shield(self._opening_audio_task), timeout=3.0)
        except Exception:
            logger.warning("opening audio not ready in time, falling back to live TTS")
            return False
        if not pcm:
            return False
        try:
            await self.session.say(
                self._opening_audio_text or self._opening_line,
                audio=_pcm_frames(pcm, self._opening_audio_rate),
                allow_interruptions=True,
            )
            logger.info("pre-recorded opening line played for call_id=%r", self._call_id)
            return True
        except Exception:
            logger.exception("pre-recorded opening failed, falling back to live TTS")
            return False

    async def on_enter(self) -> None:
        logger.info("on_enter: call_id=%r persona_name=%r greeting=%r",
                    self._call_id, self._persona_name, self._opening_line)
        try:
            if self._greeting_played:
                await self._remember_played_greeting()
                return
            if await self._play_prerecorded_opening():
                return
            if self._opening_line:
                # session.say() bypasses tts_node()/_speak() entirely, so
                # without this the opening line always played in whatever
                # language the TTS was initialized with (English) - ignoring
                # the agent's configured/forced language for every call.
                lang = _pick_tts_lang(self._opening_line, self._forced_lang)
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
        finished = False

        try:
            async for chunk in text:
                buffer += chunk
                parts = SENTENCE_END_RE.split(buffer)
                complete, i = "", 0
                while i + 1 < len(parts):
                    complete += parts[i] + parts[i + 1]
                    i += 2
                buffer = parts[i] if i < len(parts) else ""

                sentence = complete.strip()

                # Early flush: for the FIRST chunk of a reply only, don't wait for a
                # full sentence - cut at the first comma once we have enough text.
                if not sentence and spoken_chars == 0:
                    for m in FIRST_CLAUSE_RE.finditer(buffer):
                        if m.end() >= MIN_FIRST_CHUNK_CHARS:
                            sentence = buffer[:m.end()].strip()
                            buffer = buffer[m.end():]
                            break

                if not sentence:
                    continue
                has_tag = bool(BOOK_MEETING_RE.search(sentence))
                sentence = await self._handle_booking_tag(sentence)
                sentence = self._strip_end_call(sentence)
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
                tail = self._strip_end_call(tail)
                if tail and SPEAKABLE_RE.search(tail) and (not capped or has_tag):
                    async for frame in self._speak(tail):
                        yield frame
            finished = True
        finally:
            if not finished:
                # reply was interrupted by the caller -> don't hang up on them
                self._end_call_requested = False

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
        lang = _pick_tts_lang(sentence, self._forced_lang)
        try:
            self._global_tts.update_options(target_language_code=_sarvam_lang_code(lang), pace=TTS_SPEED)
        except Exception:
            logger.warning("sarvam TTS update_options failed for %r, using TTS defaults", sentence[:30])
        logger.info("speaking [%s]: %s", LANG_NAMES.get(lang, lang), sentence[:60])
        async for audio in self._global_tts.synthesize(sentence):
            yield audio.frame



def _build_instructions(agent_cfg: dict, lead: dict, meeting: dict, inbound: bool = False):
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
    if lead and inbound:
        _nm = (lead.get("name") or "").strip()
        if _nm and not re.fullmatch(r"[+\d\s\-()]+", _nm):     # a bare phone number is not a name
            base += (
                f"\n\nThe caller is {_nm}"
                + (f" from {lead.get('business_name')}" if lead.get("business_name") else "")
                + ". Use their name naturally, don't overuse it."
            )
        else:
            base += "\n\nThe caller's name is not known yet - if it comes up naturally, politely ask for it."
    elif lead:
        base += (
            f"\n\nYou are speaking with {lead.get('name', 'the lead')} from "
            f"{lead.get('business_name', 'their business')}. Use their name naturally, don't overuse it."
        )
    forced_lang = agent_cfg.get("language") if agent_cfg.get("language") in SUPPORTED_LANGUAGES else None
    if forced_lang:
        base += (
            "\nLANGUAGE RULE (always follow): "
            + _reply_language_rule(forced_lang).replace("Always reply", "By default reply")
            + " EXCEPTION: if the caller explicitly asks you to speak in another language "
              "(for example 'please speak in Tamil'), switch to that language immediately, write in "
              "its own native script, and stay in it until they ask to change again."
        )
    else:
        base += (
            "\nLanguage rule: follow the persona's language style above. Default to casual "
            "Hinglish (Hindi written in Roman/English letters, never Devanagari). "
            "Only switch to full English if the caller clearly speaks only English. "
            "If the caller clearly speaks another Indian language, or asks you to speak in one "
            "(Gujarati, Punjabi, Marathi, Bengali, Tamil, Telugu, Kannada, Malayalam, Odia, Haryanvi...), "
            "switch to that language and reply in its own native script until they ask to change."
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
        f"\nGRAMMATICAL GENDER: whenever you speak Hindi, Hinglish or any other gendered Indian "
        f"language (Punjabi, Gujarati, Marathi, Haryanvi, etc.), always refer to yourself using the "
        f"{'masculine' if agent_cfg.get('gender') == 'male' else 'feminine'} form of verbs and adjectives, "
        f"e.g. for Hindi {gender_forms}. Stay consistent for the entire call - never switch."
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
    if inbound:
        base += (
            "\n\nINBOUND CALL: the person phoned the business - you did NOT call them. Never say you are "
            "calling them and never ask 'do you have a minute'. The greeting was already said at the start "
            "of the call, so do NOT greet again or re-introduce yourself; just respond to what they say and help them."
        )
    base += (
        "\n\nENDING THE CALL: when the caller clearly wants to finish the conversation (for example 'ok bye', "
        "'bye', 'thank you bye', 'that's all', 'theek hai bye', 'rakhta hoon', 'phone rakhti hoon', 'dhanyavaad bye'), "
        "reply with ONE short polite goodbye sentence and add the exact token END_CALL at the very end of that same reply. "
        "Never say END_CALL out loud or explain it. A plain 'okay', 'hmm', 'haan' or 'theek hai' on its own is NOT a "
        "goodbye - only use END_CALL when the caller is really ending the call."
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
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=0.35)


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
    inbound = bool(call_ctx.get("inbound"))
    greeting_played = bool(call_ctx.get("greeting_played"))     # Eva already played it on the line
    greeting_text = (call_ctx.get("greeting_text") or "").strip()

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

    # Pre-recorded opening line: start downloading NOW (in parallel with session
    # setup) so it's ready the instant on_enter runs.
    opening_audio_url = (agent_cfg.get("opening_audio_url") or "").strip()
    if greeting_played:
        opening_audio_url = ""
    opening_audio_text = (agent_cfg.get("opening_audio_text") or "").strip()
    opening_audio_rate = int(getattr(global_tts, "sample_rate", 22050) or 22050)
    opening_audio_task = (
        asyncio.create_task(_prepare_opening_audio(opening_audio_url, opening_audio_rate))
        if opening_audio_url else None
    )
 
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
        stt=_build_stt(agent_cfg.get("language")),
        llm=llm,
        tts=global_tts,
        turn_detection=("vad" if TURN_DETECTION_MODE == "vad" else MultilingualModel()),
        allow_interruptions=True,
        min_interruption_duration=0.4,
        min_endpointing_delay=MIN_ENDPOINTING_DELAY,
        max_endpointing_delay=MAX_ENDPOINTING_DELAY,
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
    _pending_interim = {"text": "", "t": 0.0}

    def _on_user_transcribed(ev):
        logger.info("USER SAID (final=%s): %s", ev.is_final, ev.transcript)
        if ev.is_final:
            _pending_interim["text"] = ""
        else:
            _pending_interim["text"] = ev.transcript
            _pending_interim["t"] = time.time()

    session.on("user_input_transcribed", _on_user_transcribed)

    def _log_latency(ev):
        # Look for: end_of_utterance_delay (turn wait), ttft (LLM first token),
        # ttfb (TTS first byte). Whichever is biggest is what to fix next.
        try:
            m = ev.metrics
            if type(m).__name__ == "VADMetrics":
                return          # one every second - just noise
            logger.info("LATENCY %s %s", type(m).__name__, m)
            print(f"[EVA-LATENCY] {type(m).__name__} {m}", flush=True)
        except Exception:
            pass

    session.on("metrics_collected", _log_latency)

    transcript = []
    call_started_at = time.time()
    max_duration_secs = int(agent_cfg.get("max_duration_secs") or 0) or None

    def _on_item_added(ev):
        # VERIFY against your installed livekit-agents version: event name
        # and payload shape for conversation items have changed across
        # releases. Check `session.on(...)` options if this doesn't fire.
        item = getattr(ev, "item", None)
        if not item:
            return
        role = "lead" if getattr(item, "role", "") == "user" else "agent"
        text = getattr(item, "text_content", None) or str(item)
        text = END_CALL_RE.sub("", text).strip()
        transcript.append({"role": role, "text": text, "ts": time.time()})

    session.on("conversation_item_added", _on_item_added)

    async def _finish_and_callback():
        for usage in session.usage.model_usage:
            logger.info("usage %s/%s: %s", usage.provider, usage.model, usage)
        if not (call_id and EVA_API_SECRET):
            return

        headers = {"X-Eva-Secret": EVA_API_SECRET, "Content-Type": "application/json"}
        result = {
            "call_id": call_id,
            "callback_url": callback_url,
            "status": "completed" if transcript else "no_response",
            "hangup_reason": "completed",
            "duration_secs": round(time.time() - call_started_at, 1),
            "transcript": transcript,
        }

        # 1) Preferred: hand the transcript to Eva's Flask app. It holds it until
        #    VoiceLink's call.completed webhook brings the recording URL, then
        #    sends ONE callback to Pravaah with transcript + recording_url.
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    f"{EVA_INTERNAL_BASE_URL}/api/internal/call-result/{call_id}",
                    headers=headers, json=result, timeout=10,
                )
            if r.status_code < 300:
                logger.info("transcript handed to Eva app for call_id=%r", call_id)
                return
            logger.error("internal call-result HTTP %s: %s", r.status_code, r.text[:200])
        except Exception as e:
            logger.error(f"internal call-result POST failed for {call_id}: {e}")

        # 2) Fallback: Flask unreachable -> post straight to Pravaah (no recording url).
        if not callback_url:
            return
        try:
            async with httpx.AsyncClient() as client:
                await client.post(callback_url, headers=headers, json=result, timeout=15)
        except Exception as e:
            logger.error(f"callback POST failed for {call_id}: {e}")

    ctx.add_shutdown_callback(_finish_and_callback)

    instructions, forced_lang, persona_name = _build_instructions(agent_cfg, lead, meeting, inbound=inbound)
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
        opening_audio_task=opening_audio_task,
        opening_audio_text=opening_audio_text,
        opening_audio_rate=opening_audio_rate,
        greeting_played=greeting_played,
        greeting_text=greeting_text,
    )

    # Caller said bye -> the LLM added END_CALL -> once the goodbye has finished playing,
    # leave the room. The bridge sees the agent leave and cuts the VoiceLink call.
    _hangup_state = {"started": False}

    async def _end_call_after_goodbye():
        await asyncio.sleep(END_CALL_GRACE_SECS)
        logger.info("caller said goodbye -> ending call_id=%r", call_id)
        ctx.shutdown(reason="caller_goodbye")

    def _on_state_for_hangup(ev):
        if (str(ev.new_state) == "listening" and agent._end_call_requested
                and not _hangup_state["started"]):
            _hangup_state["started"] = True
            asyncio.create_task(_end_call_after_goodbye())

    session.on("agent_state_changed", _on_state_for_hangup)

    async def _bye_fallback():
        # let the turn start, wait for the agent's goodbye reply to finish, then leave
        await asyncio.sleep(0.8)
        t0 = time.time()
        while time.time() - t0 < 8 and str(session.agent_state) not in ("thinking", "speaking"):
            await asyncio.sleep(0.1)
        while time.time() - t0 < 25 and str(session.agent_state) in ("thinking", "speaking"):
            await asyncio.sleep(0.15)
        if _hangup_state["started"]:
            return                      # the normal END_CALL path already handled it
        _hangup_state["started"] = True
        await asyncio.sleep(END_CALL_GRACE_SECS)
        logger.info("goodbye detected in caller transcript -> ending call_id=%r", call_id)
        ctx.shutdown(reason="caller_goodbye_fallback")

    def _on_user_bye(ev):
        if not ev.is_final or _hangup_state["started"]:
            return
        text = (ev.transcript or "").strip()
        if text and len(text.split()) <= BYE_MAX_WORDS and BYE_RE.search(text):
            logger.info("goodbye phrase heard: %r", text)
            asyncio.create_task(_bye_fallback())

    session.on("user_input_transcribed", _on_user_bye)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    async def _stuck_turn_watchdog():
        """If Deepgram never sends a final transcript, force the turn through
        after 2s of silence instead of leaving the caller in dead air."""
        while True:
            await asyncio.sleep(0.5)
            text = _pending_interim["text"]
            if not text or time.time() - _pending_interim["t"] < 2.0:
                continue
            if str(getattr(session, "user_state", "listening")) == "speaking":
                continue
            if str(getattr(session, "agent_state", "listening")) != "listening":
                continue
            _pending_interim["text"] = ""
            logger.warning("STUCK TURN: forcing reply for interim=%r", text)
            try:
                if hasattr(session, "commit_user_turn"):
                    session.commit_user_turn()
                else:
                    session.generate_reply(user_input=text)
            except Exception:
                logger.exception("stuck-turn recovery failed")

    asyncio.create_task(_stuck_turn_watchdog())

    if max_duration_secs:
        async def _max_duration_watchdog():
            await asyncio.sleep(max_duration_secs)
            logger.info("call_id=%r hit max duration (%ss), closing.", call_id, max_duration_secs)
            try:
                await session.say("Thank you ji, aapse baat karke achha laga.", allow_interruptions=False)
            except Exception:
                pass
            ctx.shutdown(reason="max_duration_reached")

        _watchdog_task = asyncio.create_task(_max_duration_watchdog())


if __name__ == "__main__":
    cli.run_app(server)
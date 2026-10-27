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
from livekit.plugins import groq, openai, sarvam, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import base64
import collections
import websockets                                  # pip install -U websockets
from sarvamai import AsyncSarvamAI, RealtimeAudioInput, RealtimeEnd   # pip install -U sarvamai
from livekit.agents import stt, tts, utils, APIConnectionError, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.types import NOT_GIVEN, NotGivenOr

load_dotenv()
from settings import get as S      # all Sarvam STT/TTS/LLM tuning lives in settings.json

logger = logging.getLogger("eva-agent")
logger.setLevel(logging.INFO)

AGENT_NAME = os.environ.get("AGENT_NAME", "eva-agent")

LLM_PROVIDER = str(S("llm.provider", "sarvam")).strip().lower()
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
SARVAM_LLM_MODEL = S("llm.model", "sarvam-105b-conversations")
SARVAM_LLM_TEMPERATURE = float(S("llm.temperature", 0.5))
SARVAM_LLM_MAX_TOKENS = int(S("llm.max_tokens", 200))
SARVAM_LLM_REASONING_EFFORT = S("llm.reasoning_effort", None)

# ---------------- Sarvam (TTS) ----------------
SARVAM_TTS_MODEL = S("tts.model", "bulbul:v3")
SARVAM_TTS_TEMPERATURE = float(S("tts.temperature", 0.6))

EVA_API_SECRET = os.environ.get("EVA_API_SECRET", "")
PRAVAAH_API_BASE_URL = os.environ.get("PRAVAAH_API_BASE_URL", "").rstrip("/")

GLOBAL_TTS_MODEL = os.environ.get("GLOBAL_TTS_MODEL", "inworld/inworld-tts-2")
VOICE_MALE = S("tts.voice_male", "shubh")
VOICE_FEMALE = S("tts.voice_female", "priya")
TTS_SPEED = max(0.8, min(1.4, float(S("tts.speed", 1.0))))

# Sentence end = . ! ? । or newline, but NOT the dot inside numbers ("1.5 crore", "2.5 BHK"),
# which used to cut speech in the middle of a number and caused an audible break.
SENTENCE_END_RE = re.compile(r"([!?।\n]|\.(?!\d)|(?<!\d)\.)")
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
SARVAM_STT_MODEL = S("stt.batch_model", "saaras:v3")
SARVAM_STT_MODE = S("stt.mode", "transcribe")   # transcribe | codemix | verbatim | translit | translate
MAX_SPOKEN_CHARS = int(S("tts.max_spoken_chars", 260))

# ---------------- Sarvam REALTIME (websocket) STT / TTS ----------------
SARVAM_STT_RT_MODEL = S("stt.realtime_model", "saaras:v3-realtime")
SARVAM_STT_STREAM_TYPE = S("stt.stream_type", "balanced")
SARVAM_STT_LANGUAGE_OVERRIDE = str(S("stt.language_override", "") or "").strip()
SARVAM_STT_CHUNK_MS = int(S("stt.chunk_ms", 50))
SARVAM_TTS_WS_URL = "wss://api.sarvam.ai/text-to-speech/ws?model={model}&send_completion_event=true"
SARVAM_TTS_SAMPLE_RATE = int(S("tts.sample_rate", 16000))

# ---------------- Sarvam VOICE CLONE (websocket) ----------------
SARVAM_CLONE_WS_URL = "wss://api.sarvam.ai/voices/clone/ws?send_completion_event=true"
CLONE_TTS_CODEC = S("clone.codec", "linear16")
CLONE_SEND_SAMPLE_RATE = bool(S("clone.send_sample_rate", True))
CLONE_OUTPUT_RATE = int(S("clone.output_rate", 0) or 0)            # 0 = same as requested rate
CLONE_MIN_FIRST_CHUNK_CHARS = int(S("clone.min_first_chunk_chars", 45))
CLONE_BATCH_MIN_CHARS = int(S("clone.batch_min_chars", 140))
CLONE_EARLY_COMMA = bool(S("clone.early_comma_flush", False))
CLONE_TAIL_TIMEOUT_SECS = float(S("clone.tail_timeout_secs", 0.8))      # short tail once all expected audio has arrived
CLONE_GAP_TIMEOUT_SECS = float(S("clone.gap_timeout_secs", 6.0))        # patience if audio is clearly still missing (slow clone chunk)
CLONE_CHARS_PER_SEC = float(S("clone.chars_per_sec", 15))               # used to estimate how much audio to expect
CLONE_PREROLL_SECS = float(S("clone.preroll_secs", 0.35))               # audio held back before playback starts (clone generates ~1.25x real time)
CLONE_PREROLL_MAX_WAIT_SECS = float(S("clone.preroll_max_wait_secs", 0.8))
# Clone websocket drops a 2nd text+flush sent while the 1st is still synthesizing -> send the whole reply as ONE text
CLONE_SINGLE_SHOT = bool(S("clone.single_shot", True))
CLONE_DEVANAGARI_HINDI = bool(S("clone.devanagari_hindi", True))
CLONE_EXTRA_CONFIG = S("clone.extra_config", {}) or {}
# extra keys merged into the NON-clone Sarvam TTS config (settings.json -> tts.extra_config)
SARVAM_TTS_EXTRA_CONFIG = S("tts.extra_config", {}) or {}
TTS_KEEPER_INTERVAL_SECS = float(S("tts.keeper_interval_secs", 3.0))
TTS_TIMING_LOG = bool(S("tts.timing_log", True))
FIRST_CHUNK_SPLIT_MIN_CHARS = int(S("tts.first_chunk_split_min_chars", 48))
LLM_EMPTY_RETRY = bool(S("llm.empty_retry", True))

# ---------------- background-noise / side-talk protection ----------------
VAD_ACTIVATION_THRESHOLD = float(os.environ.get("EVA_VAD_ACTIVATION", "0.6"))     # silero default 0.5
VAD_MIN_SPEECH_SECS = float(os.environ.get("EVA_VAD_MIN_SPEECH", "0.15"))
MIN_INTERRUPTION_SECS = float(os.environ.get("EVA_MIN_INTERRUPTION_SECS", "0.45"))
MIN_INTERRUPTION_WORDS = int(os.environ.get("EVA_MIN_INTERRUPTION_WORDS", "2"))
IGNORE_RE = re.compile(r"\[IGNORE\]")       # LLM emits this when the "caller" text is just side-talk

# ---------------- Latency tuning ----------------
# How long the agent waits after the caller stops before replying, worst case.
# Was hard-coded 3.0 -> this was the biggest source of the 4-7s delay.
MIN_ENDPOINTING_DELAY = float(os.environ.get("EVA_MIN_ENDPOINTING_DELAY", "0.05"))
MAX_ENDPOINTING_DELAY = float(os.environ.get("EVA_MAX_ENDPOINTING_DELAY", "1.2"))
# "multilingual" = smart turn-detector model (default). "vad" = plain silence-based,
# fastest, but may cut in on slow talkers. Try "vad" if still too slow.
TURN_DETECTION_MODE = os.environ.get("EVA_TURN_DETECTION", "stt").strip().lower()
# First spoken chunk may be flushed at a comma once it has this many chars,
# so TTS starts before the LLM finishes the whole first sentence.
FIRST_CLAUSE_RE = re.compile(r"[,;](?=\s)")
MIN_FIRST_CHUNK_CHARS = int(S("tts.min_first_chunk_chars", 14))
# The FIRST chunk of a reply is sent to TTS immediately (>= MIN_FIRST_CHUNK_CHARS) so audio starts fast.
# After that, sentences are merged into one Sarvam TTS request until this many chars (smooth intonation, no gaps).
TTS_BATCH_MIN_CHARS = int(S("tts.batch_min_chars", 60))
# English agents speak Indian-style English with Hindi words mixed in (Hinglish). Sarvam's hi-IN voice reads
# that mix far more naturally than en-IN, and using hi-IN from the start also avoids a TTS socket
# reconnect on the first sentence. Set EVA_EN_MIX_HINDI_TTS=0 to go back to en-IN.
EN_MIX_AS_HINDI = bool(S("tts.en_mix_hindi", True))

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


# Placed LAST in the system prompt so it outranks any Roman-script example lines in the owner's custom script.
HINDI_SCRIPT_RULE = (
    "HINDI SCRIPT RULE (highest priority, overrides every example and instruction above): whenever you speak "
    "Hindi, talk the way ordinary urban Indians talk on a phone call: Hindi with everyday English words mixed "
    "into the middle of the sentence. Write every HINDI word in Devanagari and every ENGLISH word in Latin "
    "letters, together in the same sentence. NEVER write a Hindi word in Roman letters - not 'ji', 'achha', "
    "'theek hai', 'aap', 'kab tak', 'humari', 'karegi', 'hain', 'boliye', 'dobara'. "
    "Right: 'ठीक है जी, आप कब तक property खरीदना चाहेंगे?' / 'हमारी senior team आपसे जल्दी ही contact करेगी।' "
    "Wrong: 'Theek hai ji, aap kab tak property kharidna chahenge?' / 'Humari senior team aapse jald hi contact karegi.' "
    "Right: 'ठीक है जी, note कर लिया।' Wrong: 'ठीक है ji, note कर लिया।' (half-Roman Hindi breaks the voice). "
    "Keep English words, brand and place names (property, budget, flat, BHK, site visit, confirm, team, "
    "Nisha Homes, thank you) in Latin letters exactly as spelled; never convert them to Devanagari. "
    "Write amounts and times like '1.5 करोड़' and 'कल शाम 5 बजे'. "
    "If the caller speaks only English, answer in plain English (Latin letters) and switch back to this style "
    "as soon as they use Hindi."
)


def _reply_language_rule(lang: str) -> str:
    cfg = LANGUAGES.get(lang)
    if not cfg or lang == "en":
        return (
            "Always reply in natural spoken Indian English, the way educated Indians really talk on a phone call: "
            "mostly English with everyday Hindi words and short phrases mixed in (Hinglish), for example "
            "'Ji bilkul, main check karke batata hoon.' or 'Achha, so budget kitna soch rakha hai?'. "
            "Write Hindi words in Roman letters only, NEVER Devanagari. If the caller speaks mostly English, keep it "
            "mostly English with just a light Hindi touch (ji, achha, bilkul, theek hai). If the caller speaks mostly "
            "Hindi, reply mostly in Hindi. Mirror the caller."
        )
    name = cfg["name"]
    if not cfg["script"]:
        return (
            f"Always reply in casual, natural spoken {name} the way urban Indians talk on the phone: Hindi with "
            "everyday English words mixed into the sentence. Hindi words in Devanagari script, English words in "
            "Latin letters. NEVER write Hindi words in Roman letters."
        )
    rule = (f"Always reply in casual, natural spoken {name}, written in {cfg['script']} script "
            "(never Roman letters), exactly how people talk on a phone call.")
    if cfg.get("dialect"):
        rule += f" Use real {name} words, grammar and tone - not standard textbook Hindi."
    return rule


# ======================================================================
#  Sarvam REALTIME streaming STT  (saaras:v3-realtime, websocket)
# ======================================================================
_FILLER_WORDS = {
    "hmm", "hm", "hmmm", "mm", "mmm", "um", "umm", "uh", "uhh", "ah", "aah", "oh", "huh",
    "हम", "हम्म", "हूँ", "हूं", "उं", "उम", "ऊं", "आह", "अं", "अह",
}


def _allowed_script_keys(lang):
    """Scripts we accept in a transcript (Latin is always accepted). 'hi' == Devanagari."""
    if lang in (None, "en", "hi", "hr", "raj", "bho", "mr"):
        return {"hi"}
    return {lang}


def _clean_stt_text(text: str, allowed_keys: set) -> str:
    """Kills hallucinations: removes characters from scripts this call can never use
    (Telugu/Bengali/Gujarati garbage from noise), and drops filler-only utterances."""
    t = (text or "").strip()
    if not t:
        return ""
    for rx, key in SCRIPT_LANG_RES:
        if key not in allowed_keys:
            t = rx.sub("", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    if not t or not SPEAKABLE_RE.search(t):
        return ""
    words = [w for w in re.split(r"[\s,.!?।]+", t.lower()) if w]
    if words and len(words) <= 2 and all(w in _FILLER_WORDS for w in words):
        return ""
    return t


class SarvamRealtimeSTT(stt.STT):
    def __init__(self, *, language_code: str, agent_language: str = None):
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        self._language_code = language_code
        self._allowed = _allowed_script_keys(agent_language)
        self._client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)

    @property
    def model(self) -> str:
        return SARVAM_STT_RT_MODEL

    @property
    def provider(self) -> str:
        return "Sarvam"

    async def _recognize_impl(self, buffer, *, language: NotGivenOr[str] = NOT_GIVEN,
                              conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS):
        raise NotImplementedError("SarvamRealtimeSTT is streaming-only")

    def stream(self, *, language: NotGivenOr[str] = NOT_GIVEN,
               conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> "SarvamRealtimeStream":
        return SarvamRealtimeStream(stt_obj=self, conn_options=conn_options)


class SarvamRealtimeStream(stt.RecognizeStream):
    def __init__(self, *, stt_obj: SarvamRealtimeSTT, conn_options: APIConnectOptions):
        super().__init__(stt=stt_obj, conn_options=conn_options, sample_rate=16000)
        self._speaking = False
        self._ended = False
        self._chunk_bytes = max(640, int(16000 * 2 * SARVAM_STT_CHUNK_MS / 1000))
        # last ~400ms of audio, replayed if the server closes the socket and we have to reconnect
        self._preroll = collections.deque(maxlen=max(2, 400 // max(SARVAM_STT_CHUNK_MS, 10)))

    # ---------- events out ----------
    def _emit(self, ev_type, text: str = "", lang: str = "hi"):
        alts = [stt.SpeechData(language=lang, text=text)] if text else []
        self._event_ch.send_nowait(stt.SpeechEvent(type=ev_type, alternatives=alts))

    def _handle_text(self, message, final: bool):
        raw = getattr(message, "text", "") or ""
        text = _clean_stt_text(raw, self._stt._allowed)
        if raw.strip() and not text:
            logger.info("STT dropped noise/hallucination (final=%s): %r", final, raw)
        if not text:
            if final and self._speaking:
                self._speaking = False
                self._emit(stt.SpeechEventType.END_OF_SPEECH)
            return
        lang = (getattr(message, "language_code", None) or self._stt._language_code or "hi")
        lang = "hi" if lang == "unknown" else lang.split("-")[0]
        if not self._speaking:
            self._speaking = True
            self._emit(stt.SpeechEventType.START_OF_SPEECH)
        if final:
            self._emit(stt.SpeechEventType.FINAL_TRANSCRIPT, text, lang)
            self._speaking = False
            self._emit(stt.SpeechEventType.END_OF_SPEECH)
        else:
            self._emit(stt.SpeechEventType.INTERIM_TRANSCRIPT, text, lang)

    # ---------- websocket loops ----------
    async def _send_chunk(self, ws, chunk: bytes):
        await ws.send_realtime_audio_input(
            RealtimeAudioInput(audio=base64.b64encode(chunk).decode("utf-8"))
        )

    async def _send_loop(self, ws):
        for c in list(self._preroll):          # after a reconnect: replay the last ~400ms
            await self._send_chunk(ws, c)
        buf = bytearray()
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                if buf:
                    chunk = bytes(buf); buf.clear()
                    self._preroll.append(chunk)
                    await self._send_chunk(ws, chunk)
                continue
            buf.extend(bytes(data.data))
            if len(buf) >= self._chunk_bytes:
                chunk = bytes(buf); buf.clear()
                self._preroll.append(chunk)
                await self._send_chunk(ws, chunk)
        if buf:
            await self._send_chunk(ws, bytes(buf))
        self._ended = True
        await ws.send_realtime_end(RealtimeEnd())

    async def _recv_loop(self, ws):
        async for message in ws:
            ev = getattr(message, "event", "")
            if ev == "transcript.partial":
                self._handle_text(message, final=False)
            elif ev == "transcript.final":
                self._handle_text(message, final=True)
            elif ev == "error":
                code = getattr(message, "code", None)
                msg = getattr(message, "message", "")
                logger.error("Sarvam STT error (%s): %s", code, msg)
                if getattr(message, "is_fatal", False):
                    raise APIConnectionError(f"Sarvam STT fatal error {code}: {msg}")
            else:
                logger.info("Sarvam STT other event: %s", ev)    # tells us the real VAD/event names if any

    async def _run(self) -> None:
        o = self._stt
        quick_fail = 0
        while not self._ended:
            t0 = time.time()
            try:
                async with o._client.speech_to_text_realtime_streaming.connect(
                    language_code=o._language_code,
                    model=SARVAM_STT_RT_MODEL,
                    mode=SARVAM_STT_MODE,
                    stream_type=SARVAM_STT_STREAM_TYPE,
                ) as ws:
                    logger.info("Sarvam STT connected (%s, %s, lang=%s)",
                                SARVAM_STT_RT_MODEL, SARVAM_STT_STREAM_TYPE, o._language_code)
                    send = asyncio.create_task(self._send_loop(ws))
                    recv = asyncio.create_task(self._recv_loop(ws))
                    try:
                        done, _ = await asyncio.wait({send, recv}, return_when=asyncio.FIRST_COMPLETED)
                        for t in done:
                            t.result()                      # surface errors
                        if send in done and not recv.done():
                            try:                            # input finished: let the last final arrive
                                await asyncio.wait_for(asyncio.shield(recv), 2.0)
                            except Exception:
                                pass
                    finally:
                        await utils.aio.gracefully_cancel(*[t for t in (send, recv) if not t.done()])
            except (APIConnectionError, asyncio.CancelledError):
                raise
            except Exception as e:
                raise APIConnectionError(f"Sarvam STT connection failed: {e!r}") from e

            if self._ended:
                break
            # server closed the socket while the call is still going -> reconnect
            if time.time() - t0 < 1.0:
                quick_fail += 1
                if quick_fail >= 5:
                    raise APIConnectionError("Sarvam STT socket keeps closing immediately")
                await asyncio.sleep(0.2)
            else:
                quick_fail = 0
            logger.info("Sarvam STT socket closed by server - reconnecting")


# ======================================================================
#  Sarvam streaming TTS over a persistent websocket
# ======================================================================
class SarvamWSTTS(tts.TTS):
    def __init__(self, *, api_key: str, model: str, speaker: str, language_code: str = "hi-IN",
                 pace: float = 1.0, temperature: float = 0.6, sample_rate: int = 16000,
                 voice_id: str = ""):
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False),
                         sample_rate=sample_rate, num_channels=1)
        self._api_key = api_key
        self._voice_id = (voice_id or "").strip()     # Sarvam cloned voice (svc-...) -> uses the clone websocket
        self._model = model
        self._speaker = speaker
        self._language_code = language_code
        self._pace = pace
        self._temperature = temperature
        self._sr = sample_rate
        self._ws = None
        self._ws_key = None
        self.lock = asyncio.Lock()      # one sentence on the socket at a time (FIFO => spoken order kept)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "Sarvam"

    @property
    def is_clone(self) -> bool:
        return bool(self._voice_id)

    def update_options(self, *, target_language_code: str = None, pace: float = None,
                       speaker: str = None, **_ignored):
        if target_language_code:
            self._language_code = target_language_code
        if pace:
            self._pace = pace
        if speaker:
            self._speaker = speaker

    def prewarm(self):
        async def _warm():
            try:
                async with self.lock:
                    await self.get_ws(self._language_code, self._pace)
            except Exception as e:
                logger.warning("Sarvam TTS prewarm failed: %r", e)
        asyncio.ensure_future(_warm())
        # keeper: background loop that reconnects a dropped socket BEFORE the next reply needs it
        if getattr(self, "_keeper_task", None) is None or self._keeper_task.done():
            self._keeper_task = asyncio.ensure_future(self._keeper())

    def ensure_warm(self):
        ws = self._ws
        if ws is not None and getattr(ws, "close_code", None) is None:
            return
        self.prewarm()

    async def _keeper(self):
        while True:
            await asyncio.sleep(TTS_KEEPER_INTERVAL_SECS)
            try:
                ws = self._ws
                alive = ws is not None and getattr(ws, "close_code", None) is None
                if alive or self.lock.locked():
                    continue
                async with self.lock:
                    await self.get_ws(self._language_code, self._pace)
                logger.info("TTS keeper: socket was down, reconnected in background")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("TTS keeper reconnect failed: %r", e)

    def drop_ws(self):
        ws, self._ws, self._ws_key = self._ws, None, None
        if ws is not None:
            try:
                asyncio.ensure_future(ws.close())
            except Exception:
                pass

    async def get_ws(self, language: str, pace: float):
        key = (language, round(float(pace), 2), self._speaker, self._voice_id)
        ws = self._ws
        if ws is not None and self._ws_key == key and getattr(ws, "close_code", None) is None:
            return ws
        self.drop_ws()
        _t_conn = time.monotonic()
        if self._voice_id:
            # ---- cloned voice: /voices/clone/ws ----
            ws = await websockets.connect(
                SARVAM_CLONE_WS_URL,
                additional_headers={"api-subscription-key": self._api_key},
                subprotocols=[f"api-subscription-key.{self._api_key}"],
                max_size=None, open_timeout=6, ping_interval=20, ping_timeout=20,
            )
            cfg = {
                "target_language_code": language,
                "voice_id": self._voice_id,
                "output_audio_codec": CLONE_TTS_CODEC,
            }
            if CLONE_SEND_SAMPLE_RATE:
                cfg["speech_sample_rate"] = str(self._sr)
            cfg.update(CLONE_EXTRA_CONFIG)          # from settings.json -> clone.extra_config
            await ws.send(json.dumps({"type": "config", "data": cfg}))
        else:
            # ---- default Sarvam voice (unchanged) ----
            ws = await websockets.connect(
                SARVAM_TTS_WS_URL.format(model=self._model),
                additional_headers={"Api-Subscription-Key": self._api_key},
                max_size=None, open_timeout=6, ping_interval=20, ping_timeout=20,
            )
            await ws.send(json.dumps({"type": "config", "data": {
                "model": self._model,
                "target_language_code": language,
                "speaker": self._speaker,
                "pace": pace,
                "temperature": self._temperature,
                "speech_sample_rate": str(self._sr),
                "output_audio_codec": "linear16",       # raw PCM, no mp3 decoding needed
                "enable_preprocessing": True,
                # if Sarvam ever 400s on a config key, delete that key here
                **SARVAM_TTS_EXTRA_CONFIG,
            }}))
        self._ws, self._ws_key = ws, key
        self._language_code, self._pace = language, pace     # keeper reconnects with last-used settings
        if TTS_TIMING_LOG:
            logger.info("TTS-TIMING socket connect+config %.0f ms (lang=%s)",
                        (time.monotonic() - _t_conn) * 1000, language)
        return ws

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
                   language_code: str = None, pace: float = None) -> "_SarvamChunkedStream":
        return _SarvamChunkedStream(
            tts_obj=self, input_text=text, conn_options=conn_options,
            language_code=language_code or self._language_code, pace=pace or self._pace,
        )

    def open_reply(self, out_q: "asyncio.Queue") -> "_SarvamReply":
        """One whole agent reply = one _SarvamReply (see below)."""
        return _SarvamReply(self, out_q)

    async def aclose(self) -> None:
        kt = getattr(self, "_keeper_task", None)
        if kt is not None:
            kt.cancel()
        self.drop_ws()
        await super().aclose()


class _SarvamReply:
    """ONE whole agent reply on the persistent Sarvam TTS socket.

    Old flow: send sentence 1, wait for ALL its audio, only then send sentence 2, wait for its first
    byte -> an audible gap at every sentence boundary.
    New flow: every sentence is sent the moment the LLM finishes it, and every audio chunk is pushed
    straight into out_q as 40 ms frames, so the reply plays as one continuous stream."""

    def __init__(self, tts_obj, out_q):
        self._t = tts_obj
        self._q = out_q
        self._ws = None
        self._lock_obj = None
        self._have_lock = False
        self._reader = None
        self._language = "hi-IN"
        self._pace = 1.0
        self._sent = 0          # sentences sent (each followed by a flush)
        self._finals = 0        # "final" events received
        self._input_done = False
        self._aborted = False
        self._failed = False
        self._ended = False
        self.started = False
        self.got_audio = False
        self._chars = 0          # total text chars sent on this reply
        self._audio_secs = 0.0   # total audio seconds received for this reply
        self._t_start = time.monotonic()
        self._t_send = None

    # ---------- lock / lifecycle ----------
    def _release(self):
        if self._have_lock:
            self._have_lock = False
            try:
                self._lock_obj.release()
            except Exception:
                pass

    def _finish(self):
        if self._ended:
            return
        self._ended = True
        # clone socket: if not every flush got its "final", late audio/events could leak into the next reply -> reconnect
        dirty = self._aborted or self._failed or (self._t.is_clone and self._finals < self._sent)
        if dirty:
            self._t.drop_ws()          # unfinished audio left on the socket must not leak into the next reply
        self._release()
        self._q.put_nowait(None)       # tells tts_node the stream is over
        if dirty:
            try:
                self._t.prewarm()      # reconnect in the background so the next reply skips the handshake
            except Exception:
                pass

    async def start(self, language: str, pace: float):
        t = self._t
        self._language, self._pace = language, pace
        try:
            self._lock_obj = t.lock
            try:
                await asyncio.wait_for(self._lock_obj.acquire(), timeout=10)
            except asyncio.TimeoutError:
                logger.warning("Sarvam TTS lock stuck for 10s - resetting it")
                t.drop_ws()
                t.lock = asyncio.Lock()
                self._lock_obj = t.lock
                await self._lock_obj.acquire()
            self._have_lock = True
            try:
                self._ws = await t.get_ws(language, pace)
            except Exception as e:
                logger.warning("Sarvam TTS connect failed (%r) - retrying once", e)
                t.drop_ws()
                self._ws = await t.get_ws(language, pace)
        except BaseException:
            self._release()
            raise
        self.started = True
        if TTS_TIMING_LOG:
            logger.info("TTS-TIMING lock+socket ready %.0f ms after reply start",
                        (time.monotonic() - self._t_start) * 1000)
        self._reader = asyncio.create_task(self._read())

    async def send(self, text: str):
        if self._t_send is None:
            self._t_send = time.monotonic()
        await self._ws.send(json.dumps({"type": "text", "data": {"text": text}}))
        await self._ws.send(json.dumps({"type": "flush"}))
        self._sent += 1
        self._chars += len(text)

    def end_input(self):
        """No more sentences are coming."""
        self._input_done = True
        if self._reader is None:
            self._finish()
        elif self._finals >= self._sent and not self._reader.done():
            self._reader.cancel()      # everything already finished, nothing left to wait for

    def abort(self):
        """Caller barged in / reply cancelled."""
        if self._ended:
            return
        self._aborted = True
        if self._reader is not None and not self._reader.done():
            self._reader.cancel()
        else:
            self._finish()

    # ---------- audio in ----------
    async def _read(self):
        t = self._t
        carry = b""
        rs_state = None
        src_rate = (CLONE_OUTPUT_RATE or t.sample_rate) if t.is_clone else t.sample_rate
        other_logged = 0
        frame_bytes = max(2, int(t.sample_rate * 0.04) * 2)       # 40 ms frames
        try:
            while True:
                if not self._input_done and (self._finals >= self._sent or (t.is_clone and self.got_audio)):
                    timeout = 120          # idle: LLM is still thinking, nothing pending on the socket
                elif self._input_done and self.got_audio:
                    if t.is_clone:
                        # clone socket does not reliably send 'final'. If we already got (roughly) as much
                        # audio as the text should produce, end quickly. If audio is clearly still missing
                        # (slow clone chunk), be patient instead of cutting the sentence in the middle.
                        expected = self._chars / max(CLONE_CHARS_PER_SEC, 1.0)
                        timeout = (CLONE_TAIL_TIMEOUT_SECS if self._audio_secs >= expected * 0.6
                                   else CLONE_GAP_TIMEOUT_SECS)
                    else:
                        timeout = 4
                else:
                    timeout = 15
                try:
                    raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    if self._input_done and self.got_audio:
                        if t.is_clone:
                            logger.info("Sarvam clone TTS: no 'final', finishing reply after %.1fs of silence "
                                        "(got %.1fs audio for %d chars)", timeout, self._audio_secs, self._chars)
                        else:
                            logger.warning("Sarvam TTS: no 'final' event after last audio - closing reply")
                            self._failed = True
                        break
                    raise
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype == "audio":
                    b = base64.b64decode(msg["data"]["audio"])
                    if b[:4] == b"RIFF":
                        if len(b) >= 28:
                            hdr_rate = int.from_bytes(b[24:28], "little")
                            if hdr_rate and hdr_rate != src_rate:
                                logger.warning("TTS WAV header says %s Hz, expected %s Hz -> resampling from %s Hz",
                                               hdr_rate, src_rate, hdr_rate)
                                src_rate, rs_state = hdr_rate, None
                        b = b[44:]                                 # WAV header on any chunk
                    b = carry + b
                    if len(b) % 2:
                        carry, b = b[-1:], b[:-1]
                    else:
                        carry = b""
                    if b and src_rate != t.sample_rate:
                        b, rs_state = audioop.ratecv(b, 2, 1, src_rate, t.sample_rate, rs_state)
                    self._audio_secs += len(b) / 2.0 / t.sample_rate
                    for i in range(0, len(b), frame_bytes):
                        piece = b[i:i + frame_bytes]
                        self._q.put_nowait(rtc.AudioFrame(
                            data=piece, sample_rate=t.sample_rate, num_channels=1,
                            samples_per_channel=len(piece) // 2,
                        ))
                    if b:
                        if not self.got_audio and TTS_TIMING_LOG and self._t_send:
                            logger.info("TTS-TIMING Sarvam FIRST AUDIO %.0f ms after text was sent (%d chars)",
                                        (time.monotonic() - self._t_send) * 1000, self._chars)
                        self.got_audio = True
                elif mtype == "event":
                    et = (msg.get("data") or {}).get("event_type")
                    if et in ("final", "completed", "complete", "done"):
                        self._finals += 1
                        if self._input_done and self._finals >= self._sent:
                            break
                    elif other_logged < 3:
                        other_logged += 1
                        logger.info("Sarvam TTS event: %s", str(msg)[:200])
                elif mtype == "error":
                    raise RuntimeError((msg.get("data") or {}).get("message", "Sarvam TTS error"))
                elif other_logged < 3:
                    other_logged += 1
                    logger.info("Sarvam TTS unknown message: %s", str(msg)[:200])
        except asyncio.CancelledError:
            pass                       # end_input() / abort() cancelled us on purpose
        except Exception as e:
            self._failed = True
            logger.warning("Sarvam TTS reply stream broke: %r", e)
        finally:
            self._finish()


class _SarvamChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts_obj: SarvamWSTTS, input_text: str, conn_options: APIConnectOptions,
                 language_code: str, pace: float):
        super().__init__(tts=tts_obj, input_text=input_text, conn_options=conn_options)
        self._language_code = language_code
        self._pace = pace

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        t = self._tts
        text = (self._input_text or "").strip()
        output_emitter.initialize(request_id=utils.shortuuid(), sample_rate=t.sample_rate,
                                  num_channels=1, mime_type="audio/pcm")
        if not text:
            return
        got_audio = False
        async with t.lock:
            for attempt in (1, 2):
                carry, first = b"", True
                try:
                    ws = await t.get_ws(self._language_code, self._pace)
                    await ws.send(json.dumps({"type": "text", "data": {"text": text}}))
                    await ws.send(json.dumps({"type": "flush"}))
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=15)
                        msg = json.loads(raw)
                        mtype = msg.get("type")
                        if mtype == "audio":
                            b = base64.b64decode(msg["data"]["audio"])
                            if first:
                                first = False
                                if b[:4] == b"RIFF":
                                    b = b[44:]               # strip WAV header if Sarvam sends one
                            b = carry + b
                            if len(b) % 2:
                                carry, b = b[-1:], b[:-1]
                            else:
                                carry = b""
                            if b:
                                output_emitter.push(b)
                                got_audio = True
                        elif mtype == "event":
                            if (msg.get("data") or {}).get("event_type") == "final":
                                break
                        elif mtype == "error":
                            raise RuntimeError((msg.get("data") or {}).get("message", "Sarvam TTS error"))
                    output_emitter.flush()
                    return
                except asyncio.CancelledError:
                    t.drop_ws()          # barge-in: leftover audio on the socket would leak into the next sentence
                    raise
                except Exception as e:
                    t.drop_ws()
                    if got_audio:
                        logger.warning("Sarvam TTS broke mid-sentence: %r", e)
                        output_emitter.flush()
                        return
                    if attempt == 2:
                        raise APIConnectionError(f"Sarvam TTS failed: {e!r}") from e
                    logger.warning("Sarvam TTS ws error (%r) - reconnecting once", e)


def _build_stt(language):
    """Sarvam saaras:v3-realtime over websocket. en / hi / auto agents are LOCKED to hi-IN (handles
    Hindi + English code-mix) because 'unknown' auto-detect was what produced the Telugu/Bengali/
    Gujarati hallucinations on background noise. Regional agents use their own language code."""
    lang = language if language in SUPPORTED_LANGUAGES else None
    regional = lang is not None and lang not in ("en", "hi")
    if SARVAM_STT_LANGUAGE_OVERRIDE:
        code = SARVAM_STT_LANGUAGE_OVERRIDE
    elif regional:
        code = LANGUAGES[lang]["tts"]
    else:
        code = "hi-IN"
    logger.info("STT: Sarvam realtime %s (language=%s, mode=%s, stream_type=%s)",
                SARVAM_STT_RT_MODEL, code, SARVAM_STT_MODE, SARVAM_STT_STREAM_TYPE)
    return SarvamRealtimeSTT(language_code=code, agent_language=lang)

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
    if lang == "en" and EN_MIX_AS_HINDI:
        return LANGUAGES["hi"]["tts"]          # Hinglish-friendly voice for English agents
    return (LANGUAGES.get(lang) or LANGUAGES["en"])["tts"]


_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]")
_MD_RE = re.compile(r"[*_`#~|]+|^\s*[-•]\s+", re.M)


def _tts_sanitize(text: str) -> str:
    """Last line of defence before TTS: strips markdown, bullets and emojis the LLM may still emit,
    since the voice would either read them out or stumble on them."""
    text = _EMOJI_RE.sub("", text or "")
    text = _MD_RE.sub("", text)
    text = text.replace("&", " and ")
    return re.sub(r"\s{2,}", " ", text).strip()


def _split_sentences(text: str) -> list:
    """Same boundary rule as the streaming splitter, but returns the sentences as a list."""
    parts = SENTENCE_END_RE.split(text)
    out, i = [], 0
    while i + 1 < len(parts):
        piece = (parts[i] + parts[i + 1]).strip()
        if piece:
            out.append(piece)
        i += 2
    if i < len(parts) and parts[i].strip():
        out.append(parts[i].strip())
    return out or [text.strip()]


def _split_first_clause(piece: str):
    """Long first sentence -> (first clause, rest) so the first TTS request is small. Else (piece, '')."""
    if len(piece) <= FIRST_CHUNK_SPLIT_MIN_CHARS:
        return piece, ""
    for m in FIRST_CLAUSE_RE.finditer(piece):
        if m.end() >= MIN_FIRST_CHUNK_CHARS and len(piece) - m.end() >= 12:
            return piece[:m.end()].strip(), piece[m.end():].strip()
    return piece, ""


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
        self._raw_instructions = instructions
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
                _raw = self._raw_instructions or ""
                _m = re.search(r"CALL SCENARIO:\s*(first|followup|incoming|callback)", _raw)
                _scn = _m.group(1) if _m else ""
                _has_ctx = "PAST CONTEXT ABOUT THIS PERSON" in _raw
                if _scn == "callback":
                    _greet = (
                        "You are calling this person back because they asked you to call at this time. In ONE short "
                        "sentence greet them by name, say your name and that you are calling back as they requested, "
                        "and ask if it is a good time to talk. "
                    )
                elif _scn == "followup" and _has_ctx:
                    _greet = (
                        "You are calling someone you have already spoken with. In ONE short sentence greet them by "
                        "name, say your name, and mention in a few words what you discussed last time (use the PAST "
                        "CONTEXT), then ask if it is a good time to continue. Do not summarise everything. "
                    )
                elif _scn == "incoming" and _has_ctx:
                    _greet = (
                        "The caller phoned in and you already know them. In ONE short sentence welcome them back by "
                        "name and ask how you can help; you may refer to their earlier requirement in a few words. "
                    )
                else:
                    _greet = "Greet the caller warmly in ONE short sentence and ask how you can help today. "
                await self.session.generate_reply(instructions=_greet + name_hint)
        except Exception:
            logger.exception("on_enter failed for call_id=%r - greeting was not spoken", self._call_id)

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        # One continuous stream per reply: each sentence goes to Sarvam the moment it is complete
        # (no waiting for the previous sentence's audio) and audio frames flow straight back out.
        out_q: asyncio.Queue = asyncio.Queue()
        reply = self._global_tts.open_reply(out_q)
        state = {"finished": False}
        _clone = self._global_tts.is_clone
        if _clone and CLONE_SINGLE_SHOT:
            # never flush early: everything is collected and sent once, by _flush() at the end of the reply
            _min_first = _batch_min = 10 ** 9
            _early_comma = False
        else:
            _min_first = CLONE_MIN_FIRST_CHUNK_CHARS if _clone else MIN_FIRST_CHUNK_CHARS
            _batch_min = CLONE_BATCH_MIN_CHARS if _clone else TTS_BATCH_MIN_CHARS
            _early_comma = CLONE_EARLY_COMMA if _clone else True

        async def _clean(sentence: str) -> str:
            sentence = await self._handle_booking_tag(sentence)
            sentence = self._strip_end_call(sentence)
            sentence = IGNORE_RE.sub("", sentence).strip()      # [IGNORE] is never spoken
            return _tts_sanitize(sentence)                      # no markdown/emoji reaches the voice

        batch = {"text": ""}

        async def _send_batch():
            t = batch["text"].strip()
            batch["text"] = ""
            if not t:
                return
            if not reply.started:
                lang = _pick_tts_lang(t, self._forced_lang)
                logger.info("speaking [%s]: %s", LANG_NAMES.get(lang, lang), t[:80])
                await reply.start(_sarvam_lang_code(lang), TTS_SPEED)
            else:
                logger.info("speaking (cont.): %s", t[:80])
            await reply.send(t)

        async def _say(sentence: str):
            batch["text"] = (batch["text"] + " " + sentence).strip()
            # First chunk of a reply goes out IMMEDIATELY so audio starts as soon as possible.
            # Everything after it is batched for smooth intonation and no gaps between sentences.
            if (not reply.started and len(batch["text"]) >= _min_first) \
                    or len(batch["text"]) >= _batch_min:
                await _send_batch()

        async def _flush():
            await _send_batch()

        async def _produce():
            buffer = ""
            spoken_chars = 0
            capped = False
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

                    # Early flush: first chunk of a reply is cut at the first comma so audio starts sooner
                    if not sentence and spoken_chars == 0 and _early_comma:
                        for m in FIRST_CLAUSE_RE.finditer(buffer):
                            if m.end() >= _min_first:
                                sentence = buffer[:m.end()].strip()
                                buffer = buffer[m.end():]
                                break

                    if not sentence:
                        continue
                    # a fast LLM sends the whole reply in ONE chunk -> send sentence 1 alone, the rest right after
                    for piece in _split_sentences(sentence):
                        has_tag = bool(BOOK_MEETING_RE.search(piece))
                        piece = await _clean(piece)
                        if not piece or not SPEAKABLE_RE.search(piece) or (capped and not has_tag):
                            continue
                        if not reply.started and _early_comma and not batch["text"]:
                            head, rest = _split_first_clause(piece)
                            if rest:
                                await _say(head)            # >= _min_first chars -> goes out immediately
                                spoken_chars += len(head)
                                piece = rest
                        await _say(piece)
                        spoken_chars += len(piece)
                        if spoken_chars >= MAX_SPOKEN_CHARS:
                            capped = True

                tail = buffer.strip()
                if tail:
                    has_tag = bool(BOOK_MEETING_RE.search(tail))
                    tail = await _clean(tail)
                    if tail and SPEAKABLE_RE.search(tail) and (not capped or has_tag):
                        await _say(tail)
                await _flush()      # send whatever is still batched
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("tts_node producer failed")
            finally:
                reply.end_input()

        producer = asyncio.create_task(_produce())
        # Clone voice: LiveKit plays frames in real time, so if the clone synthesizes slower than that,
        # playback runs dry mid-sentence. Hold back a short pre-roll so synthesis stays ahead of playback.
        held, held_secs, t_first = [], 0.0, None
        released = not (_clone and CLONE_PREROLL_SECS > 0)
        try:
            while True:
                if released:
                    frame = await out_q.get()
                else:
                    wait = None
                    if t_first is not None:
                        wait = max(0.01, CLONE_PREROLL_MAX_WAIT_SECS - (time.monotonic() - t_first))
                    try:
                        frame = await asyncio.wait_for(out_q.get(), timeout=wait)
                    except asyncio.TimeoutError:
                        released = True                      # waited long enough, start playing what we have
                        for f in held:
                            yield f
                        held = []
                        continue
                if frame is None:
                    for f in held:
                        yield f
                    held = []
                    break
                if released:
                    yield frame
                    continue
                if t_first is None:
                    t_first = time.monotonic()
                held.append(frame)
                held_secs += frame.samples_per_channel / float(frame.sample_rate)
                if held_secs >= CLONE_PREROLL_SECS:
                    released = True
                    for f in held:
                        yield f
                    held = []
            state["finished"] = True
        finally:
            producer.cancel()
            reply.abort()                      # no-op if the reply already finished normally
            if not state["finished"]:
                # reply was interrupted by the caller -> don't hang up on them
                self._end_call_requested = False

    async def llm_node(self, chat_ctx, tools, model_settings):
        # Sarvam sometimes returns the END_CALL marker as a *tool call* even though we declare no tools.
        # 1) never send tool messages back to the LLM  2) turn any stray tool call back into spoken text.
        # 3) NEW: if the model returns no spoken text at all (seen in your log: 200 tokens, ttft=-1),
        #    retry once, then fall back to a short line - the caller must never get dead air.
        try:
            chat_ctx = chat_ctx.copy(exclude_function_call=True)
        except TypeError:
            pass
        try:
            self._global_tts.ensure_warm()      # socket handshake overlaps the LLM call
        except Exception:
            pass

        state = {"spoken": False}

        async def _run_once():
            async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
                delta = getattr(chunk, "delta", None)
                calls = getattr(delta, "tool_calls", None) if delta is not None else None
                if not calls:
                    txt = chunk if isinstance(chunk, str) else (
                        getattr(delta, "content", None) if delta is not None else None)
                    if txt and str(txt).strip():
                        state["spoken"] = True
                    yield chunk
                    continue
                if getattr(delta, "content", None):
                    state["spoken"] = True
                    yield delta.content
                for c in calls:
                    txt = (getattr(c, "name", "") or "").strip()
                    logger.warning("LLM returned a tool call although no tools exist - speaking it as text: %r", txt[:80])
                    if txt:
                        state["spoken"] = True
                        yield txt + "\n"

        async for out in _run_once():
            yield out
        if not state["spoken"] and LLM_EMPTY_RETRY:
            logger.warning("LLM produced NO spoken text (empty / reasoning-only) -> retrying once")
            async for out in _run_once():
                yield out
        if not state["spoken"]:
            logger.warning("LLM empty twice -> speaking fallback line")
            yield ("जी, एक बार फिर बोलिए?" if self._forced_lang in (None, "hi")
                   else "Ji, ek baar phir boliye?")

    async def _handle_booking_tag(self, sentence: str) -> str:
        """Strips a BOOK_MEETING: tag out of the spoken text and fires the
        webhook, same convention app.py's LLM loop used."""
        bm = BOOK_MEETING_RE.search(sentence)
        if not bm:
            return sentence
        requested_iso = f"{bm.group(1)}T{bm.group(2)}:00"
        confirmed, message = await asyncio.to_thread(
            book_meeting_via_pravaah, self._meeting, self._lead, self._call_id, requested_iso
        )
        logger.info(f"[{self._call_id or 'test'}] booking requested={requested_iso} confirmed={confirmed}")
        remainder = BOOK_MEETING_RE.sub("", sentence).strip()
        # Speak the booking result instead of the raw tag.
        return (remainder + " " + message).strip() if remainder else message

    async def _speak(self, sentence: str):
        lang = _pick_tts_lang(sentence, self._forced_lang)
        logger.info("speaking [%s]: %s", LANG_NAMES.get(lang, lang), sentence[:60])
        async for audio in self._global_tts.synthesize(
            sentence, language_code=_sarvam_lang_code(lang), pace=TTS_SPEED
        ):
            yield audio.frame



def _build_instructions(agent_cfg: dict, lead: dict, meeting: dict, inbound: bool = False, clone: bool = False, greeting_played: bool = True):
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
            "\nLANGUAGE RULE (always follow): reply in the language the caller is speaking. "
            "If they speak Hindi or a Hindi-English mix, reply in natural Hindi with everyday English words mixed "
            "in: every Hindi word in Devanagari, every English word in Latin letters, in the same sentence (see "
            "HINDI SCRIPT RULE at the end). If they speak only English, reply in simple Indian English in Latin "
            "letters. Match their ratio of Hindi to English. "
            "If the caller clearly speaks another Indian language, or asks you to speak in one "
            "(Gujarati, Punjabi, Marathi, Bengali, Tamil, Telugu, Kannada, Malayalam, Odia, Haryanvi...), "
            "switch to that language and write in its own native script (never Roman letters) until they ask to change."
        )
    # Same gender-agreement rule as app.py's EvaSession - keeps karta/karti
    # consistent with agent_cfg.gender (the same field that picks the voice
    # a few lines up in entrypoint()). Default matches the voice-selection
    # default (female) when gender isn't set.
    gender_forms = (
        "masculine (e.g. मैं करता हूँ, मैं बोल रहा हूँ, मैं था)"
        if agent_cfg.get("gender") == "male" else
        "feminine (e.g. मैं करती हूँ, मैं बोल रही हूँ, मैं थी)"
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
        "phone call. Each reply is one or two SHORT sentences. Target 90-180 "
        "characters, never above 220. Never read the example lines in the script word "
        "for word - compress them to the shortest natural version. Never say 'Sir/Ma'am' "
        "out loud as a pair: use the caller's name followed by जी (e.g. 'Bhuvi जी'), or just जी. "
        "Keep any closing summary to one sentence."
    )
    base += (
        "\n\nNATURAL ACKNOWLEDGEMENT (always follow): start every reply with a 2-4 word "
        "acknowledgement that picks up the SPECIFIC detail the caller just gave, then ask exactly "
        "ONE question. Example: caller says 'मैं South Delhi में देख रहा हूँ' -> 'South Delhi, अच्छा जी। "
        "आपका budget कितना सोच रखा है?'. Vary the openers (अच्छा, ठीक है, जी, बिल्कुल, Got it) "
        "and never use the same one twice in a row. Never repeat the caller's sentence back."
        "\nONE QUESTION RULE: a reply has at most ONE question mark. Never join two questions with "
        "'और' or 'and' (for example never 'आप कौन सी class में हैं और target exam कौन सा है')."
        "\nMEMORY RULE: before asking anything, check the whole conversation. If the caller already "
        "gave their class, exam, subject, location or online/offline choice, never ask it again. If "
        "the caller says they already told you something, say 'Sorry जी, note कर लिया' and move on."
    )
    base += (
        "\n\nBACKGROUND VOICES: the line may pick up other people talking nearby. If the caller's "
        "latest message is UNMISTAKABLY a conversation between other people about something unrelated "
        "to this call (cars, family chatter, directions) and contains no answer or question for you, "
        "output exactly [IGNORE] and nothing else. If a message is only half-heard or unclear, do not "
        "guess - say a very short 'जी, दोबारा बोलिए?' in the call's language and script. If the message is "
        "even partly an answer to you, answer it normally. Never mention noise or background voices."
    )
    base += (
        "\n\nVOICE STYLE (always follow): your words are converted straight to speech, so write exactly how "
        "they should be SPOKEN. Sound like a warm, relaxed Indian professional on a real phone call: "
        "contractions, everyday words, and light fillers used sparingly (अच्छा, हाँ जी, देखिए, तो, बिल्कुल, "
        "ठीक है). Use short sentences with commas and full stops where a person would naturally pause. "
        "Never use markdown, asterisks, bullets, numbered lists, emojis, brackets or symbols. Write amounts, "
        "dates and times the way they would be said aloud (e.g. '1.5 करोड़', 'कल शाम 5 बजे'). "
        "Never sound scripted, stiff or like a customer-care recording, and never use the same opener or "
        "filler twice in a row."
    )
    if inbound and greeting_played:
        base += (
            "\n\nINBOUND CALL: the person phoned the business - you did NOT call them. Never say you are "
            "calling them and never ask 'do you have a minute'. The greeting was already said at the start "
            "of the call, so do NOT greet again or re-introduce yourself; just respond to what they say and help them."
        )
    elif inbound:
        base += (
            "\n\nINBOUND CALL: the person phoned the business - you did NOT call them. Never say you are "
            "calling them and never ask 'do you have a minute'. You open the call yourself with one short, warm "
            "greeting, then respond to what they say and help them."
        )
    base += (
        "\n\nENDING THE CALL: when the caller clearly wants to finish the conversation (for example 'ok bye', "
        "'bye', 'thank you bye', 'that's all', 'theek hai bye', 'rakhta hoon', 'phone rakhti hoon', 'dhanyavaad bye'), "
        "reply with ONE short polite goodbye sentence and then write [END_CALL] right after it, on the same line, as the very last thing in that reply (never at the start, never on its own line, never twice). "
        "Never say END_CALL out loud or explain it. A plain 'okay', 'hmm', 'haan' or 'theek hai' on its own is NOT a "
        "goodbye - only use END_CALL when the caller is really ending the call."
    )
    if clone:
        base += (
            "\n\nVOICE SCRIPT (overrides any earlier instruction about Roman letters): your voice is a cloned "
            "human voice that reads Devanagari best. Whenever you speak Hindi, write the Hindi words in Devanagari "
            "script and keep English words (property, budget, plot, flat, site visit, Nisha Homes) in Latin letters, "
            "for example: 'जी, आपका budget कितना रहेगा?'. Keep sentences short with natural punctuation."
            "\nCLONE LENGTH RULE (overrides the speaking length rule above): every reply is ONE short sentence of "
            "40-80 characters, never above 90, with at most one question. Compress any script line that is longer, "
            "including the closing lines, for example 'ठीक है जी। आप किस area में property देख रहे हैं?'. "
            "Never use two sentences when one is enough."
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
    # Last on purpose: the most recent instruction wins over Roman-script lines in the owner's custom prompt.
    if forced_lang in (None, "hi"):
        base += "\n\n" + HINDI_SCRIPT_RULE
    return base, forced_lang, persona_name

def prewarm(proc: JobProcess) -> None:
    # Higher activation threshold + longer min speech = faint background voices and mic bumps
    # no longer count as "the caller is speaking".
    _vad_kwargs = dict(
        min_silence_duration=0.35,
        min_speech_duration=VAD_MIN_SPEECH_SECS,
        activation_threshold=VAD_ACTIVATION_THRESHOLD,
    )
    _vad_valid = set(inspect.signature(silero.VAD.load).parameters)
    proc.userdata["vad"] = silero.VAD.load(**{k: v for k, v in _vad_kwargs.items() if k in _vad_valid})


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
    _lang0 = agent_cfg.get("language") if agent_cfg.get("language") in SUPPORTED_LANGUAGES else "hi"
    _clone_voice_id = (agent_cfg.get("cloned_voice_id") or "").strip()
    if _clone_voice_id:
        logger.info("call_id=%r using CLONED voice %s", call_id, _clone_voice_id)
    global_tts = SarvamWSTTS(
        api_key=SARVAM_API_KEY,
        voice_id=_clone_voice_id,
        model=SARVAM_TTS_MODEL,
        speaker=voice,
        language_code=_sarvam_lang_code(_lang0),   # same language the opening line will use -> no reconnect
        pace=TTS_SPEED,
        temperature=SARVAM_TTS_TEMPERATURE,
        sample_rate=SARVAM_TTS_SAMPLE_RATE,
    )
    global_tts.prewarm()   # opens the Sarvam TTS websocket now, not on the first sentence

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
            extra_body={"max_tokens": SARVAM_LLM_MAX_TOKENS, "reasoning_effort": SARVAM_LLM_REASONING_EFFORT},
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
        turn_detection=("vad" if TURN_DETECTION_MODE == "vad" else
                        "stt" if TURN_DETECTION_MODE == "stt" else MultilingualModel()),
        allow_interruptions=True,
        min_interruption_duration=MIN_INTERRUPTION_SECS,
        min_interruption_words=MIN_INTERRUPTION_WORDS,   # barge-in needs real words, not a noise burst
        min_endpointing_delay=MIN_ENDPOINTING_DELAY,
        max_endpointing_delay=MAX_ENDPOINTING_DELAY,
        preemptive_generation=True,       # start generating before the user's turn is fully finalized
        resume_false_interruption=True,   # resume speaking if a "barge-in" turns out to be noise
        false_interruption_timeout=1.0,
    )
    try:
        llm.prewarm()      # opens the HTTPS connection to Sarvam LLM now (saves the TLS handshake on turn 1)
    except Exception:
        pass
    _valid_params = set(inspect.signature(AgentSession.__init__).parameters)
    _session_kwargs = {k: v for k, v in _session_kwargs.items() if k in _valid_params}
    session = AgentSession(**_session_kwargs)

    session.on("error", lambda ev: logger.error("SESSION ERROR: %r", getattr(ev, "error", ev)))
    session.on("agent_state_changed",
               lambda ev: logger.info("agent state: %s -> %s", ev.old_state, ev.new_state))
    _pending_interim = {"text": "", "t": 0.0}

    def _on_user_transcribed(ev):
        if ev.is_final:
            logger.info("USER SAID: %s", ev.transcript)
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
        text = IGNORE_RE.sub("", text).strip()
        if role == "agent" and not text:
            return
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

    instructions, forced_lang, persona_name = _build_instructions(
        agent_cfg, lead, meeting, inbound=inbound,
        clone=bool(_clone_voice_id) and CLONE_DEVANAGARI_HINDI,
        greeting_played=(greeting_played
                         or bool((agent_cfg.get("opening_line") or "").strip())
                         or bool((agent_cfg.get("opening_audio_url") or "").strip())),
    )
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
                await session.say(
                    "Thank you जी, आपसे बात करके अच्छा लगा।" if tts_lang == "hi"
                    else "Thank you ji, aapse baat karke achha laga.",
                    allow_interruptions=False,
                )
            except Exception:
                pass
            ctx.shutdown(reason="max_duration_reached")

        _watchdog_task = asyncio.create_task(_max_duration_watchdog())


if __name__ == "__main__":
    cli.run_app(server)
import os
import re
import sys
import json
import time
import math
import array
import base64
import queue
import random
import threading
import uuid
from datetime import datetime

import httpx
import requests
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_sock import Sock
from dotenv import load_dotenv

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import VoiceResponse, Connect

import websocket as vanisetu_ws_lib   # pip install websocket-client

import collections
import gevent.monkey as _gevent_monkey
# Genuine OS thread class (pre-monkeypatch) - same trick as livekit_bridge.py
_RealThread = _gevent_monkey.get_original("threading", "Thread")


# --- NEW ---
import math
import array
import audioop          # stdlib on <3.13, audioop-lts backport on 3.13+
import asyncio
import base64
import queue
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
from livekit.agents import inference   # replaces sarvamai
from livekit.agents.utils import http_context   # replaces sarvamai
from twilio.rest import Client as TwilioClient

import livekit_bridge


# Load .env before reading any LiveKit / Eva configuration.
load_dotenv()

LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")

livekit_bridge.init(
    LIVEKIT_URL,
    LIVEKIT_API_KEY,
    LIVEKIT_API_SECRET,
    os.environ.get("AGENT_NAME", "eva-agent"),
)

# ---------------- Config ----------------
PORT = int(os.environ.get("PORT", 8420))

MIC_RATE = 16000               # PCM16 the browser sends to us
TTS_SAMPLE_RATE = 22050        # PCM16 we send back to the browser

SENTENCE_END_RE = re.compile(r"([.!?।\n])")
# Matches Latin, Devanagari, Bengali, Tamil, Telugu, Kannada, or Malayalam
# characters — used to decide if a chunk of text has anything speakable.
# --- NEW ---
SPEAKABLE_RE = re.compile(r"[A-Za-z0-9\u0900-\u0D7F]")   # Latin + all major Indic scripts
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
BOOK_MEETING_RE = re.compile(r"BOOK_MEETING:\s*(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})")


# ---------------- Language support: English, Hindi + Indian languages/dialects ----------------
# tts = Sarvam language code (also used for Sarvam STT). script None = Roman/Hinglish.
LANGUAGES = {
    "en":  {"name": "English",    "tts": "en-IN", "script": None},
    "hi":  {"name": "Hindi",      "tts": "hi-IN", "script": None},
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
SUPPORTED_LANGUAGES = set(LANGUAGES)
LANG_NAMES = {k: v["name"] for k, v in LANGUAGES.items()}

# script -> language key (Devanagari maps to "hi")
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


def reply_language_rule(lang: str) -> str:
    """Prompt line telling the LLM which language + script to reply in."""
    cfg = LANGUAGES.get(lang)
    if not cfg or lang == "en":
        return "Reply in English only."
    name = cfg["name"]
    if not cfg["script"]:
        return f"Reply in casual {name}, written in Roman/English letters (Hinglish) - do NOT use Devanagari or any native script."
    rule = (f"Reply only in casual, natural spoken {name}, written in {cfg['script']} script "
            f"(never Roman letters), the way people actually talk on a phone call.")
    if cfg.get("dialect"):
        rule += f" Use real {name} words, grammar and tone - not standard textbook Hindi."
    return rule

# ---------------- LiveKit voice (TTS) ----------------
# No more Sarvam — TTS now goes through LiveKit's hosted Inference API,
# authenticated with LIVEKIT_URL/LIVEKIT_API_KEY/LIVEKIT_API_SECRET
# (same creds Eva V2 already uses for its web widget), not a per-vendor key.
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")
LIVEKIT_TTS_MODEL = os.environ.get("LIVEKIT_TTS_MODEL", "inworld/inworld-tts-2")
# One voice per gender, reused across the whole call regardless of which
# of the two supported languages is being spoken.
# Sarvam bulbul:v2 speaker names. Female: anushka/manisha/vidya/arya.
# Male: abhilash/karun/hitesh. Full/updated list: docs.sarvam.ai (TTS).
# bulbul:v3 speaker catalog (different from v2's). Male: shubh (default),
# aditya, rahul, rohan, amit, dev... Female: ritu, priya, neha, pooja...
VOICE_MALE = os.environ.get("EVA_VOICE_MALE", "shubh")
VOICE_FEMALE = os.environ.get("EVA_VOICE_FEMALE", "priya")
# Speaking-rate multiplier passed to the TTS engine (1.0 = normal). Bump
# slightly if replies feel slow/robotic; not every provider build accepts
# this kwarg, so it's applied with a fallback wherever it's used below.
TTS_SPEED = float(os.environ.get("EVA_TTS_SPEED", "1.0"))

MAX_HISTORY_MESSAGES = 16

# How long Eva waits, after the user goes quiet, before she actually replies.
# Mimics a natural human turn-taking gap instead of jumping in instantly.
# Lowered from 0.4 -> 0.22: the old value was adding noticeable dead-air
# on every single turn. Tune via env if it starts cutting people off.
RESPONSE_DELAY_SECS = float(os.environ.get("EVA_RESPONSE_PAUSE_SECS", 0.22))
# Small random jitter added on top of the base pause so Eva doesn't reply
# on the exact same beat every time - a perfectly fixed delay is what
# makes a voice bot feel mechanical.
RESPONSE_DELAY_JITTER_SECS = float(os.environ.get("EVA_RESPONSE_PAUSE_JITTER", 0.12))
# Short acknowledgements ("yes", "okay", "no thanks") get a shorter pause -
# humans reply to quick confirmations faster than to longer statements.
SHORT_UTTERANCE_MAX_WORDS = int(os.environ.get("EVA_SHORT_UTTERANCE_MAX_WORDS", 3))
SHORT_UTTERANCE_DELAY_SECS = float(os.environ.get("EVA_SHORT_UTTERANCE_PAUSE_SECS", 0.18))

BARGE_IN_GRACE_SECS = float(os.environ.get("EVA_BARGE_IN_GRACE_SECS", 1.0))
# VAD (SpeechStarted) fires on ANY audio energy spike - coughs, breathing,
# mic bumps - not just real speech. We no longer interrupt Eva on VAD
# alone; we wait to see if Deepgram actually transcribes real words within
# this window before treating it as a genuine barge-in.
BARGE_IN_CONFIRM_MIN_CHARS = int(os.environ.get("EVA_BARGE_IN_MIN_CHARS", 3))
BARGE_IN_CONFIRM_TIMEOUT_SECS = float(os.environ.get("EVA_BARGE_IN_CONFIRM_TIMEOUT", 0.8))
# Amplitude-based barge-in trigger, independent of (and faster than)
# Deepgram's VAD. We look at the raw volume of what's actually coming in on
# the mic/line while Eva is talking. This is the line between "the user is
# talking" and "there's noise in the background": ambient sound (traffic,
# a fan, other people across the room) is quieter than the user's own voice
# because it isn't right on the mic/handset, so it normally stays under
# these numbers. Crossing the threshold only ARMS a candidate barge-in,
# same as VAD - it still needs a real transcribed word from Deepgram to
# actually interrupt Eva (see _arm_barge_in_candidate), so one loud
# one-off noise (a horn, a door) that isn't speech won't cut her off alone.
# Linear16 samples (browser mic) range roughly -32768..32767.
BARGE_IN_MIN_VOLUME_LINEAR16 = int(os.environ.get("EVA_BARGE_IN_MIN_VOLUME_LINEAR16", 600))
# mu-law (phone calls) decodes to a smaller effective range (~-8031..8031),
# so this threshold is scaled down to match.
BARGE_IN_MIN_VOLUME_MULAW = int(os.environ.get("EVA_BARGE_IN_MIN_VOLUME_MULAW", 350))
# A-law (VoiceLink calls) has a similar effective dynamic range to mu-law,
# so the same default is a reasonable starting point — tune independently
# via EVA_BARGE_IN_MIN_VOLUME_ALAW if VoiceLink calls prove more/less sensitive.
BARGE_IN_MIN_VOLUME_ALAW = int(os.environ.get("EVA_BARGE_IN_MIN_VOLUME_ALAW", 350))
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
SARVAM_TTS_MODEL = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v3")
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_SUPPORTED_RATES = (8000, 16000, 22050, 24000)
# bulbul:v3 has no pitch/loudness controls (unlike v2) but adds
# "temperature" (0.01-1.0, default 0.6) - controls expressiveness/randomness.
SARVAM_TTS_TEMPERATURE = float(os.environ.get("SARVAM_TTS_TEMPERATURE", "0.6"))
# Internal "en"/"hi" -> Sarvam's BCP-47 target_language_code.
SARVAM_LANG_CODES = {k: v["tts"] for k, v in LANGUAGES.items()}

# ---------------- IVR (play recording -> wait -> STT -> hang up) ----------------
import io
import wave

IVR_WAIT_SECS = float(os.environ.get("EVA_IVR_WAIT_SECS", 5))
IVR_MAX_EXTEND_SECS = float(os.environ.get("EVA_IVR_MAX_EXTEND_SECS", 3))   # extra time if caller is still mid-sentence
IVR_STT_LANGUAGE = os.environ.get("EVA_IVR_STT_LANGUAGE", "multi")
IVR_MIN_SPEECH_RMS = int(os.environ.get("EVA_IVR_MIN_SPEECH_RMS", 250))     # 16-bit RMS below this = silence/noise
IVR_PLAY_LEAD_SECS = 0.3

IVR_NO_TOKENS = {
    "no", "nope", "nah", "nahi", "nahin", "nahii", "nai", "mat", "नहीं", "नही", "नहिं", "मत",
    # Marathi / Haryanvi / Rajasthani / Bhojpuri
    "नाही", "नको", "नका", "कोनी", "नाहीं", "नइखे", "नइखी",
    # Gujarati / Punjabi
    "ના", "નહીં", "નહિ", "નથી", "ਨਹੀਂ", "ਨਹੀ", "ਨਾ",
    # Bengali / Odia
    "না", "নয়", "ନାହିଁ", "ନା", "ନୁହେଁ",
    # Tamil / Telugu / Kannada / Malayalam
    "இல்லை", "வேண்டாம்", "வேணாம்", "లేదు", "వద్దు", "కాదు", "ಇಲ್ಲ", "ಬೇಡ", "ഇല്ല", "വേണ്ട", "അല്ല",
}
# "ना" is "no" in Haryanvi/Bhojpuri but also a filler in Hindi ("theek hai na") -> only counts in very short replies
IVR_NO_TOKENS_SHORT_ONLY = {"ना"}
IVR_NO_PHRASES = ("not interested", "don't", "dont", "do not", "no thanks", "no thank you")

_IVR_AUDIO_CACHE = {}
_ivr_cache_lock = threading.Lock()


def ivr_is_negative(text: str) -> bool:
    t = (text or "").lower()
    if any(p in t for p in IVR_NO_PHRASES):
        return True
    tokens = re.findall(r"[\w\u0900-\u0DFF]+", t)   # \u0900-\u0DFF keeps Indic vowel signs inside words
    if any(tok in IVR_NO_TOKENS for tok in tokens):
        return True
    return len(tokens) <= 3 and any(tok in IVR_NO_TOKENS_SHORT_ONLY for tok in tokens)

# ---------------- LLM provider switch ----------------
# Set LLM_PROVIDER=gemini in .env to swap Eva's brain from Groq to Gemini,
# or LLM_PROVIDER=groq to go back — no code changes needed either way.
# Both API keys can sit in .env at once; only the selected one is used.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "sarvam").strip().lower()  # "sarvam" | "cloudflare" | "groq" | "gemini"

# ---------------- Sarvam LLM (conversational, voice-agent tuned) ----------------
# Reuses SARVAM_API_KEY (same key as TTS). Reasoning is disabled in the
# request because reasoning tokens are billed and add latency on calls.
SARVAM_LLM_MODEL = os.environ.get("SARVAM_LLM_MODEL", "sarvam-105b-conversations")
SARVAM_LLM_URL = "https://api.sarvam.ai/v1/chat/completions"
SARVAM_LLM_TEMPERATURE = float(os.environ.get("SARVAM_LLM_TEMPERATURE", "0.4"))
# ~650 chars is about 1 minute of speech. Token cap is the hard cost ceiling;
# the char cap below ends the reply cleanly at a sentence boundary.
SARVAM_LLM_MAX_TOKENS = int(os.environ.get("SARVAM_LLM_MAX_TOKENS", "240"))
SARVAM_LLM_MAX_REPLY_CHARS = int(os.environ.get("SARVAM_LLM_MAX_REPLY_CHARS", "650"))

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

# ---------------- Cloudflare Workers AI (LLM) ----------------
# Uses Workers AI's OpenAI-compatible endpoint, so the SSE delta shape is
# identical to Groq's - _stream_chat_cloudflare() below mirrors
# _stream_chat_groq() almost line for line.
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_MODEL = os.environ.get("CLOUDFLARE_MODEL", "openai/gpt-6-sol")
# Kept small on purpose - Eva's replies are 1-3 spoken sentences anyway,
# so there's no reason to pay for (or wait on) a long completion.
CLOUDFLARE_MAX_TOKENS = int(os.environ.get("CLOUDFLARE_MAX_TOKENS", "120"))
CLOUDFLARE_TEMPERATURE = float(os.environ.get("CLOUDFLARE_TEMPERATURE", "0.4"))
CLOUDFLARE_CHAT_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1/chat/completions"
    if CLOUDFLARE_ACCOUNT_ID else ""
)


def check_missing_keys():
    checks = [
 #       ("DEEPGRAM_API_KEY", DEEPGRAM_API_KEY),
        ("LIVEKIT_URL", LIVEKIT_URL),
        ("LIVEKIT_API_KEY", LIVEKIT_API_KEY),
        ("LIVEKIT_API_SECRET", LIVEKIT_API_SECRET),
    ]
    if LLM_PROVIDER == "sarvam":
        pass   # SARVAM_API_KEY is already checked at the bottom of this function
    elif LLM_PROVIDER == "cloudflare":
        checks.append(("CLOUDFLARE_ACCOUNT_ID", CLOUDFLARE_ACCOUNT_ID))
        checks.append(("CLOUDFLARE_API_TOKEN", CLOUDFLARE_API_TOKEN))
    elif LLM_PROVIDER == "gemini":
        checks.append(("GEMINI_API_KEY", GEMINI_API_KEY))
    else:
        checks.append(("GROQ_API_KEY", GROQ_API_KEY))
    checks.append(("SARVAM_API_KEY", SARVAM_API_KEY))
    return [n for n, v in checks if not v]
# ---------------- Twilio (phone call) config ----------------
PHONE_RATE = 8000               # Twilio Media Streams is fixed at 8kHz mu-law

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
# PUBLIC_BASE_URL = your ngrok (or other tunnel) https URL, no trailing slash
# e.g. https://abcd1234.ngrok-free.app
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

# Shared secret used to authenticate requests coming FROM PravaahAI
# (POST /api/calls) and requests Eva sends back TO PravaahAI's callback_url.
EVA_API_SECRET = os.environ.get("EVA_API_SECRET", "")

PRAVAAH_API_BASE_URL = os.environ.get("PRAVAAH_API_BASE_URL", "").rstrip("/")

# ---------------- VaniSetu (number provider) config ----------------
VANISETU_WS_URL = os.environ.get("VANISETU_WS_URL", "wss://voice.varnet.in/v1/ai/connect")
VANISETU_TCODE = os.environ.get("VANISETU_TCODE", "")
VANISETU_TOKEN = os.environ.get("VANISETU_TOKEN", "")   # full "Bearer xxxx" string
VANISETU_RATE = 8000   # G.711 mu-law over telephony — same rate as Twilio's PHONE_RATE

# ---------------- VoiceLink (number provider) config ----------------
# Login credentials come per-call from PravaahAI (each owner has their own
# VoiceLink account), NOT from env vars — mirrors how Twilio creds arrive
# per-call rather than being global to this service.
VOICELINK_BASE_URL = os.environ.get("VOICELINK_BASE_URL", "https://app.voicelink.co.in").rstrip("/")
# Confirmed via VoiceLink's WebSocket Events docs: media_format.encoding is
# "audio/alaw" at 8kHz — this is A-law, NOT mu-law like VaniSetu/Twilio.
VOICELINK_CODEC = os.environ.get("VOICELINK_CODEC", "alaw")
VOICELINK_RATE = int(os.environ.get("VOICELINK_RATE", 8000))


def _e164(num: str) -> str:
    """Best-effort normalize to E.164 (assumes country code is already included)."""
    num = (num or "").strip().replace(" ", "").replace("-", "")
    if num and not num.startswith("+"):
        num = "+" + num
    return num


TWILIO_PHONE_NUMBER = _e164(os.environ.get("TWILIO_PHONE_NUMBER", ""))  # the Twilio number that calls you
MY_PHONE_NUMBER = _e164(os.environ.get("MY_PHONE_NUMBER", ""))          # your verified number, gets called

twilio_client = (
    TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN
    else None
)

# call_id -> {"agent": {...}, "lead": {...}, "callback_url": "...", "created_at": epoch}
# Populated by POST /api/calls, consumed by /ws/twilio-outbound/<call_id> once
# Twilio opens the Media Stream socket for that call.
_pending_calls_lock = threading.Lock()
PENDING_CALLS = {}

# call_id -> EvaSession, populated once VoiceLink's websocket actually
# connects. Unlike VaniSetu's shared multiplexed socket, each VoiceLink
# call gets its own websocket scoped by call_id in the URL, so this is a
# simple direct lookup rather than a FIFO-matching table.
VOICELINK_BRIDGES = {}   # call_id -> VoiceLinkBridge (replaces VOICELINK_SESSIONS)

# ---------------- VoiceLink: transcript + recording URL are delivered together ----------------
# Transcript comes from agent.py when the call ends; recordingUrl comes later in
# VoiceLink's call.completed webhook. We hold both here and POST ONE callback to
# Pravaah once both are in (or after RECORDING_WAIT_SECS, without the recording).
# NOTE: in-memory -> run gunicorn with a single worker (as you do now for websockets).
RECORDING_WAIT_SECS = int(os.environ.get("EVA_RECORDING_WAIT_SECS", 90))
CALL_RESULTS = {}
_call_results_lock = threading.Lock()


def _new_call_result():
    return {
        "callback_url": None, "transcript": None, "status": None,
        "hangup_reason": "completed", "hangup_cause": "", "answered": False,
        "duration_secs": 0,
        "recording_url": None, "recording_done": False,
        "sent": False, "timer": None,
        "extra": {},          # extra fields merged into the Pravah callback (used by IVR)
    }


def _arm_finalize_timer(call_id):
    """Safety net: if call.completed never arrives, send without recording."""
    with _call_results_lock:
        e = CALL_RESULTS.setdefault(call_id, _new_call_result())
        if e["sent"] or e["timer"]:
            return
        t = threading.Timer(RECORDING_WAIT_SECS, _finalize_call, args=(call_id, True))
        t.daemon = True
        e["timer"] = t
        t.start()


def _classify_call_result(e, transcript):
    """Returns: answered | answered_no_reply | cut_by_user | not_answered"""
    reason = (e.get("hangup_reason") or "").lower()
    m = re.match(r"\s*(\d+)", str(e.get("hangup_cause") or ""))
    code = m.group(1) if m else ""
    lead_spoke = any(t.get("role") == "lead" for t in (transcript or []))
    try:
        dur = float(e.get("duration_secs") or 0)
    except (TypeError, ValueError):
        dur = 0.0

    if not (e.get("answered") or lead_spoke):
        # never picked up. Cause 16 while ringing = customer rejected/cut it; anything else = no answer/busy/unreachable
        return "cut_by_user" if code == "16" else "not_answered"
    if lead_spoke:
        return "answered"
    if reason == "hangup_during_playback":
        return "cut_by_user"
    if (e.get("extra") or {}).get("call_mode") not in ("ivr", "manual") and dur < 10:
        return "cut_by_user"          # picked up and hung up straight away
    return "answered_no_reply"        # picked up but stayed silent


def _post_call_result(callback_url, payload):
    """POST to Pravah, 3 attempts."""
    for attempt in range(3):
        try:
            r = requests.post(
                callback_url,
                headers={"X-Eva-Secret": EVA_API_SECRET, "Content-Type": "application/json"},
                json=payload, timeout=15,
            )
            log("CALLBACK", f"{payload.get('call_id')} -> Pravah HTTP {r.status_code} result={payload.get('call_result')}")
            if r.status_code < 500:
                return r.status_code < 300
        except Exception as ex:
            log("CALLBACK", f"POST failed for {payload.get('call_id')} (try {attempt + 1}): {ex}")
        time.sleep(2)
    return False


def _finalize_call(call_id, force=False):
    with _call_results_lock:
        e = CALL_RESULTS.get(call_id)
        if not e or e["sent"]:
            return
        ready = e["transcript"] is not None and e["recording_done"]
        if not (ready or force):
            return
        if not e["callback_url"]:
            if force:
                CALL_RESULTS.pop(call_id, None)
            return
        e["sent"] = True
        if e["timer"]:
            e["timer"].cancel()
        transcript = e["transcript"] or []
        call_result = _classify_call_result(e, transcript)
        payload = {
            "call_id": call_id,
            "status": e["status"] or ("completed" if transcript else "no_response"),
            "hangup_reason": e["hangup_reason"],
            "hangup_cause": e.get("hangup_cause", ""),
            "answered": bool(e.get("answered")),
            "call_result": call_result,
            "duration_secs": e["duration_secs"],
            "transcript": transcript,
            "recording_url": e["recording_url"] or "",
        }
        payload.update(e.get("extra") or {})
        callback_url = e["callback_url"]

    _post_call_result(callback_url, payload)
    with _call_results_lock:
        CALL_RESULTS.pop(call_id, None)



def log(stage: str, msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{stage}] {msg}", flush=True)


def is_speakable(text: str) -> bool:
    return bool(SPEAKABLE_RE.search(text))


# --- NEW ---
def detect_lang(text: str) -> str:
    """Script-based detection: Devanagari -> hi, Gujarati script -> gu, Gurmukhi -> pa, etc.
    Anything else (including Roman-script Hinglish) falls back to English."""
    for rx, code in SCRIPT_LANG_RES:
        if rx.search(text or ""):
            return code
    return "en"


def _build_ulaw_decode_table():
    """Standard ITU-T G.711 mu-law -> linear16 expansion, precomputed once
    so per-chunk volume checks on phone audio don't redo the math per byte."""
    table = []
    for i in range(256):
        u_val = ~i & 0xFF
        t = ((u_val & 0x0F) << 3) + 0x84
        t <<= (u_val & 0x70) >> 4
        val = (t - 0x84) if (u_val & 0x80) else (0x84 - t)
        table.append(val)
    return table


_ULAW_TO_LINEAR16 = _build_ulaw_decode_table()


def _rms_pcm16(data: bytes) -> float:
    """RMS volume of raw linear16 (browser mic) audio."""
    usable_len = len(data) - (len(data) % 2)
    if usable_len < 2:
        return 0.0
    samples = array.array('h')
    samples.frombytes(data[:usable_len])
    if not samples:
        return 0.0
    total = sum(s * s for s in samples)
    return math.sqrt(total / len(samples))


def _rms_mulaw(data: bytes) -> float:
    """RMS volume of raw mu-law (phone) audio, decoded to linear first."""
    if not data:
        return 0.0
    total = 0
    for b in data:
        v = _ULAW_TO_LINEAR16[b]
        total += v * v
    return math.sqrt(total / len(data))


def _build_alaw_decode_table():
    """Standard ITU-T G.711 A-law -> linear16 expansion, precomputed once.
    VoiceLink's WebSocket audio is A-law, NOT mu-law — see WebSocket Events
    docs (media_format.encoding = 'audio/alaw')."""
    table = []
    for i in range(256):
        a_val = i ^ 0x55
        t = (a_val & 0x0F) << 4
        seg = (a_val & 0x70) >> 4
        if seg == 0:
            t += 8
        elif seg == 1:
            t += 0x108
        else:
            t += 0x108
            t <<= (seg - 1)
        val = t if (a_val & 0x80) else -t
        table.append(val)
    return table


_ALAW_TO_LINEAR16 = _build_alaw_decode_table()


def _rms_alaw(data: bytes) -> float:
    """RMS volume of raw A-law (VoiceLink) audio, decoded to linear first."""
    if not data:
        return 0.0
    total = 0
    for b in data:
        v = _ALAW_TO_LINEAR16[b]
        total += v * v
    return math.sqrt(total / len(data))


def render_call_vars(text: str, lead: dict) -> str:
    """Replace {{name}}, {{business_name}}, etc. with lead field values
    (same merge-tag convention as PravaahAI's templates)."""
    for key in ("name", "business_name", "email", "phone", "website", "description"):
        text = text.replace("{{%s}}" % key, str((lead or {}).get(key, "") or ""))
    text = re.sub(r'[“”„«»"]', "", text)
    return text


app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/static")
sock = Sock(app)

# ---------------- LiveKit TTS: sync/async bridge ----------------
# inference.TTS is async-only; everything else in this file is sync
# (Flask + flask-sock under gunicorn's gevent worker). One persistent
# background thread owns a single asyncio loop for the life of the
# process — every session bridges its TTS calls into it via
# run_coroutine_threadsafe rather than each spinning up its own loop.
class _AsyncLoopRunner:
    """Does NOT start its own loop. Under gunicorn+gevent, threading.Thread
    (even the 'original' one) still becomes a greenlet on the SAME OS thread,
    and asyncio allows only one running loop per thread. So we reuse the loop
    livekit_bridge already runs, and just open the LiveKit http context on it."""
    def __init__(self):
        self.loop = livekit_bridge._bridge_loop.loop
        self.request_q = None
        self._is_ready = False
        self._error = None
        self._tasks = set()

        asyncio.run_coroutine_threadsafe(self._main(), self.loop)

        # gevent-cooperative wait; never hang the gunicorn worker forever.
        deadline = time.time() + 20
        while not self._is_ready and self._error is None and time.time() < deadline:
            time.sleep(0.05)
        if self._error is not None:
            raise RuntimeError(f"LiveKit TTS loop failed to start: {self._error!r}")
        if not self._is_ready:
            raise RuntimeError("LiveKit TTS loop did not start within 20s")

    async def _main(self):
        try:
            self.request_q = asyncio.Queue()
            if hasattr(http_context, "_new_session_ctx"):
                # livekit-agents 1.x: sets the http-session ContextVar in THIS
                # task's context; every child task created below inherits it.
                http_context._new_session_ctx()
                self._is_ready = True
                await self._serve()
            else:
                async with http_context.open():
                    self._is_ready = True
                    await self._serve()
        except BaseException as e:
            self._error = e
            log("TTS-LOOP", f"fatal: {type(e).__name__}: {e!r}")
            raise

    async def _serve(self):
        while True:
            coro_fn = await self.request_q.get()
            t = asyncio.create_task(coro_fn())
            self._tasks.add(t)
            t.add_done_callback(self._tasks.discard)

    def submit(self, coro_fn):
        """coro_fn: zero-arg callable returning a coroutine. Safe from any thread/greenlet."""
        self.loop.call_soon_threadsafe(self.request_q.put_nowait, coro_fn)


_async_loop = _AsyncLoopRunner()

def _stream_livekit_tts(tts_client, text, lang):
    """Bridges LiveKit's async TTS generator (real thread) into a sync
    generator of (pcm16_bytes, sample_rate) for gevent code. Uses a deque +
    cooperative sleep instead of queue.Queue, so it's safe across the
    real-thread / gevent boundary."""
    out = collections.deque()
    SENTINEL = object()

    async def _pump():
        try:
            try:
                tts_client.update_options(language=lang, speed=TTS_SPEED)
            except TypeError:
                # Installed TTS build doesn't accept 'speed' - language-only is fine.
                tts_client.update_options(language=lang)
            async for audio in tts_client.synthesize(text):
                frame = audio.frame
                out.append((bytes(frame.data), frame.sample_rate))
        except Exception as e:
            out.append(e)
        finally:
            out.append(SENTINEL)

    _async_loop.submit(_pump)

    last_activity = time.time()
    while True:
        try:
            item = out.popleft()
        except IndexError:
            if time.time() - last_activity > 20:
                raise TimeoutError("LiveKit TTS produced no audio for 20s")
            time.sleep(0.005)   # gevent-cooperative
            continue
        last_activity = time.time()
        if item is SENTINEL:
            return
        if isinstance(item, Exception):
            raise item
        yield item

def sarvam_tts_synthesize(text: str, lang: str, voice: str, sample_rate: int):
    """One blocking REST call to Sarvam TTS. Returns (pcm16_bytes,
    actual_sample_rate). Sarvam has no TTS websocket/streaming, so this is
    a plain requests.post() - fine since it only ever runs on the per-call
    TTS-loop thread (_tts_loop), never on a gevent-cooperative thread.
    actual_sample_rate may differ from the requested one (snapped to the
    nearest rate Sarvam supports) - _tts_loop's existing audioop.ratecv
    resample step (already there from the old LiveKit TTS path) silently
    handles that mismatch, so callers don't need to care."""
    sr = min(SARVAM_SUPPORTED_RATES, key=lambda r: abs(r - sample_rate))
    resp = requests.post(
        SARVAM_TTS_URL,
        headers={"API-Subscription-Key": SARVAM_API_KEY, "Content-Type": "application/json"},
        json={
            "inputs": [text],
            "target_language_code": SARVAM_LANG_CODES.get(lang, "en-IN"),
            "speaker": voice,
            "pace": TTS_SPEED,
            "temperature": SARVAM_TTS_TEMPERATURE,  # ignored/harmless if model is ever switched back to bulbul:v2
            "speech_sample_rate": sr,
            "enable_preprocessing": True,
            "model": SARVAM_TTS_MODEL,
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    wav_bytes = base64.b64decode(data["audios"][0])
    # Sarvam returns a standard 44-byte-header PCM WAV at speech_sample_rate.
    pcm16 = wav_bytes[44:] if wav_bytes[:4] == b"RIFF" else wav_bytes
    return pcm16, sr


def _stream_sarvam_tts(text: str, lang: str, voice: str, sample_rate: int, chunk_ms: int = 100):
    """Same (pcm_bytes, src_rate) generator contract _tts_loop already
    consumes (previously satisfied by _stream_livekit_tts). Sarvam gives us
    the whole sentence in one shot, so we chunk it ourselves afterward -
    keeps _tts_loop's interrupt_flag check responsive mid-sentence instead
    of sending one giant blob."""
    pcm16, sr = sarvam_tts_synthesize(text, lang, voice, sample_rate)
    chunk_bytes = max(2, int(sr * (chunk_ms / 1000.0)) * 2)
    chunk_bytes -= chunk_bytes % 2
    for i in range(0, len(pcm16), chunk_bytes):
        yield pcm16[i:i + chunk_bytes], sr

# ============================================================
# One EvaSession per WebSocket connection
# ============================================================
class EvaSession:
    # --- NEW ---
    def __init__(self, ws, mode: str = "browser",
                 call_id: str = None, agent: dict = None, lead: dict = None,
                 callback_url: str = None, meeting: dict = None,
                 transport: str = "twilio", vanisetu_session_id: int = None):
        agent = agent or {}
        lead = lead or {}
        self.meeting = meeting or {}

        self.ws = ws
        self.mode = mode                # "browser" or "phone"               # "browser" or "phone"
        self.transport = transport      # "twilio" | "vanisetu" — only meaningful when mode == "phone"
        self.vanisetu_session_id = vanisetu_session_id  # numeric id VaniSetu assigned this call
        self.stream_sid = None          # set once Twilio's "start" event arrives (phone + twilio only)
        self.ws_lock = threading.Lock()
        self.stop_event = threading.Event()

        # --- barge-in / turn-taking state ---
        self.eva_speaking = threading.Event()    # set while audio is actively being sent
        self.interrupt_flag = threading.Event()  # set when the user barges in mid-response
        self.pending_lock = threading.Lock()
        self.pending_transcript = ""             # accumulates final STT chunks pre-response
        self.pending_timer = None                # fires RESPONSE_DELAY_SECS after last speech
        self.barge_in_grace_until = 0.0          # epoch time; ignore VAD barge-in until this passes

        # VAD fires a "candidate" barge-in; it only becomes a real
        # interrupt once actual transcribed speech confirms it (see
        # _dg_speech_started / _dg_transcript). Fixes Eva stopping on
        # every small mic noise.
        self.barge_in_candidate = threading.Event()
        self.barge_in_candidate_lock = threading.Lock()
        self.barge_in_candidate_timer = None

        # True while Eva is in the middle of a single response turn
        # (from the first sentence she starts speaking until she's fully
        # done and gone quiet). Used so the barge-in grace window only
        # fires once per turn instead of re-arming on every sentence.
        self.turn_active = False

        # --- campaign-call metadata (all None/empty for plain browser/dev calls) ---
        self.call_id = call_id
        self.agent = agent
        self.lead = lead
        self.callback_url = callback_url
        self.transcript = []           # [{"role": "lead"|"agent", "text": "...", "ts": epoch}]
        self.call_started_at = None
        self.hangup_reason = "completed"
        self._callback_sent = False
        self._callback_lock = threading.Lock()

        forced_lang = agent.get("language")
        self.forced_language = forced_lang if forced_lang in SUPPORTED_LANGUAGES else None

        # Voice selection: agent.speaker (explicit name) always wins.
        # Otherwise pick from SPEAKER_MAP by gender — previously agent.gender
        # was stored but never actually used, so every call silently used
        # DEFAULT_SPEAKER no matter what gender was picked in the dashboard.
        # Set ONCE here and never reassigned mid-call, which is what keeps
        # Eva's voice/tone consistent across a call even as language shifts
        # sentence-to-sentence.
        # --- NEW ---
        agent_gender = agent.get("gender") if agent.get("gender") in ("male", "female") else "female"
        self.gender = agent_gender
        self.voice_name = VOICE_MALE if agent_gender == "male" else VOICE_FEMALE

        self.max_duration_secs = int(agent.get("max_duration_secs") or 0) or None
        self.min_duration_secs = int(agent.get("min_duration_secs") or 0) or None

        self.user_text_q: "queue.Queue[str]" = queue.Queue()
        self.sentence_q: "queue.Queue[tuple]" = queue.Queue()

        self.history = []

        persona_name = (agent.get("name") or "").strip()
        custom_prompt = re.split(r"You can also book meetings on the account owner",
                                 (agent.get("system_prompt") or ""))[0].strip()
        base_prompt = custom_prompt or (
            "You are a warm, concise voice assistant on a call for this business. "
            "Reply in 1-3 short spoken sentences."
        )
        if persona_name:
            base_prompt += f" Your name is {persona_name}; never say any other name."
        else:
            base_prompt += " Never state a name for yourself."
        if lead:
            base_prompt += (
                f" You're speaking with {lead.get('name', 'the lead')} from "
                f"{lead.get('business_name', 'their business')}."
            )

        if self.forced_language:
            base_prompt += " " + reply_language_rule(self.forced_language)
        else:
            base_prompt += (
                " Default to English; if the user speaks Hindi, reply in Hinglish (Roman script, never Devanagari). "
                "If the user clearly speaks another Indian language (Gujarati, Punjabi, Marathi, Bengali, Tamil, "
                "Telugu, Kannada, Malayalam, Odia, Haryanvi...), reply in that same language in its own native script."
            )

        gender_forms = "masculine" if self.gender == "male" else "feminine"
        base_prompt += f" In Hindi/Hinglish and every other gendered Indian language always use {gender_forms} self-reference verb forms, consistently."

        base_prompt += (
            " LENGTH RULE (always follow): this is a live phone call. Reply in 1-2 short "
            "spoken sentences, under 500 characters, never more than 650. No lists, no long "
            "explanations - give the key point only and let them ask for more."
        )
        base_prompt += (
            " Never reply using only emojis. Stay in character always: never say you're "
            "an AI, bot, or assistant; if asked, answer briefly like a real staff member, "
            "no technical/infra details. Speak like a real phone call - contractions, warm, "
            "brief, never repeat back what they said."
        )

        if self.meeting:
            base_prompt += (
                f" You can book meetings ({self.meeting.get('duration_minutes', 30)} min). "
                f"Available: {self.meeting.get('availability_text', '')}. The lead's name/phone "
                "are already known - only ask their preferred date/time. On confirmation output "
                "EXACTLY: BOOK_MEETING: YYYY-MM-DD HH:MM (24h UTC) - nothing else on that line, "
                "never say it aloud."
            )

        self.system_prompt = {"role": "system", "content": base_prompt}

        config = DeepgramClientOptions(options={"keepalive": "true"})
        self.deepgram = DeepgramClient(DEEPGRAM_API_KEY, config)
        self.dg_connection = self.deepgram.listen.websocket.v("1")
        self.dg_connection.on(LiveTranscriptionEvents.Open, self._dg_open)
        self.dg_connection.on(LiveTranscriptionEvents.Transcript, self._dg_transcript)
        self.dg_connection.on(LiveTranscriptionEvents.SpeechStarted, self._dg_speech_started)
        self.dg_connection.on(LiveTranscriptionEvents.Error, self._dg_error)
        self.dg_connection.on(LiveTranscriptionEvents.Close, self._dg_close)

        # TTS is now a plain Sarvam REST call per sentence (see
        # sarvam_tts_synthesize / _stream_sarvam_tts above) - no persistent
        # client object needed here like the old LiveKit inference.TTS.
    # ---------- outbound helpers ----------
    def _send_json(self, obj):
        # Twilio's Media Stream socket only understands its own event schema
        # (media/mark/clear) - our status/transcript chatter is browser-only.
        if self.mode == "phone":
            return
        with self.ws_lock:
            try:
                self.ws.send(json.dumps(obj))
            except Exception:
                pass

    def _send_raw(self, obj):
        """Like _send_json but NOT skipped in phone mode - for real Twilio
        Media Stream control events (e.g. "clear")."""
        with self.ws_lock:
            try:
                self.ws.send(json.dumps(obj))
            except Exception:
                pass

    def _send_audio(self, audio_bytes: bytes):
        if self.mode == "phone" and self.transport == "vanisetu":
            # VaniSetu's socket is shared/multiplexed across every call on the
            # account, so audio doesn't go through self.ws at all here — it
            # goes out through the single VaniSetuClient connection, framed
            # with this session's 4-byte session ID.
            if self.vanisetu_session_id is not None:
                vanisetu_client.send_audio(self.vanisetu_session_id, audio_bytes)
            return
        if self.mode == "phone" and self.transport == "voicelink":
            # Confirmed via WebSocket Events docs: outbound audio is a JSON
            # event with a base64 payload, e.g. {"event":"media","media":
            # {"payload":"<base64>"}} — NOT raw binary. Only send once
            # "start" has actually arrived (stream_sid set), mirroring the
            # Twilio outbound path.
            if not self.stream_sid:
                return
            with self.ws_lock:
                try:
                    self.ws.send(json.dumps({
                        "event": "media",
                        "media": {"payload": base64.b64encode(audio_bytes).decode("ascii")},
                    }))
                except Exception:
                    pass
            return
        with self.ws_lock:
            try:
                if self.mode == "phone":
                    if not self.stream_sid:
                        return
                    self.ws.send(json.dumps({
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": base64.b64encode(audio_bytes).decode("ascii")},
                    }))
                else:
                    self.ws.send(audio_bytes)
            except Exception:
                pass

    # ---------- barge-in ----------
    def _interrupt_playback(self):
        """Called once a barge-in is CONFIRMED (real transcribed speech, not
        just VAD noise). Stops Eva immediately: drops anything queued to be
        spoken, breaks the in-flight TTS stream, and tells the client/Twilio
        to flush any audio already sent but not yet played."""
        self.interrupt_flag.set()
        with self.sentence_q.mutex:
            self.sentence_q.queue.clear()

        with self.barge_in_candidate_lock:
            self.barge_in_candidate.clear()
            if self.barge_in_candidate_timer:
                self.barge_in_candidate_timer.cancel()
                self.barge_in_candidate_timer = None

        # The turn Eva was mid-way through is over now - the next thing
        # she says (the new answer) is a fresh turn and earns its own
        # grace window (see _enqueue_sentence).
        self.turn_active = False

        if self.mode == "phone":
            if self.transport == "vanisetu":
                if self.vanisetu_session_id is not None:
                    vanisetu_client.send_command(self.vanisetu_session_id, {"command": "FLUSH_MEDIA"})
            elif self.transport == "voicelink":
                if self.stream_sid:
                    self._send_raw({"event": "clear", "stream_sid": self.stream_sid})
            elif self.stream_sid:
                self._send_raw({"event": "clear", "streamSid": self.stream_sid})
        else:
            self._send_json({"type": "interrupt"})
            self._send_json({"type": "status", "state": "listening"})

        self.eva_speaking.clear()

    # ---------- Deepgram callbacks ----------
    def _dg_open(self, *_a, **_k):
        log("STT", "Deepgram connection open.")

    def _dg_speech_started(self, *_a, **_k):
        # Ignore VAD triggers that land inside the protection window right
        # after Eva was just queued to say something (see BARGE_IN_GRACE_SECS)
        # - these are almost always a false positive from telephony line
        # noise at call/stream start, not the caller actually talking.
        if time.time() < self.barge_in_grace_until:
            return

        if not (self.eva_speaking.is_set() or not self.sentence_q.empty()):
            return

        # IMPORTANT: VAD alone does NOT interrupt Eva anymore. Deepgram's
        # SpeechStarted fires on any energy spike (coughs, breathing, mic
        # bumps), which used to cut Eva off on the tiniest noise. Instead we
        # mark this as a *candidate* barge-in and wait up to
        # BARGE_IN_CONFIRM_TIMEOUT_SECS for _dg_transcript to actually see
        # real transcribed words - only then do we treat it as a genuine
        # barge-in and stop her. If nothing gets transcribed in time, this
        # candidate silently expires (was just noise).
        self._arm_barge_in_candidate(source="vad")

    def _check_volume_barge_in(self, raw_audio: bytes):
        """Second, independent barge-in trigger based on raw mic/line
        volume, running alongside Deepgram's VAD (_dg_speech_started). Real
        background noise - traffic, a fan, other people across the room -
        is quieter than the user's own voice on their own mic/handset, so
        it normally never crosses the threshold. Like VAD, this only arms a
        candidate; _dg_transcript still has to see actual transcribed words
        before Eva is interrupted, so a loud one-off noise that isn't
        speech won't trigger anything on its own."""
        if time.time() < self.barge_in_grace_until:
            return
        if not (self.eva_speaking.is_set() or not self.sentence_q.empty()):
            return
        if self.barge_in_candidate.is_set():
            return  # already armed - no need to recompute volume

        if self.mode == "phone":
            if self.transport == "voicelink":
                rms = _rms_alaw(raw_audio)
                threshold = BARGE_IN_MIN_VOLUME_ALAW
            else:
                rms = _rms_mulaw(raw_audio)
                threshold = BARGE_IN_MIN_VOLUME_MULAW
        else:
            rms = _rms_pcm16(raw_audio)
            threshold = BARGE_IN_MIN_VOLUME_LINEAR16

        if rms >= threshold:
            self._arm_barge_in_candidate(source=f"volume({rms:.0f}>={threshold})")

    def _arm_barge_in_candidate(self, source: str):
        """Shared by both barge-in triggers (VAD and volume). Arms a
        candidate interruption that _dg_transcript will confirm - or let
        silently expire - once it sees (or doesn't see) real transcribed
        words within BARGE_IN_CONFIRM_TIMEOUT_SECS."""
        with self.barge_in_candidate_lock:
            already_armed = self.barge_in_candidate.is_set()
            self.barge_in_candidate.set()
            if self.barge_in_candidate_timer:
                self.barge_in_candidate_timer.cancel()
            self.barge_in_candidate_timer = threading.Timer(
                BARGE_IN_CONFIRM_TIMEOUT_SECS, self._clear_barge_in_candidate
            )
            self.barge_in_candidate_timer.daemon = True
            self.barge_in_candidate_timer.start()
        if not already_armed:
            log("BARGE-IN", f"[{self.call_id or 'browser'}] candidate armed via {source}")

    def _clear_barge_in_candidate(self):
        """Candidate barge-in expired unconfirmed - it was noise, not speech."""
        with self.barge_in_candidate_lock:
            self.barge_in_candidate.clear()
            self.barge_in_candidate_timer = None
    

    def _dg_transcript(self, *_a, result=None, **_k):
        if result is None:
            return
        transcript = result.channel.alternatives[0].transcript
        if not transcript:
            return

        # A candidate barge-in (raised by VAD in _dg_speech_started) is only
        # confirmed once we see real transcribed words - this is what
        # filters out noise/breath triggers vs an actual interruption.
        if self.barge_in_candidate.is_set() and len(transcript.strip()) >= BARGE_IN_CONFIRM_MIN_CHARS:
            with self.barge_in_candidate_lock:
                self.barge_in_candidate.clear()
                if self.barge_in_candidate_timer:
                    self.barge_in_candidate_timer.cancel()
                    self.barge_in_candidate_timer = None
            if self.eva_speaking.is_set() or not self.sentence_q.empty():
                with self.pending_lock:
                    if self.pending_timer:
                        self.pending_timer.cancel()
                        self.pending_timer = None
                log("MAIN", f"[{self.call_id or 'browser'}] Barge-in confirmed - interrupting Eva.")
                self._interrupt_playback()

        if result.is_final:
            log("STT", f"Final transcript: {transcript}")
            self._send_json({"type": "user_transcript", "text": transcript, "final": True})
            self.transcript.append({"role": "lead", "text": transcript, "ts": time.time()})
            self._queue_with_pause(transcript)
        else:
            self._send_json({"type": "user_transcript", "text": transcript, "final": False})

    def _dg_error(self, *_a, error=None, **_k):
        log("STT", f"ERROR: {error}")

    def _dg_close(self, *_a, **_k):
        log("STT", "Deepgram connection closed.")

    # ---------- natural turn-taking pause ----------
    def _queue_with_pause(self, transcript: str):
        """Buffer finalized speech and only hand it to the LLM once the user
        has been quiet for a short pause. Each new final transcript resets
        the timer, so short mid-thought pauses don't get cut off.

        The pause length now varies instead of being a single fixed number:
        - short replies ("yes", "sounds good") get a quicker turnaround,
          since that's how people actually respond to quick confirmations
        - a small random jitter is added on top either way, so Eva never
          replies on the exact same beat twice - that uniformity is what
          made her feel robotic."""
        with self.pending_lock:
            self.pending_transcript = (self.pending_transcript + " " + transcript).strip()
            if self.pending_timer:
                self.pending_timer.cancel()

            word_count = len(self.pending_transcript.split())
            base_delay = (
                SHORT_UTTERANCE_DELAY_SECS
                if word_count <= SHORT_UTTERANCE_MAX_WORDS
                else RESPONSE_DELAY_SECS
            )
            delay = base_delay + random.uniform(0, RESPONSE_DELAY_JITTER_SECS)

            self.pending_timer = threading.Timer(delay, self._flush_pending_transcript)
            self.pending_timer.daemon = True
            self.pending_timer.start()

    def _flush_pending_transcript(self):
        with self.pending_lock:
            text = self.pending_transcript.strip()
            self.pending_transcript = ""
            self.pending_timer = None
        if text:
            self.user_text_q.put(text)

    # ---------- lifecycle ----------
    def start(self):
        if self.mode == "phone":
            if self.transport == "voicelink":
                # Deepgram's LIVE STREAMING endpoint does not accept
                # encoding=alaw (alaw is documented for pre-recorded/batch
                # transcription and for Deepgram TTS output only) — sending
                # it caused Deepgram to reject the websocket handshake with
                # HTTP 400. We decode VoiceLink's A-law audio to linear16
                # ourselves in feed_audio() below, so tell Deepgram to
                # expect linear16 here, not alaw.
                encoding, sample_rate = "linear16", VOICELINK_RATE
            elif self.transport == "vanisetu":
                encoding, sample_rate = "mulaw", VANISETU_RATE
            else:
                encoding, sample_rate = "mulaw", PHONE_RATE
        else:
            encoding, sample_rate = "linear16", MIC_RATE

        options = LiveOptions(
            model="nova-3",
            language="multi",
            smart_format=True,
            interim_results=True,
            endpointing=200,          # ms of silence treated as a likely pause — steadier utterance-end detection
            utterance_end_ms="600",
            vad_events=True,
            encoding=encoding,
            sample_rate=sample_rate,
            channels=1,
        )
        if not self.dg_connection.start(options):
            log("STT", "Failed to start Deepgram connection.")
            self._send_json({"type": "error", "message": "Could not start speech recognition."})
            return False

        threading.Thread(target=self._llm_loop, daemon=True, name="LLM").start()
        threading.Thread(target=self._tts_loop, daemon=True, name="TTS").start()
        if self.max_duration_secs:
            threading.Thread(target=self._max_duration_watchdog, daemon=True, name="MaxDuration").start()
        return True

    def feed_audio(self, data: bytes):
        try:
            self._check_volume_barge_in(data)
        except Exception as e:
            log("BARGE-IN", f"volume check error: {e}")
        try:
            # VoiceLink's wire audio is A-law, but Deepgram's live
            # streaming endpoint rejects encoding=alaw with HTTP 400 (see
            # start()). Decode to linear16 before forwarding so Deepgram
            # actually accepts the stream; other transports are untouched.
            if self.mode == "phone" and self.transport == "voicelink":
                data_for_dg = audioop.alaw2lin(data, 2)
            else:
                data_for_dg = data
            self.dg_connection.send(data_for_dg)
        except Exception as e:
            log("STT", f"send error: {e}")

    def feed_text(self, text: str):
        """Allow a typed message to skip STT and go straight to the LLM."""
        self._send_json({"type": "user_transcript", "text": text, "final": True})
        self.transcript.append({"role": "lead", "text": text, "ts": time.time()})
        self.user_text_q.put(text)

    def _enqueue_sentence(self, text: str, lang: str):
        """Puts a line on the TTS queue. Only the FIRST sentence of a brand
        new turn (Eva was fully idle beforehand) opens the barge-in grace
        window. This used to re-open on every single sentence whenever the
        queue happened to be momentarily empty between sentences of the
        SAME reply - which is most of the time during a multi-sentence
        answer - so Eva was almost continuously "protected" and a real
        interruption could never land. Now it fires once, right as she
        starts talking, and stays off for the rest of that turn."""
        if self.sentence_q.empty() and not self.turn_active:
            self.barge_in_grace_until = time.time() + BARGE_IN_GRACE_SECS
        self.turn_active = True
        self.sentence_q.put((text, lang))

    def speak(self, text: str, lang: str = "en"):
        """Queue a line straight to TTS, bypassing the LLM (e.g. an opening greeting)."""
        self._enqueue_sentence(text, lang)

    def _trigger_meeting_booking(self, date_str: str, time_str: str):
        """Called from the LLM loop the instant a BOOK_MEETING tag is seen.
        Blocking — runs synchronously inside the LLM thread, which briefly
        pauses further token consumption from Mistral for that turn. Fine
        for a single short HTTP call; worth revisiting if this ever needs
        to be non-blocking."""
        if not self.meeting:
            return
        requested_iso = f"{date_str}T{time_str}:00"
        confirmed, message = book_meeting_via_pravaah(self.meeting, self.lead, self.call_id, requested_iso)
        log("MEETING", f"[{self.call_id or 'test'}] requested={requested_iso} confirmed={confirmed}")
        self.history.append({
            "role": "system",
            "content": f"[Booking result: {'confirmed' if confirmed else 'not available'}] {message}",
        })
        self._enqueue_sentence(message, detect_lang(message))

    def close(self):
        self.stop_event.set()
        with self.pending_lock:
            if self.pending_timer:
                self.pending_timer.cancel()
                self.pending_timer = None
        with self.barge_in_candidate_lock:
            if self.barge_in_candidate_timer:
                self.barge_in_candidate_timer.cancel()
                self.barge_in_candidate_timer = None
            self.barge_in_candidate.clear()
        if self.mode == "phone" and self.transport == "vanisetu" and self.vanisetu_session_id is not None:
            vanisetu_client.send_command(self.vanisetu_session_id, {"command": "HANGUP"})
            vanisetu_client.unregister_session(self.vanisetu_session_id)
        try:
            self.dg_connection.finish()
        except Exception:
            pass

    def _max_duration_watchdog(self):
        """Hangs the call up once it's run past agent.max_duration_secs."""
        if self.stop_event.wait(self.max_duration_secs):
            return  # call already ended naturally
        log("MAIN", f"Call {self.call_id} hit max duration ({self.max_duration_secs}s), hanging up.")
        self.hangup_reason = "max_duration_reached"
        self.close()
        try:
            self.ws.close()
        except Exception:
            pass

    def _finish_and_callback(self, hangup_reason: str = None):
        """POSTs the final transcript back to PravaahAI. Safe to call more
        than once — only fires the HTTP request the first time."""
        with self._callback_lock:
            if self._callback_sent or not self.callback_url or not self.call_id:
                return
            self._callback_sent = True
        duration_secs = round(time.time() - self.call_started_at, 1) if self.call_started_at else 0
        payload = {
            "call_id": self.call_id,
            "status": "completed" if self.transcript else "no_response",
            "hangup_reason": hangup_reason or self.hangup_reason,
            "duration_secs": duration_secs,
            "transcript": self.transcript,
        }
        try:
            requests.post(
                self.callback_url,
                headers={"X-Eva-Secret": EVA_API_SECRET, "Content-Type": "application/json"},
                json=payload, timeout=15,
            )
        except Exception as e:
            log("MAIN", f"callback POST failed for {self.call_id}: {e}")

    # ---------- LLM loop ----------
    def _trim_history(self):
        if len(self.history) > MAX_HISTORY_MESSAGES:
            self.history = self.history[-MAX_HISTORY_MESSAGES:]

    def _stream_chat(self, client: httpx.Client, messages):
        """Dispatches to whichever provider LLM_PROVIDER selects. All
        generators below yield plain text deltas, so nothing downstream
        (sentence-splitting, TTS queue, BOOK_MEETING detection) needs to
        know or care which LLM is actually live."""
        if LLM_PROVIDER == "sarvam":
            yield from self._stream_chat_sarvam(client, messages)
        elif LLM_PROVIDER == "cloudflare":
            yield from self._stream_chat_cloudflare(client, messages)
        elif LLM_PROVIDER == "gemini":
            yield from self._stream_chat_gemini(client, messages)
        else:
            yield from self._stream_chat_groq(client, messages)

    def _stream_chat_sarvam(self, client: httpx.Client, messages):
        """Sarvam chat completions (OpenAI-style SSE). Sarvam is strict about
        message shape, so we normalize first: only one leading system message,
        later system notes (booking result, per-turn language note) get folded
        into the message before them, consecutive same-role messages are
        merged, and the first non-system message must be from the user."""
        system_parts, chat = [], []
        for m in messages:
            role, text = m.get("role"), (m.get("content") or "")
            if not text:
                continue
            if role == "system":
                if chat:
                    chat[-1]["content"] += "\n" + text
                else:
                    system_parts.append(text)
                continue
            if chat and chat[-1]["role"] == role:
                chat[-1]["content"] += "\n" + text
            else:
                chat.append({"role": role, "content": text})
        while chat and chat[0]["role"] != "user":
            chat.pop(0)

        out_messages = chat
        if system_parts:
            out_messages = [{"role": "system", "content": "\n\n".join(system_parts)}] + chat

        payload = {
            "model": SARVAM_LLM_MODEL,
            "messages": out_messages,
            "stream": True,
            "max_tokens": SARVAM_LLM_MAX_TOKENS,
            "temperature": SARVAM_LLM_TEMPERATURE,
            "reasoning_effort": None,   # no reasoning tokens = cheaper + faster
        }
        headers = {"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"}

        sent_chars = 0
        with client.stream("POST", SARVAM_LLM_URL, json=payload, headers=headers, timeout=30) as resp:
            if resp.status_code >= 400:
                resp.read()
                log("LLM", f"Sarvam HTTP {resp.status_code}: {resp.text[:300]}")
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0]["delta"].get("content")
                except Exception:
                    continue
                if not delta:
                    continue

                # Char cap: once we're past the limit, finish the current
                # sentence and stop, so the reply never ends mid-word.
                if sent_chars + len(delta) >= SARVAM_LLM_MAX_REPLY_CHARS:
                    m_end = SENTENCE_END_RE.search(delta)
                    if m_end:
                        yield delta[:m_end.end()]
                        return
                sent_chars += len(delta)
                yield delta    

    def _stream_chat_groq(self, client: httpx.Client, messages):
        payload = {"model": GROQ_MODEL, "messages": messages, "stream": True}
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        with client.stream("POST", "https://api.groq.com/openai/v1/chat/completions",
                            json=payload, headers=headers, timeout=30) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta
                except Exception:
                    continue
                
    def _stream_chat_cloudflare(self, client: httpx.Client, messages):
        """Cloudflare Workers AI via its OpenAI-compatible endpoint - same
        SSE delta shape as Groq, so this mirrors _stream_chat_groq exactly."""
        payload = {
            "model": CLOUDFLARE_MODEL, "messages": messages, "stream": True,
            "max_tokens": CLOUDFLARE_MAX_TOKENS, "temperature": CLOUDFLARE_TEMPERATURE,
        }
        headers = {"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}", "Content-Type": "application/json"}
        with client.stream("POST", CLOUDFLARE_CHAT_URL, json=payload, headers=headers, timeout=30) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta
                except Exception:
                    continue            

    def _stream_chat_gemini(self, client: httpx.Client, messages):
        """Gemini has no OpenAI-style flat messages list — "system" role
        text goes into a separate systemInstruction block, and the turn
        history uses "user"/"model" roles instead of "user"/"assistant".
        Streams over SSE (alt=sse), same idea as Groq's chunked streaming."""
        system_parts, contents = [], []
        for m in messages:
            role = m.get("role")
            text = m.get("content") or ""
            if not text:
                continue
            if role == "system":
                system_parts.append(text)
            else:
                gem_role = "model" if role == "assistant" else "user"
                # Gemini needs strictly alternating user/model turns - merge
                # consecutive same-role messages instead of sending them as
                # separate turns.
                if contents and contents[-1]["role"] == gem_role:
                    contents[-1]["parts"][0]["text"] += "\n" + text
                else:
                    contents.append({"role": gem_role, "parts": [{"text": text}]})

        payload = {"contents": contents}
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:streamGenerateContent?alt=sse"
        headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
        with client.stream("POST", url, json=payload, headers=headers, timeout=30) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                    parts = obj["candidates"][0]["content"]["parts"]
                    delta = "".join(p.get("text", "") for p in parts)
                    if delta:
                        yield delta
                except Exception:
                    continue

    def _llm_loop(self):
        with httpx.Client() as client:
            while not self.stop_event.is_set():
                try:
                    user_text = self.user_text_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                # Fresh turn - clear out any interrupt flag left over from
                # whatever Eva was saying before this turn started.
                self.interrupt_flag.clear()

                user_lang = self.forced_language or detect_lang(user_text)
                # Tells the model exactly which script to answer in for THIS
                # turn, matching what detect_lang() picked up. Native script
                # (not romanized) so Sarvam's TTS pronounces it correctly.
                lang_note = {"role": "system", "content": f"({reply_language_rule(user_lang)})"}
                self.history.append({"role": "user", "content": user_text})
                self._send_json({"type": "status", "state": "thinking"})

                buffer, full_reply = "", ""
                try:
                    messages = [self.system_prompt] + self.history + [lang_note]
                    for delta in self._stream_chat(client, messages):
                        if self.interrupt_flag.is_set():
                            log("LLM", "Interrupted mid-generation, stopping stream.")
                            break

                        buffer += delta
                        full_reply += delta
                        self._send_json({"type": "assistant_delta", "text": delta})

                        parts = SENTENCE_END_RE.split(buffer)
                        complete, i = "", 0
                        while i + 1 < len(parts):
                            complete += parts[i] + parts[i + 1]
                            i += 2
                        remainder = parts[i] if i < len(parts) else ""

                        sentence = complete.strip()
                        if sentence:
                            bm = BOOK_MEETING_RE.search(sentence)
                            if bm:
                                self._trigger_meeting_booking(bm.group(1), bm.group(2))
                                sentence = BOOK_MEETING_RE.sub("", sentence).strip()
                            if sentence and is_speakable(sentence):
                                self._enqueue_sentence(sentence, user_lang)
                        buffer = remainder
                except Exception as e:
                    log("LLM", f"ERROR: {e}")
                    self._send_json({"type": "error", "message": "Eva had trouble thinking that through."})
                    continue

                if not self.interrupt_flag.is_set():
                    tail = buffer.strip()
                    if tail:
                        bm = BOOK_MEETING_RE.search(tail)
                        if bm:
                            self._trigger_meeting_booking(bm.group(1), bm.group(2))
                            tail = BOOK_MEETING_RE.sub("", tail).strip()
                        if tail and is_speakable(tail):
                            self._enqueue_sentence(tail, user_lang)

                if full_reply.strip():
                    self.transcript.append({"role": "agent", "text": full_reply.strip(), "ts": time.time()})

                self.history.append({"role": "assistant", "content": full_reply})
                self._trim_history()
                self._send_json({"type": "assistant_done", "text": full_reply})

    # ---------- TTS loop ----------
# --- NEW (full method) ---
    def _tts_loop(self):
        pending_fetch = None  # (sentence, lang, generator) prefetched while previous sentence plays
        while not self.stop_event.is_set():
            if pending_fetch is not None:
                sentence, lang, gen = pending_fetch
                pending_fetch = None
            else:
                try:
                    sentence, lang = self.sentence_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                gen = None
            if not is_speakable(sentence):
                continue
            if self.interrupt_flag.is_set():
                continue

            self._send_json({"type": "status", "state": "speaking"})
            self.eva_speaking.set()

            if self.mode == "phone" and self.transport == "vanisetu" and self.vanisetu_session_id is not None:
                vanisetu_client.send_command(self.vanisetu_session_id, {"command": "START_MEDIA_BUFFERING"})

            # LiveKit's TTS always hands back linear16 PCM at its own
            # native sample rate — we resample + (for phone) mu-law/A-law
            # encode it ourselves with audioop, replacing what Sarvam used
            # to do internally via output_audio_codec/speech_sample_rate.
            if self.mode == "phone":
                if self.transport == "vanisetu":
                    target_rate, target_codec = VANISETU_RATE, "mulaw"
                elif self.transport == "voicelink":
                    target_rate, target_codec = VOICELINK_RATE, VOICELINK_CODEC
                else:
                    target_rate, target_codec = PHONE_RATE, "mulaw"
            else:
                target_rate, target_codec = TTS_SAMPLE_RATE, "linear16"

            if gen is None:
                gen = _stream_sarvam_tts(sentence, lang, self.voice_name, target_rate)

            resample_state = None
            leftover = b""
            prefetch_started = False
            try:
                 for pcm, src_rate in gen:
                    # Once we know THIS sentence is actually producing audio,
                    # start synthesizing the NEXT queued sentence in parallel
                    # so it's ready the instant this one finishes playing.
                    if not prefetch_started:
                        prefetch_started = True
                        try:
                            next_sentence, next_lang = self.sentence_q.get_nowait()
                            pending_fetch = (next_sentence, next_lang,
                                             _stream_sarvam_tts(next_sentence, next_lang, self.voice_name, target_rate))
                        except queue.Empty:
                            pass
                    if self.interrupt_flag.is_set():
                        break
                    if src_rate != target_rate:
                        pcm, resample_state = audioop.ratecv(pcm, 2, 1, src_rate, target_rate, resample_state)

                    if target_codec == "mulaw":
                        self._send_audio(audioop.lin2ulaw(pcm, 2))
                    elif target_codec == "alaw":
                        self._send_audio(audioop.lin2alaw(pcm, 2))
                    else:  # linear16 — browser widget
                        data = leftover + pcm
                        if len(data) % 2 != 0:
                            leftover = data[-1:]
                            data = data[:-1]
                        else:
                            leftover = b""
                        if data:
                            self._send_audio(data)
            except Exception as e:
                log("TTS", f"ERROR: {e}")

            self.eva_speaking.clear()

            if self.sentence_q.empty() and not self.interrupt_flag.is_set():
                self._send_json({"type": "status", "state": "listening"})
                self.turn_active = False



# ---------------- Web widget <-> PravaahAI bridge ----------------

def fetch_widget_config(public_id: str):
    """Asks PravaahAI which agent this public widget id belongs to, and
    whether the owner still has Eva minutes."""
    if not PRAVAAH_API_BASE_URL or not EVA_API_SECRET:
        return None, "Eva is not configured to talk to PravaahAI (PRAVAAH_API_BASE_URL/EVA_API_SECRET missing)"
    try:
        resp = requests.get(
            f"{PRAVAAH_API_BASE_URL}/api/public/widget-config/{public_id}",
            headers={"X-Eva-Secret": EVA_API_SECRET}, timeout=10,
        )
        data = resp.json()
        if resp.status_code >= 400:
            return None, data.get("error", "Widget not found")
        return data, None
    except Exception as e:
        return None, str(e)


def report_widget_lead(owner_id, widget_id, name, phone, email):
    """Fired the moment a visitor submits the lead form, so the lead exists
    in PravaahAI even if the call drops immediately after."""
    if not PRAVAAH_API_BASE_URL or not EVA_API_SECRET:
        return None
    try:
        resp = requests.post(
            f"{PRAVAAH_API_BASE_URL}/api/eva-webhook/widget-lead",
            headers={"X-Eva-Secret": EVA_API_SECRET, "Content-Type": "application/json"},
            json={"owner_id": owner_id, "widget_id": widget_id, "name": name, "phone": phone, "email": email},
            timeout=10,
        )
        return resp.json().get("lead_id")
    except Exception as e:
        log("WIDGET", f"lead report failed: {e}")
        return None


def report_widget_session_end(owner_id, widget_id, lead_id, duration_secs, transcript):
    """Fired when the widget WS closes — this is what deducts Eva minutes."""
    if not PRAVAAH_API_BASE_URL or not EVA_API_SECRET:
        return
    try:
        requests.post(
            f"{PRAVAAH_API_BASE_URL}/api/eva-webhook/widget-session-result",
            headers={"X-Eva-Secret": EVA_API_SECRET, "Content-Type": "application/json"},
            json={
                "owner_id": owner_id, "widget_id": widget_id, "lead_id": lead_id or "",
                "duration_secs": duration_secs, "transcript": transcript,
            },
            timeout=15,
        )
    except Exception as e:
        log("WIDGET", f"session-result report failed: {e}")


def book_meeting_via_pravaah(meeting_ctx: dict, lead: dict, call_id: str, requested_iso: str):
    """Calls PravaahAI's booking webhook when Eva emits a BOOK_MEETING tag
    mid-call. Returns (confirmed: bool, message_to_speak: str) — the message
    is spoken verbatim, so it's kept short and deterministic rather than
    trusting the LLM to phrase an API result correctly under time pressure."""
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
            alt_text = " or ".join(alts[:2])
            return False, f"That time isn't available. Would {alt_text} (UTC) work instead?"
        return False, data.get("error") or "That time isn't available — could you share another date and time?"
    except Exception as e:
        log("MEETING", f"booking webhook failed: {e}")
        return False, "Sorry, I had trouble booking that — could we try again?"



def fetch_caller_id_config(number: str):
    """Asks PravaahAI which owner+agent a VaniSetu number belongs to, for
    incoming calls. Mirrors fetch_widget_config()."""
    if not PRAVAAH_API_BASE_URL or not EVA_API_SECRET:
        return None, "Eva is not configured to talk to PravaahAI"
    try:
        resp = requests.get(
            f"{PRAVAAH_API_BASE_URL}/api/public/caller-id-config/{number}",
            headers={"X-Eva-Secret": EVA_API_SECRET}, timeout=10,
        )
        data = resp.json()
        if resp.status_code >= 400:
            return None, data.get("error", "Caller ID not found")
        return data, None
    except Exception as e:
        return None, str(e)


# ---------------- VoiceLink auth + call placement ----------------

_voicelink_token_lock = threading.Lock()
_voicelink_tokens = {}  # login_email -> {"token": "...", "expires_at": epoch}


def voicelink_login(login_username: str, login_password: str):
    """Logs into VoiceLink and returns a bearer token, cached per
    login_username so we don't re-auth on every single call. TODO: confirm the
    real token lifetime/field name from VoiceLink's Login response — the
    50-minute expiry below is a conservative guess, not a confirmed value."""
    with _voicelink_token_lock:
        cached = _voicelink_tokens.get(login_username)
        if cached and cached["expires_at"] > time.time():
            return cached["token"], None
    try:
        resp = requests.post(
            f"{VOICELINK_BASE_URL}/api/v1/auth/login",
            json={"username": login_username, "password": login_password},
            timeout=15,
        )
    except Exception as e:
        log("VOICELINK-AUTH", f"login request failed to send: {e}")
        return None, f"Could not reach VoiceLink: {e}"

    # Log the raw response BEFORE trying to parse it as JSON, so a bad
    # endpoint path / auth shape shows up clearly in the logs instead of
    # surfacing only as a cryptic "Expecting value" JSON decode error.
    log("VOICELINK-AUTH", f"login response: status={resp.status_code} body={resp.text[:500]!r}")

    try:
        data = resp.json()
    except ValueError:
        return None, (
            f"VoiceLink login did not return JSON (status {resp.status_code}). "
            f"Check the login endpoint path/payload — got: {resp.text[:200]!r}"
        )

    if resp.status_code >= 400:
        return None, data.get("message") or data.get("error") or f"VoiceLink login failed (status {resp.status_code})"
    inner = data.get("data") or {}
    token = (
        data.get("token")
        or data.get("access_token")
        or inner.get("token")
        or inner.get("access_token")
    )
    if not token:
        return None, f"VoiceLink login succeeded but no token found in the response: {data}"
    with _voicelink_token_lock:
        _voicelink_tokens[login_username] = {"token": token, "expires_at": time.time() + 50 * 60}
    return token, None


def voicelink_add_lead(login_username, login_password, did_number, customer_number,
                        websocket_url, webhook_url, custom_parameters=None):
    """Queues one outbound call via VoiceLink's add_lead API. Retries once
    on a 401/403 in case the cached token had just expired."""
    def _do_call(token):
        return requests.post(
            f"{VOICELINK_BASE_URL}/api/v1/add_lead",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "did_number": did_number,
                "customer_number": customer_number,
                "websocket_url": websocket_url,
                "webhook_url": webhook_url,
                "custom_parameters": json.dumps(custom_parameters or {}),
            },
            timeout=15,
        )

    token, err = voicelink_login(login_username, login_password)
    if err:
        return None, err

    def _parse(resp):
        log("VOICELINK-AUTH", f"add_lead response: status={resp.status_code} body={resp.text[:500]!r}")
        try:
            return resp.json(), None
        except ValueError:
            return None, f"VoiceLink add_lead did not return JSON (status {resp.status_code}): {resp.text[:200]!r}"

    try:
        resp = _do_call(token)
        if resp.status_code in (401, 403):
            with _voicelink_token_lock:
                _voicelink_tokens.pop(login_username, None)
            token2, err2 = voicelink_login(login_username, login_password)
            if err2:
                return None, err2
            resp = _do_call(token2)

        data, parse_err = _parse(resp)
        if parse_err:
            return None, parse_err
        if resp.status_code >= 400:
            return None, data.get("message") or data.get("error") or f"VoiceLink add_lead failed (status {resp.status_code})"
        return data, None
    except Exception as e:
        return None, str(e)


# VaniSetu's doc (v1.2, 22 Aug 2026) only documents INCOMING_CALL and
# MEDIA_START as events sent to us - it does NOT document what event
# fires when a call ends (caller hangs up, etc). Without handling that,
# a VaniSetu EvaSession never gets close()'d or POSTed back to Pravaah's
# callback_url, so the call sits stuck as "queued" on Pravaah's side even
# though the call itself ended fine on VaniSetu's end.
# These are best-guess event names based on common telephony-provider
# conventions. CONFIRM THE REAL NAME WITH VARNET (or check the
# "unhandled payload" log line after one real test hangup) and tighten
# this set once confirmed.
VANISETU_CALL_END_EVENTS = {
    "HANGUP", "CALL_ENDED", "CALL_END", "CALL_COMPLETED",
    "CHANNEL_HANGUP", "MEDIA_STOP", "DISCONNECTED", "CALL_DISCONNECTED",
}


# ============================================================
# VaniSetu — single multiplexed WS connection to the number provider
# ============================================================
class VaniSetuClient:
    """One persistent WebSocket to VaniSetu carries every call on this
    account, multiplexed by a numeric session_id VaniSetu assigns per call.
    Owns that connection, authenticates, and routes audio/events to/from
    the right EvaSession."""

    def __init__(self):
        self._wsapp = None
        self._send_lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = False

        self.sessions = {}                    # vanisetu session_id (int) -> EvaSession
        self._pending_incoming_call_ids = []   # FIFO of call_ids from INCOMING_CALL, not yet bound
        self._pending_incoming_lock = threading.Lock()
        self._pending_outbound = {}            # request_id -> {"to_number", "caller_id"}
        self._pending_outbound_lock = threading.Lock()

    # ---------- lifecycle ----------
    def start(self):
        if not (VANISETU_TCODE and VANISETU_TOKEN):
            log("VANISETU", "VANISETU_TCODE/VANISETU_TOKEN not set — VaniSetu disabled.")
            return
        threading.Thread(target=self._run_forever, daemon=True, name="VaniSetuClient").start()

    def _run_forever(self):
        while not self._stop:
            try:
                self._ready.clear()
                self._connect_once()
            except Exception as e:
                log("VANISETU", f"connection error: {e}")
            self._ready.clear()
            time.sleep(3)   # reconnect backoff

    def _connect_once(self):
        def on_open(wsapp):
            log("VANISETU", "socket open, sending auth...")
            wsapp.send(json.dumps({"type": "auth", "tcode": VANISETU_TCODE, "token": VANISETU_TOKEN}))

        def on_message(wsapp, message):
            try:
                self._handle_message(message)
            except Exception as e:
                log("VANISETU", f"message handling error: {e}")

        def on_error(wsapp, error):
            log("VANISETU", f"socket error: {error}")

        def on_close(wsapp, code, msg):
            log("VANISETU", f"socket closed: {code} {msg}")
            self._ready.clear()

        self._wsapp = vanisetu_ws_lib.WebSocketApp(
            VANISETU_WS_URL, on_open=on_open, on_message=on_message,
            on_error=on_error, on_close=on_close,
        )
        self._wsapp.run_forever(ping_interval=25, ping_timeout=10)

    # ---------- wire out ----------
    def _send_text(self, obj):
        if not self._wsapp:
            return False
        with self._send_lock:
            try:
                self._wsapp.send(json.dumps(obj))
                return True
            except Exception as e:
                log("VANISETU", f"send error: {e}")
                return False

    def send_command(self, session_id: int, payload: dict):
        return self._send_text({"session_id": session_id, "payload": payload})

    def send_audio(self, session_id: int, audio_bytes: bytes):
        if not self._wsapp:
            return False
        frame = session_id.to_bytes(4, "big") + audio_bytes
        with self._send_lock:
            try:
                self._wsapp.send(frame, opcode=vanisetu_ws_lib.ABNF.OPCODE_BINARY)
                return True
            except Exception as e:
                log("VANISETU", f"audio send error: {e}")
                return False

    def place_outbound_call(self, request_id: str, endpoint: str, caller_id: str):
        with self._pending_outbound_lock:
            self._pending_outbound[request_id] = {"to_number": endpoint, "caller_id": caller_id}
        return self._send_text({"payload": {
            "command": "OUTBOUND_CALL", "endpoint": endpoint,
            "caller_id": caller_id, "request_id": request_id,
        }})

    def unregister_session(self, session_id: int):
        self.sessions.pop(session_id, None)

    # ---------- wire in ----------
    def _handle_message(self, message):
        if isinstance(message, (bytes, bytearray)):
            self._handle_binary(message)
            return

        obj = json.loads(message)

        if obj.get("event") == "INCOMING_CALL":
            self._on_incoming_call(obj)
            return
        if obj.get("type") in ("auth_ok", "auth_success") or obj.get("status") == "ok":
            log("VANISETU", "authenticated")
            self._ready.set()
            return
        if obj.get("type") == "error":
            log("VANISETU", f"error from VaniSetu: {obj}")
            return

        if "session_id" in obj:
            session_id = obj["session_id"]
            payload = obj.get("payload", {}) or {}
            event = payload.get("event")
            if event == "MEDIA_START":
                self._on_media_start(session_id, payload)
            elif event == "INCOMING_CALL":
                # Genuine inbound calls arrive UNWRAPPED (no session_id) per
                # VaniSetu's doc - see _on_incoming_call below. If we see
                # INCOMING_CALL wrapped with a session_id already attached,
                # it's actually VaniSetu telling us one of OUR OUTBOUND_CALL
                # requests just got answered, not a new inbound call. This
                # was previously falling through to "unhandled payload" and
                # getting silently dropped - the follow-up MEDIA_START then
                # had nothing to match against and got hung up as a bogus
                # inbound call. See _on_outbound_connected.
                self._on_outbound_connected(session_id, payload)
            elif event in VANISETU_CALL_END_EVENTS or (event and event.upper() in VANISETU_CALL_END_EVENTS):
                self._on_call_ended(session_id, payload)
            else:
                # If you just hung up a test call and landed here, THIS is
                # the real event name/shape VaniSetu uses for call-end -
                # copy the exact "event" value from the log line below into
                # VANISETU_CALL_END_EVENTS and it'll route correctly next time.
                log("VANISETU", f"session {session_id}: unhandled payload {payload}")

    def _handle_binary(self, data: bytes):
        if len(data) < 4:
            return
        session_id = int.from_bytes(data[:4], "big")
        session = self.sessions.get(session_id)
        if session:
            session.feed_audio(data[4:])

    def _on_incoming_call(self, obj):
        call_id = obj.get("call_id")
        log("VANISETU", f"INCOMING_CALL call_id={call_id}")
        with self._pending_incoming_lock:
            self._pending_incoming_call_ids.append(call_id)


    def _on_media_start(self, session_id, payload):
        # If this session is already bound (e.g. it was just connected via
        # the wrapped INCOMING_CALL "call answered" event in
        # _on_outbound_connected), this MEDIA_START is just confirming the
        # media path is up - NOT a new call. Re-running the binding logic
        # here would wrongly treat it as inbound and hang up a live,
        # already-answered outbound call.
        if session_id in self.sessions:
            log("VANISETU", f"session {session_id}: MEDIA_START on already-bound session, ignoring")
            return

        # Check whether this session is one of our own outbound requests.
        # VaniSetu's doc doesn't show the exact field name request_id comes
        # back under on the connect event, so we check a couple of likely
        # names defensively — confirm with Varnet and tighten this if needed.
        request_id = payload.get("request_id") or payload.get("requestId")
        outbound_cfg = None
        if request_id:
            with self._pending_outbound_lock:
                outbound_cfg = self._pending_outbound.pop(request_id, None)
        if outbound_cfg:
            self._bind_outbound_session(session_id, request_id)
            return

        # Otherwise treat it as an inbound call, matched FIFO to the oldest
        # still-unbound INCOMING_CALL (sessions come up in call order).
        with self._pending_incoming_lock:
            call_id = self._pending_incoming_call_ids.pop(0) if self._pending_incoming_call_ids else None
        self._bind_incoming_session(session_id, call_id)


    def _on_call_ended(self, session_id, payload):
        """Fires when VaniSetu tells us a call is over (see
        VANISETU_CALL_END_EVENTS above - exact event name unconfirmed with
        Varnet, verify against real traffic). Without this, VaniSetu
        sessions never got close()'d or reported back to Pravaah, which is
        why completed calls were stuck showing "queued" on Pravaah's side -
        the transcript-completion POST never fired."""
        session = self.sessions.get(session_id)
        if not session:
            log("VANISETU", f"session {session_id}: call-ended event with no matching session (already closed?)")
            return
        log("VANISETU", f"session {session_id}: call ended ({payload.get('event')}), closing + reporting to Pravaah")
        session.hangup_reason = "completed"
        session.close()
        session._finish_and_callback(hangup_reason="completed")
        self.unregister_session(session_id)

    def _on_outbound_connected(self, session_id, payload):
        """Binds a wrapped INCOMING_CALL event to the oldest still-unbound
        outbound request (FIFO). VaniSetu doesn't echo our request_id back
        on this event, so exact matching isn't possible - FIFO is correct
        as long as answers come back in roughly the order calls were
        placed. Worth confirming with Varnet if you ever run many
        concurrent outbound calls and see mis-binding."""
        request_id = None
        with self._pending_outbound_lock:
            request_id = next(iter(self._pending_outbound), None)
            if request_id:
                self._pending_outbound.pop(request_id, None)

        if request_id:
            log("VANISETU", f"session {session_id}: outbound call answered, binding to request_id={request_id}")
            self._bind_outbound_session(session_id, request_id)
        else:
            # No outbound call was waiting - treat as a genuine inbound call.
            log("VANISETU", f"session {session_id}: wrapped INCOMING_CALL with no pending outbound request, treating as real inbound")
            with self._pending_incoming_lock:
                call_id = self._pending_incoming_call_ids.pop(0) if self._pending_incoming_call_ids else None
            self._bind_incoming_session(session_id, call_id)

    # ---------- binding sessions ----------
    def _bind_outbound_session(self, session_id, request_id):
        call_id = request_id  # we use call_id as request_id when placing the call
        with _pending_calls_lock:
            cfg = PENDING_CALLS.get(call_id)
        if not cfg:
            log("VANISETU", f"no PENDING_CALLS entry for outbound call_id={call_id}, hanging up")
            self.send_command(session_id, {"command": "HANGUP"})
            return

        session = EvaSession(
            ws=None, mode="phone", transport="vanisetu", vanisetu_session_id=session_id,
            call_id=call_id, agent=cfg["agent"], lead=cfg["lead"],
            callback_url=cfg["callback_url"], meeting=cfg.get("meeting"),
        )
        self.sessions[session_id] = session
        if not session.start():
            self.send_command(session_id, {"command": "HANGUP"})
            with _pending_calls_lock:
                PENDING_CALLS.pop(call_id, None)
            return

        session.call_started_at = time.time()
        self.send_command(session_id, {"command": "ANSWER"})
        opening = render_call_vars(
            cfg["agent"].get("opening_line") or "Hi {{name}}, do you have a quick minute?", cfg["lead"],
        )
        opening_lang = cfg["agent"].get("language") if cfg["agent"].get("language") in SUPPORTED_LANGUAGES else "en"
        session.speak(opening, opening_lang)
        log("VANISETU", f"Outbound call {call_id} connected as session {session_id}")

    def _bind_incoming_session(self, session_id, call_id):
        # TODO: VaniSetu doesn't document a "dialed number"/DID field on
        # INCOMING_CALL or MEDIA_START — that's what we need to look up the
        # right owner+agent. Confirm the real field with Varnet and set
        # `number` from it. Until then, incoming VaniSetu calls are rejected.
        number = None
        config, err = (fetch_caller_id_config(number) if number else (None, "No DID field available from VaniSetu yet"))
        if not config:
            log("VANISETU", f"incoming call {call_id} (session {session_id}) rejected: {err}")
            self.send_command(session_id, {"command": "HANGUP"})
            return

        agent = config.get("agent", {})
        session = EvaSession(
            ws=None, mode="phone", transport="vanisetu", vanisetu_session_id=session_id,
            agent=agent, lead={},
        )
        self.sessions[session_id] = session
        if not session.start():
            self.send_command(session_id, {"command": "HANGUP"})
            return
        session.call_started_at = time.time()
        self.send_command(session_id, {"command": "ANSWER"})
        session.speak(agent.get("opening_line") or "Hi, how can I help you today?", "en")
        log("VANISETU", f"Incoming call {call_id} connected as session {session_id}")


# --- NEW ---
VANISETU_ENABLED = os.environ.get("VANISETU_ENABLED", "false").lower() == "true"

vanisetu_client = VaniSetuClient()
if VANISETU_ENABLED:
    vanisetu_client.start()
else:
    log("VANISETU", "VANISETU_ENABLED=false — skipping VaniSetu connection.")


# ============================================================
# Routes
# ============================================================
@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/widget.js")
def widget_js():
    return send_from_directory("static", "widget.js", mimetype="application/javascript")

@app.route("/api/widget-meta/<public_id>")
def widget_meta(public_id):
    """Public, browser-callable (no X-Eva-Secret) — returns only the safe
    subset of a widget's config so the embed script can schedule the
    auto-greet timer *before* opening a voice session. Never exposes
    owner_id or the agent's system_prompt."""
    config, err = fetch_widget_config(public_id)
    if err or not config:
        resp = jsonify({"error": err or "Widget unavailable"})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp, 404

    agent = config.get("agent", {}) or {}
    auto_greet = config.get("auto_greet") or {}
    resp = jsonify({
        "greeting": config.get("greeting") or agent.get("opening_line") or "Hi! How can I help you today?",
        "icon_url": config.get("icon_url", ""),
        "require_lead_before_chat": config.get("require_lead_before_chat", True),
        "auto_greet": {
            "enabled": bool(auto_greet.get("enabled", False)),
            "message": auto_greet.get("message", ""),
            "delay_secs": int(auto_greet.get("delay_secs", 5) or 5),
        },
    })
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

@app.route("/embed/widget.js")
def embed_widget_js():
    """Self-contained embed script. Users paste ONE tag on their site:
      <script src="{EVA_PUBLIC_BASE_URL}/embed/widget.js"
              data-public-id="wgt_xxx" data-color="#2454E8" async></script>
    Renders a floating mic bubble bottom-right on desktop, and a full-width
    centered bottom bubble on mobile (see @media block in the CSS below)."""
    js = r"""
 (function(){
   var cur = document.currentScript;
   var publicId = cur.getAttribute('data-public-id');
   var color = cur.getAttribute('data-color') || '#2454E8';
   if(!publicId){ console.error('[EvaWidget] missing data-public-id'); return; }
   var evaOrigin = cur.src.split('/embed/widget.js')[0];
   var wsUrl = evaOrigin.replace(/^http/, 'ws') + '/ws/widget/' + publicId;
   var metaUrl = evaOrigin + '/api/widget-meta/' + publicId;

  var css = document.createElement('style');
  css.textContent = `
    #eva-w-bubble{position:fixed;bottom:22px;right:22px;width:62px;height:62px;border-radius:50%;
      background:${color};box-shadow:0 6px 20px rgba(0,0,0,.25);display:flex;align-items:center;
      justify-content:center;cursor:pointer;z-index:999999;transition:transform .2s;background-size:cover;
      background-position:center;background-repeat:no-repeat;}
    #eva-w-bubble:hover{transform:scale(1.06);}
    #eva-w-bubble svg{width:26px;height:26px;fill:#fff;}
    #eva-w-bubble.eva-live{animation:eva-pulse 1.4s infinite;}
    #eva-w-bubble.eva-has-icon svg{display:none;}
    @keyframes eva-pulse{0%{box-shadow:0 0 0 0 ${color}66;}70%{box-shadow:0 0 0 16px ${color}00;}100%{box-shadow:0 0 0 0 ${color}00;}}
    #eva-w-panel{position:fixed;bottom:96px;right:22px;width:340px;max-width:92vw;height:480px;max-height:76vh;
      background:#fff;border-radius:16px;box-shadow:0 12px 40px rgba(0,0,0,.22);display:none;flex-direction:column;
      overflow:hidden;z-index:999999;font-family:-apple-system,Segoe UI,Roboto,sans-serif;}
    #eva-w-panel.open{display:flex;}
    #eva-w-head{background:${color};color:#fff;padding:12px 14px;font-size:14px;font-weight:600;
      display:flex;align-items:center;justify-content:space-between;}
    #eva-w-close{cursor:pointer;font-size:12.5px;font-weight:600;padding:6px 10px;border-radius:16px;
      background:rgba(255,255,255,.16);display:flex;align-items:center;gap:5px;user-select:none;}
    #eva-w-close:hover{background:rgba(255,255,255,.28);}
    #eva-w-body{flex:1;padding:16px;overflow:auto;font-size:13px;color:#222;display:flex;flex-direction:column;}
    #eva-w-greeting{background:#f2f4f8;color:#222;padding:10px 12px;border-radius:12px;margin-bottom:14px;font-size:13px;}
    #eva-w-form input{width:100%;box-sizing:border-box;margin-bottom:8px;padding:10px 12px;border:1px solid #ddd;
      border-radius:8px;font-size:13px;}
    #eva-w-form button{width:100%;padding:10px;border:none;border-radius:8px;background:${color};color:#fff;
      font-weight:600;cursor:pointer;}
    #eva-w-status{text-align:center;color:#888;font-size:12px;margin-top:10px;}
    #eva-w-mic{width:74px;height:74px;border-radius:50%;background:${color};margin:16px auto;display:flex;
      align-items:center;justify-content:center;animation:eva-mic-pulse 1.8s infinite;}
    @keyframes eva-mic-pulse{0%{box-shadow:0 0 0 0 ${color}55;}70%{box-shadow:0 0 0 14px ${color}00;}100%{box-shadow:0 0 0 0 ${color}00;}}
    #eva-w-mic svg{width:30px;height:30px;fill:#fff;}
    #eva-w-transcript{font-size:12.5px;line-height:1.6;flex:1;overflow:auto;margin-top:8px;}
    #eva-w-transcript .u{color:#111;font-weight:600;}
    #eva-w-transcript .a{color:${color};font-weight:600;}
    #eva-w-chatbar{display:none;gap:6px;margin-top:10px;}
    #eva-w-chatbar input{flex:1;padding:9px 11px;border:1px solid #ddd;border-radius:20px;font-size:13px;}
    #eva-w-chatbar button{padding:9px 14px;border:none;border-radius:20px;background:${color};color:#fff;
      font-weight:600;cursor:pointer;}
    #eva-w-open-label{position:fixed;bottom:92px;right:22px;background:#111;color:#fff;padding:7px 14px;
      border-radius:20px;font-size:12px;font-family:-apple-system,Segoe UI,Roboto,sans-serif;cursor:pointer;
      box-shadow:0 4px 14px rgba(0,0,0,.25);z-index:999999;display:none;white-space:nowrap;}
    @media(max-width:520px){
      #eva-w-bubble{left:50%;right:auto;bottom:18px;transform:translateX(-50%);}
      #eva-w-bubble:hover{transform:translateX(-50%) scale(1.06);}
      #eva-w-panel{left:0;right:0;bottom:0;transform:none;width:100%;height:100%;
        max-height:100%;border-radius:0;max-width:100%;}
      #eva-w-open-label{left:50%;right:auto;transform:translateX(-50%);bottom:88px;}
    }
  `;
  document.head.appendChild(css);

  var bubble = document.createElement('div'); bubble.id = 'eva-w-bubble';
  bubble.innerHTML = '<svg viewBox="0 0 24 24"><path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2z"/></svg>';

  var panel = document.createElement('div'); panel.id = 'eva-w-panel';
  panel.innerHTML =
    '<div id="eva-w-head">' +
      '<span>Talk to us</span>' +
      '<span id="eva-w-close">Close &times;</span>' +
    '</div>' +
    '<div id="eva-w-body">' +
      '<div id="eva-w-greeting" style="display:none;"></div>' +
      '<div id="eva-w-form" style="display:none;">' +
        '<input id="eva-w-name" placeholder="Your name">' +
        '<input id="eva-w-phone" placeholder="Phone number">' +
        '<input id="eva-w-email" placeholder="Email (optional)">' +
        '<button id="eva-w-lead-submit">Continue</button>' +
      '</div>' +
      '<div id="eva-w-call" style="display:none;text-align:center;">' +
        '<div id="eva-w-mic"><svg viewBox="0 0 24 24"><path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2z"/></svg></div>' +
        '<div id="eva-w-status">Connecting…</div>' +
      '</div>' +
      '<div id="eva-w-transcript"></div>' +
      '<div id="eva-w-chatbar">' +
        '<input id="eva-w-chat-input" placeholder="Type a message…">' +
        '<button id="eva-w-chat-send">Send</button>' +
      '</div>' +
    '</div>';
  document.body.appendChild(bubble);
  document.body.appendChild(panel);

  var openLabel = document.createElement('div');
  openLabel.id = 'eva-w-open-label';
  openLabel.textContent = 'Click to open';
  document.body.appendChild(openLabel);

  var ws, audioCtx, mic, processor, playHead = 0;
  var autoGreetTimer = null, sessionStarted = false, voiceStarted = false;

  function show(id, disp){ document.getElementById(id).style.display = disp; }

  function cancelAutoGreet(){
    if(autoGreetTimer){ clearTimeout(autoGreetTimer); autoGreetTimer = null; }
  }

  function openWidgetPanel(){
    panel.classList.add('open');
    openLabel.style.display = 'none';
    cancelAutoGreet();
    if(!sessionStarted) connectSession();
  }
  function closeWidgetPanel(){
    panel.classList.remove('open');
    openLabel.style.display = 'block';
  }
  bubble.addEventListener('click', function(){
    if(panel.classList.contains('open')) closeWidgetPanel();
    else openWidgetPanel();
  });
  openLabel.addEventListener('click', openWidgetPanel);
  document.getElementById('eva-w-close').addEventListener('click', function(e){
    e.stopPropagation();
    closeWidgetPanel();
  });

  // ---- 1) connect the socket right away — this ONLY delivers the text
  //         greeting + status. Eva never speaks and no mic opens yet.
  function connectSession(){
    if(sessionStarted) return;
    sessionStarted = true;
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    ws.onopen = function(){ document.getElementById('eva-w-status').textContent = 'Connecting…'; };
    wireSocketEvents();
  }

  function wireSocketEvents(){
    ws.onmessage = function(ev){
      if(typeof ev.data === 'string'){
        var msg = JSON.parse(ev.data);

        if(msg.type === 'greeting'){
          // TEXT ONLY — this is never spoken/played as audio.
          var g = document.getElementById('eva-w-greeting');
          g.textContent = msg.text; g.style.display = 'block';
        }
        if(msg.type === 'status' && msg.state === 'awaiting_lead_info'){
          show('eva-w-form', 'block');
        }
        if(msg.type === 'status' && msg.state === 'auto_start_voice'){
          show('eva-w-form', 'none');
          autoStartVoice();
        }
        if(msg.type === 'status' && msg.state){
          document.getElementById('eva-w-status').textContent =
            msg.state === 'speaking' ? 'Eva is speaking…' : (msg.state === 'thinking' ? 'Thinking…' : 'Listening…');
        }
        if(msg.type === 'assistant_delta'){ appendTranscript('a', msg.text, true); }
        if(msg.type === 'assistant_done'){ appendTranscript('a', '', false); }
        if(msg.type === 'user_transcript' && msg.final){ appendTranscript('u', msg.text, false); }
        if(msg.type === 'error'){ document.getElementById('eva-w-status').textContent = msg.message; }
      } else {
        playAudio(ev.data);
      }
    };
    ws.onclose = function(){ bubble.classList.remove('eva-live'); sessionStarted = false; stopMic(); };
  }

  // ---- 2) lead form ----
  document.getElementById('eva-w-lead-submit').addEventListener('click', function(){
    var name = document.getElementById('eva-w-name').value.trim();
    var phone = document.getElementById('eva-w-phone').value.trim();
    var email = document.getElementById('eva-w-email').value.trim();
    if(!name || !phone){ alert('Please share your name and phone number'); return; }
    if(ws && ws.readyState === WebSocket.OPEN){
      ws.send(JSON.stringify({type:'lead_info', name:name, phone:phone, email:email}));
    }
  });

  // ---- 3) automatically starts voice the moment lead info is captured
  //         (or right after the greeting, if no lead form is required) ----
  function autoStartVoice(){
    if(voiceStarted) return;
    voiceStarted = true;
    show('eva-w-call', 'block');
    show('eva-w-chatbar', 'flex');
    if(ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({type:'start_voice'}));
    startMic();
    bubble.classList.add('eva-live');
  }

  document.getElementById('eva-w-chat-send').addEventListener('click', sendChatText);
  document.getElementById('eva-w-chat-input').addEventListener('keydown', function(e){
    if(e.key === 'Enter') sendChatText();
  });
  function sendChatText(){
    var input = document.getElementById('eva-w-chat-input');
    var text = input.value.trim();
    if(!text || !ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({type:'text', text:text}));
    input.value = '';
  }

  // ---- proactive auto-greet: owner opt-in, speaks right away on a timer ----
  function startAutoGreetSession(){
    if(sessionStarted) return;
    sessionStarted = true;
    voiceStarted = true;
    show('eva-w-call', 'block');
    show('eva-w-chatbar', 'flex');
    document.getElementById('eva-w-status').textContent = 'Connecting…';
    openLabel.style.display = 'none';
    panel.classList.add('open');
    ws = new WebSocket(wsUrl + '?auto_greet=1');
    ws.binaryType = 'arraybuffer';
    ws.onopen = function(){ startMic(); bubble.classList.add('eva-live'); };
    wireSocketEvents();
  }

  var lastRole = null, lastLine = null;
  function appendTranscript(role, text, streaming){
    var box = document.getElementById('eva-w-transcript');
    if(streaming && lastRole === role && lastLine){ lastLine.lastChild.textContent += text; }
    else{
      lastLine = document.createElement('div');
      lastLine.innerHTML = '<span class="'+role+'">'+(role==='u'?'You: ':'Eva: ')+'</span>';
      lastLine.appendChild(document.createTextNode(text));
      box.appendChild(lastLine); lastRole = role;
    }
    box.scrollTop = box.scrollHeight;
  }

  function startMic(){
    navigator.mediaDevices.getUserMedia({audio:{channelCount:1,sampleRate:16000}}).then(function(stream){
      audioCtx = new (window.AudioContext||window.webkitAudioContext)({sampleRate:16000});
      mic = audioCtx.createMediaStreamSource(stream);
      processor = audioCtx.createScriptProcessor(4096,1,1);
      mic.connect(processor); processor.connect(audioCtx.destination);
      processor.onaudioprocess = function(e){
        if(!ws || ws.readyState !== 1) return;
        var input = e.inputBuffer.getChannelData(0);
        var pcm = new Int16Array(input.length);
        for(var i=0;i<input.length;i++){ var s = Math.max(-1,Math.min(1,input[i])); pcm[i] = s<0?s*0x8000:s*0x7FFF; }
        ws.send(pcm.buffer);
      };
    }).catch(function(){ document.getElementById('eva-w-status').textContent = 'Microphone access denied'; });
  }
  function stopMic(){
    if(processor){ processor.disconnect(); }
    if(mic){ mic.disconnect(); }
    if(audioCtx){ audioCtx.close(); }
    voiceStarted = false;
  }

  function playAudio(buf){
    if(!audioCtx) return;
    var pcm = new Int16Array(buf);
    var float32 = new Float32Array(pcm.length);
    for(var i=0;i<pcm.length;i++) float32[i] = pcm[i]/0x8000;
    var abuf = audioCtx.createBuffer(1, float32.length, 22050);
    abuf.copyToChannel(float32, 0);
    var src = audioCtx.createBufferSource();
    src.buffer = abuf; src.connect(audioCtx.destination);
    var startAt = Math.max(audioCtx.currentTime, playHead);
    src.start(startAt); playHead = startAt + abuf.duration;
  }

  fetch(metaUrl).then(function(r){ return r.json(); }).then(function(meta){
    if(!meta) return;
    if(meta.icon_url){
      var iconImg = new Image();
      iconImg.onload = function(){
        bubble.style.backgroundImage = 'url("' + meta.icon_url + '")';
        bubble.classList.add('eva-has-icon');
      };
      iconImg.onerror = function(){ console.warn('[EvaWidget] icon failed to load:', meta.icon_url); };
      iconImg.src = meta.icon_url;
    }
    if(meta.auto_greet && meta.auto_greet.enabled && meta.auto_greet.message){
      var delayMs = Math.max(1, meta.auto_greet.delay_secs || 5) * 1000;
      autoGreetTimer = setTimeout(function(){
        autoGreetTimer = null;
        if(!sessionStarted) startAutoGreetSession();
      }, delayMs);
    }
  }).catch(function(err){ console.warn('[EvaWidget] widget-meta fetch failed:', err); });
})();
"""
    return js, 200, {"Content-Type": "application/javascript"}


@sock.route("/ws/widget/<public_id>")
def widget_ws(ws, public_id):
    """A visitor on some customer's website connects here. We look up which
    agent + owner this public_id belongs to, run the normal Eva voice
    pipeline, capture the lead, and bill Eva minutes on close.

    Fixed flow (previously Eva spoke + the mic effectively started the
    instant the socket opened, with no text greeting and no lead-gate when
    require_lead_before_chat was off):
      1. Server sends a plain TEXT "greeting" message immediately — nothing
         is spoken, no mic is expected yet.
      2. If require_lead_before_chat is on, server sends
         status=awaiting_lead_info and waits for a "lead_info" message.
         If it's off, server sends status=ready_for_mode_select right away.
      3. Client then explicitly chooses a mode:
           - {"type":"start_chat"}  -> pure text, no mic, no auto-speak
           - {"type":"start_voice"} -> Eva speaks the opening line, client
             is expected to start streaming mic audio from here
      4. Meeting/site-visit booking now works from the widget: the full
         `meeting` context (webhook url, owner_id, agent_id, lead_id) is
         passed into EvaSession so BOOK_MEETING: triggers actually fire,
         exactly like a phone call.

    ?auto_greet=1 is unchanged — that's the owner's own opt-in "Eva speaks
    first after N seconds" feature and intentionally skips all of this."""
    config, err = fetch_widget_config(public_id)
    if err or not config:
        try:
            ws.send(json.dumps({"type": "error", "message": err or "Widget unavailable"}))
        except Exception:
            pass
        return
   #new
    owner_id = config["owner_id"]
    widget_id = config["widget_id"]
    agent = config.get("agent", {})
    require_lead_first = config.get("require_lead_before_chat", True)
    auto_greet_cfg = config.get("auto_greet") or {}
    is_auto_greet = request.args.get("auto_greet") == "1" and auto_greet_cfg.get("enabled")
    meeting_ctx = config.get("meeting")  # None unless agent's book_meeting task is on

    session = EvaSession(ws, mode="browser", agent=agent, lead={}, meeting=meeting_ctx)
    session.call_started_at = time.time()  # reused purely for widget duration billing
    if not session.start():
        return

    greet_text = config.get("greeting") or agent.get("opening_line") or "Hi! How can I help you today?"

    if is_auto_greet:
        session._send_json({"type": "ready"})
        auto_text = auto_greet_cfg.get("message") or greet_text
        session.speak(auto_text, detect_lang(auto_text))
    else:
        # Always show a plain text greeting first — this is NEVER spoken/played as audio.
        session._send_json({"type": "greeting", "text": greet_text})
        if require_lead_first:
            session._send_json({"type": "status", "state": "awaiting_lead_info"})
        else:
            # No lead form required — go straight into a live voice session.
            session._send_json({"type": "status", "state": "auto_start_voice"})

    lead_id_holder = {"lead_id": None}
    log("MAIN", f"Widget visitor connected: {public_id}" + (" (auto-greet)" if is_auto_greet else ""))

    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            if isinstance(msg, (bytes, bytearray)):
                session.feed_audio(bytes(msg))
                continue
            try:
                payload = json.loads(msg)
            except Exception:
                continue

            mtype = payload.get("type")
            if mtype == "lead_info":
                name = (payload.get("name") or "").strip()
                phone = (payload.get("phone") or "").strip()
                email = (payload.get("email") or "").strip()
                session.lead = {"name": name, "phone": phone, "email": email}
                lead_id_holder["lead_id"] = report_widget_lead(owner_id, widget_id, name, phone, email)
                if session.meeting and lead_id_holder["lead_id"]:
                    session.meeting["lead_id"] = lead_id_holder["lead_id"]
                if not is_auto_greet:
                    # Lead captured — go straight into a live voice session, no extra click needed.
                    session._send_json({"type": "status", "state": "auto_start_voice"})

            elif mtype == "start_chat":
                # Pure text mode — no mic, no auto-speak. Eva only replies
                # once the visitor actually types something (mtype=="text").
                session._send_json({"type": "ready"})

            elif mtype == "start_voice":
                # Visitor chose to talk — NOW Eva speaks the opening line,
                # and the client is expected to start streaming mic audio.
                if not is_auto_greet:
                    opening = render_call_vars(
                        agent.get("opening_line") or "Hi {{name}}, how can I help you today?", session.lead,
                    )
                    session._send_json({"type": "ready"})
                    session.speak(opening, "en")

            elif mtype == "text":
                session.feed_text(payload.get("text", ""))
            elif mtype == "ping":
                session._send_json({"type": "pong"})
    except Exception as e:
        log("MAIN", f"widget ws loop error: {e}")
    finally:
        session.close()
        duration_secs = round(time.time() - session.call_started_at, 1)
        report_widget_session_end(owner_id, widget_id, lead_id_holder["lead_id"], duration_secs, session.transcript)
        log("MAIN", f"Widget visitor disconnected: {public_id} ({duration_secs}s)")

@app.route("/health")
def health():
    missing = check_missing_keys()
    return jsonify({"status": "ok" if not missing else "missing_keys", "missing": missing})


@sock.route("/ws/eva")
def eva_ws(ws):
    missing = check_missing_keys()
    if missing:
        ws.send(json.dumps({"type": "error", "message": f"Server missing env keys: {', '.join(missing)}"}))
        return

    session = EvaSession(ws)
    if not session.start():
        return

    session._send_json({"type": "ready"})
    log("MAIN", "Browser client connected.")

    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            if isinstance(msg, (bytes, bytearray)):
                session.feed_audio(bytes(msg))
            else:
                try:
                    payload = json.loads(msg)
                except Exception:
                    continue
                mtype = payload.get("type")
                if mtype == "text":
                    session.feed_text(payload.get("text", ""))
                elif mtype == "ping":
                    session._send_json({"type": "pong"})
    except Exception as e:
        log("MAIN", f"ws loop error: {e}")
    finally:
        session.close()
        log("MAIN", "Browser client disconnected.")


@app.route("/call-eva")
def call_eva_page():
    return render_template("call_eva.html")


@app.route("/call", methods=["POST"])
def trigger_call():
    """Places an outbound call from your Twilio number to MY_PHONE_NUMBER.
    (Dev/demo route — real campaign calls go through /api/calls instead.)"""
    if not twilio_client:
        return jsonify({"ok": False, "error": "TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN missing in .env"}), 400
    if not TWILIO_PHONE_NUMBER or not MY_PHONE_NUMBER:
        return jsonify({"ok": False, "error": "TWILIO_PHONE_NUMBER or MY_PHONE_NUMBER missing in .env"}), 400
    if not PUBLIC_BASE_URL:
        return jsonify({
            "ok": False,
            "error": "PUBLIC_BASE_URL missing in .env - set it to your ngrok https URL, e.g. https://abcd1234.ngrok-free.app",
        }), 400

    try:
        call = twilio_client.calls.create(
            to=MY_PHONE_NUMBER,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{PUBLIC_BASE_URL}/twiml",
            method="POST",
        )
        log("MAIN", f"Outbound call started: {call.sid} -> {MY_PHONE_NUMBER}")
        return jsonify({"ok": True, "call_sid": call.sid})
    except Exception as e:
        log("MAIN", f"Call failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/twiml", methods=["GET", "POST"])
def twiml():
    """Twilio fetches this once the call connects; it tells Twilio to open
    a Media Stream WebSocket to us so audio can flow both ways."""
    ws_url = PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://") + "/ws/twilio"
    resp = VoiceResponse()
    connect = Connect()
    connect.stream(url=ws_url)
    resp.append(connect)
    return str(resp), 200, {"Content-Type": "text/xml"}


@sock.route("/ws/twilio")
def twilio_ws(ws):
    missing = check_missing_keys()
    if missing:
        log("MAIN", f"Twilio call rejected, missing keys: {missing}")
        return

    session = EvaSession(ws, mode="phone")
    if not session.start():
        return

    log("MAIN", "Twilio call connected.")
    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            try:
                data = json.loads(msg)
            except Exception:
                continue

            event = data.get("event")
            if event == "start":
                session.stream_sid = data["start"]["streamSid"]
                log("MAIN", f"Twilio stream started: {session.stream_sid}")
                session.speak("Hi there, how can I help you today?", "en")
            elif event == "media":
                audio = base64.b64decode(data["media"]["payload"])
                session.feed_audio(audio)
            elif event == "stop":
                log("MAIN", "Twilio stream stopped.")
                break
    except Exception as e:
        log("MAIN", f"twilio ws loop error: {e}")
    finally:
        session.close()
        log("MAIN", "Twilio call disconnected.")


# ============================================================
# Campaign calling API — called by PravaahAI
# ============================================================
@app.route("/api/calls", methods=["POST"])
def api_place_call():
    """PravaahAI calls this to have Eva place an outbound call to a lead.
    Body: {call_id, to_number, twilio:{account_sid,auth_token,from_number},
           agent:{...}, lead:{...}, callback_url}
    Auth: header X-Eva-Secret must match EVA_API_SECRET.
    """
    if not EVA_API_SECRET or request.headers.get("X-Eva-Secret") != EVA_API_SECRET:
        return jsonify({"ok": False, "error": "Invalid or missing X-Eva-Secret"}), 401
    if not PUBLIC_BASE_URL:
        return jsonify({"ok": False, "error": "PUBLIC_BASE_URL not set in Eva's .env"}), 400

    missing = check_missing_keys()
    if missing:
        return jsonify({"ok": False, "error": f"Eva missing env keys: {', '.join(missing)}"}), 500

    data = request.get_json(silent=True) or {}
    call_id = data.get("call_id")
    to_number = data.get("to_number")
    twilio_creds = data.get("twilio", {}) or {}
    agent = data.get("agent", {}) or {}
    lead = data.get("lead", {}) or {}
    meeting = data.get("meeting") or {}
    callback_url = data.get("callback_url")

    if not (call_id and to_number and callback_url):
        return jsonify({"ok": False, "error": "call_id, to_number and callback_url are required"}), 400

    account_sid = twilio_creds.get("account_sid")
    auth_token = twilio_creds.get("auth_token")
    from_number = _e164(twilio_creds.get("from_number", ""))
    if not (account_sid and auth_token and from_number):
        return jsonify({"ok": False, "error": "Twilio account_sid/auth_token/from_number are required"}), 400

    with _pending_calls_lock:
        PENDING_CALLS[call_id] = {
            "agent": agent, "lead": lead, "callback_url": callback_url, "created_at": time.time(),
            "meeting": meeting,
        }

    try:
        call_client = TwilioClient(account_sid, auth_token)
        call = call_client.calls.create(
            to=_e164(to_number),
            from_=from_number,
            url=f"{PUBLIC_BASE_URL}/twiml/outbound/{call_id}",
            method="POST",
        )
        log("MAIN", f"Outbound campaign call started: {call.sid} -> {to_number} (call_id={call_id})")
        return jsonify({"ok": True, "call_sid": call.sid})
    except Exception as e:
        with _pending_calls_lock:
            PENDING_CALLS.pop(call_id, None)
        log("MAIN", f"api_place_call failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/twiml/outbound/<call_id>", methods=["GET", "POST"])
def twiml_outbound(call_id):
    """Twilio fetches this once a campaign call connects; tells Twilio to
    open a Media Stream WebSocket scoped to this specific call_id."""
    ws_url = (
        PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
        + f"/ws/twilio-outbound/{call_id}"
    )
    resp = VoiceResponse()
    connect = Connect()
    connect.stream(url=ws_url)
    resp.append(connect)
    return str(resp), 200, {"Content-Type": "text/xml"}


@sock.route("/ws/twilio-outbound/<call_id>")
def twilio_outbound_ws(ws, call_id):
    with _pending_calls_lock:
        cfg = PENDING_CALLS.get(call_id)
    if not cfg:
        log("MAIN", f"No pending config for call_id={call_id}, closing.")
        return

    missing = check_missing_keys()
    if missing:
        log("MAIN", f"Outbound call rejected, missing keys: {missing}")
        return
#
    session = EvaSession(
        ws, mode="phone", call_id=call_id,
        agent=cfg["agent"], lead=cfg["lead"], callback_url=cfg["callback_url"],
        meeting=cfg.get("meeting"),
    )
    if not session.start():
        session._finish_and_callback(hangup_reason="failed_to_start")
        with _pending_calls_lock:
            PENDING_CALLS.pop(call_id, None)
        return

    log("MAIN", f"Outbound call {call_id} connected.")
    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            try:
                data = json.loads(msg)
            except Exception:
                continue

            event = data.get("event")
            if event == "start":
                session.stream_sid = data["start"]["streamSid"]
                session.call_started_at = time.time()
                opening = render_call_vars(
                    cfg["agent"].get("opening_line") or "Hi {{name}}, do you have a quick minute?",
                    cfg["lead"],
                )
                opening_lang = cfg["agent"].get("language") if cfg["agent"].get("language") in SUPPORTED_LANGUAGES else "en"
                session.speak(opening, opening_lang)
            elif event == "media":
                audio = base64.b64decode(data["media"]["payload"])
                session.feed_audio(audio)
            elif event == "stop":
                log("MAIN", f"Outbound call {call_id} stream stopped.")
                break
    except Exception as e:
        log("MAIN", f"twilio-outbound ws loop error: {e}")
        session.hangup_reason = "error"
    finally:
        session.close()
        session._finish_and_callback()
        with _pending_calls_lock:
            PENDING_CALLS.pop(call_id, None)
        log("MAIN", f"Outbound call {call_id} disconnected.")


@app.route("/api/calls/vanisetu", methods=["POST"])
def api_place_call_vanisetu():
    """PravaahAI calls this to have Eva place an outbound call over VaniSetu
    instead of Twilio. Body: {call_id, to_number, caller_id, agent, lead,
    callback_url}. Auth: header X-Eva-Secret must match EVA_API_SECRET."""
    if not EVA_API_SECRET or request.headers.get("X-Eva-Secret") != EVA_API_SECRET:
        return jsonify({"ok": False, "error": "Invalid or missing X-Eva-Secret"}), 401
    if not (VANISETU_TCODE and VANISETU_TOKEN):
        return jsonify({"ok": False, "error": "Eva has no VANISETU_TCODE/VANISETU_TOKEN configured"}), 400

    missing = check_missing_keys()
    if missing:
        return jsonify({"ok": False, "error": f"Eva missing env keys: {', '.join(missing)}"}), 500

    data = request.get_json(silent=True) or {}
    call_id = data.get("call_id")
    to_number = data.get("to_number")
    caller_id = data.get("caller_id")
    agent = data.get("agent", {}) or {}
    lead = data.get("lead", {}) or {}
    meeting = data.get("meeting") or {}
    callback_url = data.get("callback_url")

    if not (call_id and to_number and caller_id and callback_url):
        return jsonify({"ok": False, "error": "call_id, to_number, caller_id and callback_url are required"}), 400

    with _pending_calls_lock:
        PENDING_CALLS[call_id] = {
            "agent": agent, "lead": lead, "callback_url": callback_url,
            "created_at": time.time(), "meeting": meeting,
        }

    ok = vanisetu_client.place_outbound_call(request_id=call_id, endpoint=to_number, caller_id=caller_id)
    if not ok:
        with _pending_calls_lock:
            PENDING_CALLS.pop(call_id, None)
        return jsonify({"ok": False, "error": "VaniSetu connection not ready"}), 503

    log("MAIN", f"Outbound VaniSetu call requested: call_id={call_id} -> {to_number} (caller_id={caller_id})")
    return jsonify({"ok": True, "call_sid": call_id})


@app.route("/api/calls/voicelink", methods=["POST"])
def api_place_call_voicelink():
    """PravaahAI calls this to have Eva place an outbound call over
    VoiceLink instead of Twilio/VaniSetu. Body: {call_id, customer_number,
    did_number, voicelink_login_email, voicelink_login_password, agent,
    lead, callback_url, call_mode ("ai"|"ivr"), ivr:{audio_url, wait_secs, text}}.
    Auth: header X-Eva-Secret must match EVA_API_SECRET."""
    if not EVA_API_SECRET or request.headers.get("X-Eva-Secret") != EVA_API_SECRET:
        return jsonify({"ok": False, "error": "Invalid or missing X-Eva-Secret"}), 401
    if not PUBLIC_BASE_URL:
        return jsonify({"ok": False, "error": "PUBLIC_BASE_URL not set in Eva's .env"}), 400

    missing = check_missing_keys()
    if missing:
        return jsonify({"ok": False, "error": f"Eva missing env keys: {', '.join(missing)}"}), 500

    data = request.get_json(silent=True) or {}
    call_id = data.get("call_id")
    customer_number = data.get("customer_number")
    did_number = data.get("did_number")
    login_username = data.get("voicelink_login_username")
    login_password = data.get("voicelink_login_password")
    agent = data.get("agent", {}) or {}
    lead = data.get("lead", {}) or {}
    meeting = data.get("meeting") or {}
    callback_url = data.get("callback_url")
    call_mode = (data.get("call_mode") or "ai").strip().lower()
    browser_token = data.get("browser_token") or ""
    ivr = data.get("ivr") or {}
    if call_mode == "ivr" and not ivr.get("audio_url"):
        return jsonify({"ok": False, "error": "IVR call requires ivr.audio_url"}), 400
    if call_mode == "manual" and not browser_token:
        return jsonify({"ok": False, "error": "Manual call requires browser_token"}), 400

    log("MAIN", f"[VoiceLink] /api/calls/voicelink call_id={call_id} mode={call_mode} -> "
                f"customer_number={customer_number} did_number={did_number}")
    log("MAIN", f"[VoiceLink] agent from Pravah: {json.dumps(agent, default=str)}")
    log("MAIN", f"[VoiceLink] lead from Pravah: {json.dumps(lead, default=str)}")
    log("MAIN", f"[VoiceLink] meeting from Pravah: {json.dumps(meeting, default=str)}")
    if call_mode == "ivr":
        log("MAIN", f"[VoiceLink] ivr from Pravah: {json.dumps(ivr, default=str)}")

    if not (call_id and customer_number and did_number and callback_url):
        return jsonify({"ok": False, "error": "call_id, customer_number, did_number and callback_url are required"}), 400
    if not (login_username and login_password):
        return jsonify({"ok": False, "error": "voicelink_login_username and voicelink_login_password are required"}), 400
    if call_mode == "ivr" and not ivr.get("audio_url"):
        return jsonify({"ok": False, "error": "IVR call requires ivr.audio_url"}), 400

    with _pending_calls_lock:
        PENDING_CALLS[call_id] = {
            "agent": agent, "lead": lead, "callback_url": callback_url,
            "created_at": time.time(), "meeting": meeting,
            "call_mode": call_mode, "ivr": ivr,
            "browser_token": browser_token,
        }

    ws_scheme_base = PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
    call_websocket_url = f"{ws_scheme_base}/ws/voicelink/{call_id}"
    call_webhook_url = f"{PUBLIC_BASE_URL}/api/voicelink/webhook/{call_id}"

    result, err = voicelink_add_lead(
        login_username, login_password, did_number, customer_number,
        websocket_url=call_websocket_url, webhook_url=call_webhook_url,
        custom_parameters={"call_id": call_id},
    )
    if err:
        with _pending_calls_lock:
            PENDING_CALLS.pop(call_id, None)
        log("MAIN", f"api_place_call_voicelink failed: {err}")
        return jsonify({"ok": False, "error": err}), 502

    queue_id = result.get("outbound_queue_id") or (result.get("data") or {}).get("outbound_queue_id")
    log("MAIN", f"Outbound VoiceLink call requested: call_id={call_id} -> {customer_number} (did={did_number}, mode={call_mode})")
    return jsonify({"ok": True, "call_sid": queue_id or call_id})

@app.route("/api/voicelink/webhook/<call_id>", methods=["POST"])
def voicelink_webhook(call_id):
    payload = request.get_json(silent=True) or {}
    # VoiceLink nests call details under "call" in some payloads - flatten.
    _call = payload.get("call") if isinstance(payload.get("call"), dict) else {}
    if _call:
        payload = {**_call, **{k: v for k, v in payload.items() if k != "call"}}
        if _call.get("durationSec") is not None:
            payload["duration"] = _call["durationSec"]
    event = (payload.get("event") or "").lower()
    log("VOICELINK", f"webhook call_id={call_id} event={event} recordingUrl={payload.get('recordingUrl')!r}")
    log("VOICELINK", f"webhook FULL payload: {json.dumps(payload, default=str)[:2000]}")

    # Calls we placed carry a real 24-hex call_id in the URL. Calls where the customer
    # dialed our DID arrive as call_id="placeholder" -> map them back to the Pravah call
    # record that the websocket handler created for that call.
    if not re.fullmatch(r"[0-9a-f]{24}", call_id or ""):
        mapped = _lookup_inbound_call(payload)
        if not mapped:
            return jsonify({"received": True})   # e.g. call.initiated - websocket not connected yet
        log("VOICELINK", f"inbound webhook mapped via id/number -> call_id={mapped}")
        call_id = mapped

    hangup_cause = str(payload.get("hangupCause") or "")
    call_status = str(payload.get("callStatus") or "").upper()
    answered = bool(payload.get("answeredAt")) or event == "call.answered" or call_status == "ANSWERED"
    duration = payload.get("duration") or payload.get("durationSec") or 0

    if event in ("call.initiated", "call.ringing"):
        _mb = MANUAL_BRIDGES.get(call_id)
        if _mb:
            _mb.send_status("ringing" if event == "call.ringing" else "dialing")
        return jsonify({"received": True})

    if event == "call.answered":
        with _call_results_lock:
            e = CALL_RESULTS.setdefault(call_id, _new_call_result())
            e["answered"] = True
        _mb = MANUAL_BRIDGES.get(call_id)
        if _mb:
            _mb.send_status("answered")
        return jsonify({"received": True})

    if event == "call.completed":
        # Recording is ready. Store the URL, send to Pravah once the transcript is in too.
        with _call_results_lock:
            e = CALL_RESULTS.setdefault(call_id, _new_call_result())
            _rec = (
                payload.get("recordingUrl") or payload.get("recording_url")
                or payload.get("recordingURL") or payload.get("recording") or ""
            )
            e["recording_url"] = _rec if isinstance(_rec, str) else ""
            e["recording_done"] = True
            e["answered"] = True
            if hangup_cause:
                e["hangup_cause"] = hangup_cause
            if duration:
                e["duration_secs"] = duration
        _arm_finalize_timer(call_id)
        _finalize_call(call_id)
        return jsonify({"received": True})

    if event in ("call.ended", "call.failed"):
        with _call_results_lock:
            e = CALL_RESULTS.get(call_id)
            if e is not None:
                if answered:
                    e["answered"] = True
                if hangup_cause:
                    e["hangup_cause"] = hangup_cause
                if duration:
                    e["duration_secs"] = duration

        _mb = MANUAL_BRIDGES.get(call_id)
        if _mb:
            _mb.send_status("ended", reason=(payload.get("hangupReason") or call_status or event))

        bridge = VOICELINK_BRIDGES.get(call_id)
        if bridge:
            bridge.close()
            VOICELINK_BRIDGES.pop(call_id, None)
            return jsonify({"received": True})

        # Websocket never connected (no answer / rejected / busy / failed) -> report to Pravah now
        with _pending_calls_lock:
            cfg = PENDING_CALLS.pop(call_id, None)
        if cfg and cfg.get("callback_url"):
            info = {
                "answered": answered, "hangup_cause": hangup_cause,
                "hangup_reason": "", "duration_secs": 0,
                "extra": {"call_mode": cfg.get("call_mode", "ai")},
            }
            result = _classify_call_result(info, [])
            _post_call_result(cfg["callback_url"], {
                "call_id": call_id, "status": "no_response",
                "hangup_reason": payload.get("hangupReason") or call_status or event,
                "hangup_cause": hangup_cause, "answered": answered,
                "call_result": result, "call_mode": cfg.get("call_mode", "ai"),
                "duration_secs": 0, "transcript": [],
                "recording_url": payload.get("recordingUrl") or "",
            })
    return jsonify({"received": True})

@app.route("/api/internal/call-result/<call_id>", methods=["POST"])
def api_internal_call_result(call_id):
    """agent.py posts the final transcript here when a VoiceLink call ends.
    Auth: same X-Eva-Secret as everything else."""
    if not EVA_API_SECRET or request.headers.get("X-Eva-Secret") != EVA_API_SECRET:
        return jsonify({"ok": False, "error": "Invalid or missing X-Eva-Secret"}), 401
    data = request.get_json(silent=True) or {}
    with _call_results_lock:
        e = CALL_RESULTS.setdefault(call_id, _new_call_result())
        if data.get("callback_url"):
            e["callback_url"] = data["callback_url"]
        e["transcript"] = data.get("transcript") or []
        e["status"] = data.get("status")
        e["hangup_reason"] = data.get("hangup_reason") or "completed"
        if not e["duration_secs"]:
            e["duration_secs"] = data.get("duration_secs") or 0
    log("VOICELINK", f"transcript received for {call_id} ({len(e['transcript'])} lines), waiting for recording url")
    _arm_finalize_timer(call_id)
    _finalize_call(call_id)
    return jsonify({"ok": True})


# ============================================================
# VoiceLink IVR — plays a fixed recording, waits, STT, hangs up
# (completely separate from the LiveKit bridge)
# ============================================================
def _norm_codec(c) -> str:
    c = str(c or "").lower()
    return "ulaw" if ("ulaw" in c or "mulaw" in c or "pcmu" in c) else "alaw"


def _codec_from_start(start_data) -> str:
    """The line codec VoiceLink announces in its `start` event (media_format.encoding).
    The docs say A-law, but real inbound calls arrive as audio/ulaw - so never assume."""
    sd = start_data or {}
    mf = (sd.get("start") or {}).get("media_format") or sd.get("media_format") or {}
    enc = str(mf.get("encoding") or "")
    return _norm_codec(enc) if enc else _norm_codec(VOICELINK_CODEC)


def _ivr_decode(b: bytes, codec: str = None) -> bytes:
    """Phone codec -> linear16."""
    if _norm_codec(codec or VOICELINK_CODEC) == "ulaw":
        return audioop.ulaw2lin(b, 2)
    return audioop.alaw2lin(b, 2)


def _ivr_encode(pcm: bytes, codec: str = None) -> bytes:
    """linear16 -> phone codec."""
    if _norm_codec(codec or VOICELINK_CODEC) == "ulaw":
        return audioop.lin2ulaw(pcm, 2)
    return audioop.lin2alaw(pcm, 2)


_IVR_PCM_CACHE = {}   # url -> linear16 8kHz mono (codec-independent, encoded per call)


def load_ivr_audio(url: str, codec: str = None) -> bytes:
    """Downloads the WAV once, converts it to 8kHz mono linear16 (cached per URL),
    then encodes it in the codec this call's line actually uses."""
    with _ivr_cache_lock:
        pcm = _IVR_PCM_CACHE.get(url)
    if pcm is None:
        resp = requests.get(url, timeout=25)
        resp.raise_for_status()
        with wave.open(io.BytesIO(resp.content), "rb") as wf:
            channels, width, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        if width != 2:
            frames = audioop.lin2lin(frames, width, 2)
        if channels > 1:
            frames = audioop.tomono(frames, 2, 0.5, 0.5)
        if rate != VOICELINK_RATE:
            frames, _ = audioop.ratecv(frames, 2, 1, rate, VOICELINK_RATE, None)
        pcm = frames
        with _ivr_cache_lock:
            if len(_IVR_PCM_CACHE) > 50:
                _IVR_PCM_CACHE.clear()
            _IVR_PCM_CACHE[url] = pcm
    return _ivr_encode(pcm, codec)


def deepgram_transcribe_pcm(pcm16: bytes, rate: int) -> str:
    """One-shot (pre-recorded) Deepgram STT on the caller's reply."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm16)
    resp = requests.post(
        "https://api.deepgram.com/v1/listen",
        params={"model": "nova-3", "language": IVR_STT_LANGUAGE, "smart_format": "true", "punctuate": "true"},
        headers={"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/wav"},
        data=buf.getvalue(), timeout=25,
    )
    resp.raise_for_status()
    alt = resp.json()["results"]["channels"][0]["alternatives"][0]
    return (alt.get("transcript") or "").strip()


def deepgram_transcribe_segments(pcm16: bytes, rate: int):
    """Pre-recorded Deepgram STT that returns [(start_secs, text), ...] so two
    sides of a manual call can be merged back into one conversation."""
    if len(pcm16) < int(rate * 2 * 0.5):
        return []
    win = int(rate * 0.1) * 2
    if not any(audioop.rms(pcm16[i:i + win], 2) >= 150 for i in range(0, len(pcm16), win)):
        return []   # silence only - don't pay for STT
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm16)
    resp = requests.post(
        "https://api.deepgram.com/v1/listen",
        params={"model": "nova-3", "language": IVR_STT_LANGUAGE, "smart_format": "true",
                "punctuate": "true", "utterances": "true"},
        headers={"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/wav"},
        data=buf.getvalue(), timeout=120,
    )
    resp.raise_for_status()
    res = resp.json()["results"]
    utts = res.get("utterances") or []
    out = [(float(u.get("start", 0) or 0), (u.get("transcript") or "").strip()) for u in utts]
    out = [(s, t) for s, t in out if t]
    if out:
        return out
    alt = res["channels"][0]["alternatives"][0]
    txt = (alt.get("transcript") or "").strip()
    return [(0.0, txt)] if txt else []


SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_STT_MODEL = os.environ.get("SARVAM_STT_MODEL", "saaras:v3")
SARVAM_STT_MODE = os.environ.get("SARVAM_STT_MODE", "transcribe")   # saaras:v3: transcribe | codemix | verbatim | translit | translate


def sarvam_transcribe_pcm(pcm16: bytes, rate: int, language_code: str = "unknown") -> str:
    """One-shot Sarvam STT (saaras:v3). language_code: 'unknown' = auto-detect, or hi-IN / en-IN / ta-IN ..."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm16)
    form = {"model": SARVAM_STT_MODEL, "language_code": language_code}
    if SARVAM_STT_MODEL.startswith("saaras"):
        form["mode"] = SARVAM_STT_MODE
    last_err = None
    for _ in range(2):
        try:
            resp = requests.post(
                SARVAM_STT_URL,
                headers={"api-subscription-key": SARVAM_API_KEY},
                files={"file": ("reply.wav", buf.getvalue(), "audio/wav")},
                data=form, timeout=25,
            )
            resp.raise_for_status()
            return (resp.json().get("transcript") or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(0.3)
    raise last_err


def transcribe_ivr_pcm(pcm16: bytes, rate: int, language: str = "") -> str:
    """IVR reply -> text, Sarvam only. English / Hindi / auto -> 'unknown' (auto-detect, so a Hindi reply
    to an English IVR still works). Regional IVR languages -> that language's code."""
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set")
    cfg = LANGUAGES.get(language or "")
    code = cfg["tts"] if (cfg and language not in ("en", "hi")) else "unknown"
    return sarvam_transcribe_pcm(pcm16, rate, code)



def sarvam_transcribe_pcm(pcm16: bytes, rate: int, language_code: str) -> str:
    """One-shot Sarvam STT (supports hi/bn/gu/kn/ml/mr/od/pa/ta/te/en)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm16)
    resp = requests.post(
        SARVAM_STT_URL,
        headers={"api-subscription-key": SARVAM_API_KEY},
        files={"file": ("reply.wav", buf.getvalue(), "audio/wav")},
        data={"model": SARVAM_STT_MODEL, "language_code": language_code},
        timeout=25,
    )
    resp.raise_for_status()
    return (resp.json().get("transcript") or "").strip()


def transcribe_ivr_pcm(pcm16: bytes, rate: int, language: str = "") -> str:
    """Regional IVR languages -> Sarvam STT. English / Hindi / auto -> Deepgram (unchanged)."""
    cfg = LANGUAGES.get(language or "")
    if cfg and language not in ("en", "hi") and SARVAM_API_KEY:
        try:
            return sarvam_transcribe_pcm(pcm16, rate, cfg["tts"])
        except Exception as e:
            log("IVR", f"Sarvam STT failed ({language}), falling back to Deepgram: {e}")
    return deepgram_transcribe_pcm(pcm16, rate)




class VoiceLinkIVRSession:
    """One IVR call: play audio -> wait for reply -> STT -> hang up -> report."""

    def __init__(self, ws, call_id, cfg):
        self.ws = ws
        self.call_id = call_id
        self.cfg = cfg
        self.ivr = cfg.get("ivr") or {}
        self.callback_url = cfg.get("callback_url")
        self.audio_url = self.ivr.get("audio_url", "")
        try:
            self.wait_secs = float(self.ivr.get("wait_secs") or IVR_WAIT_SECS)
        except (TypeError, ValueError):
            self.wait_secs = IVR_WAIT_SECS

        self.started = threading.Event()
        self.stop_event = threading.Event()
        self.listening = threading.Event()
        self.done = threading.Event()
        self.ws_lock = threading.Lock()
        self.inbound_lock = threading.Lock()
        self.inbound = bytearray()      # linear16 of what the caller said during the wait window
        self.last_rms = 0

        self.stream_sid = None
        self.call_started_at = None
        self.codec = _norm_codec(VOICELINK_CODEC)   # overwritten from the start event
        self.duration = 0
        self.transcript = []
        self.reply_text = ""
        self.answered = False
        self.lead_status = "cold"
        self.hangup_reason = "ivr_completed"
        self._reported = False
        self._report_lock = threading.Lock()

    # ----- called from the websocket loop -----
    def start(self):
        threading.Thread(target=self._run, daemon=True, name=f"IVR-{self.call_id}").start()

    def on_start(self, data):
        start_obj = data.get("start") or {}
        self.stream_sid = (start_obj.get("stream_sid") or start_obj.get("streamSid")
                           or data.get("stream_sid") or data.get("streamSid"))
        # needed to hang up the call from our side (stop event carries callSid)
        self.call_sid = (start_obj.get("call_sid") or start_obj.get("callSid")
                         or data.get("call_sid") or data.get("callSid"))
        self.call_started_at = time.time()
        self.codec = _codec_from_start(data)
        self.started.set()
        log("IVR", f"call {self.call_id}: start event received (call_sid={self.call_sid}, codec={self.codec})")

    def feed_alaw(self, audio: bytes):
        if not audio or not self.listening.is_set():
            return
        try:
            pcm = _ivr_decode(audio, self.codec)
            self.last_rms = audioop.rms(pcm, 2)
        except Exception:
            return
        with self.inbound_lock:
            self.inbound.extend(pcm)

    def close(self):
        """Called when the call/websocket ends (or VoiceLink's call.ended webhook arrives)."""
        self.stop_event.set()

    # ----- internals -----
    def _send_media(self, audio_bytes: bytes) -> bool:
        with self.ws_lock:
            try:
                self.ws.send(json.dumps({
                    "event": "media",
                    "media": {"payload": base64.b64encode(audio_bytes).decode("ascii")},
                }))
                return True
            except Exception:
                return False

    def _play(self, audio: bytes) -> bool:
        """Sends the audio paced at real-time speed. Returns False if the call ended mid-way."""
        chunk_bytes = max(160, int(VOICELINK_RATE * 0.1))   # 100ms per chunk (1 byte/sample)
        t0 = time.time()
        for i in range(0, len(audio), chunk_bytes):
            if self.stop_event.is_set():
                return False
            wait = (t0 + (i / VOICELINK_RATE) - IVR_PLAY_LEAD_SECS) - time.time()
            if wait > 0 and self.stop_event.wait(wait):
                return False
            if not self._send_media(audio[i:i + chunk_bytes]):
                return False
        end_at = t0 + len(audio) / VOICELINK_RATE + 0.1
        remaining = end_at - time.time()
        if remaining > 0 and self.stop_event.wait(remaining):
            return False
        return True

    def _transcribe(self) -> str:
        with self.inbound_lock:
            pcm = bytes(self.inbound)
        if len(pcm) < int(VOICELINK_RATE * 2 * 0.3):
            return ""
        win = int(VOICELINK_RATE * 0.1) * 2
        loud = any(audioop.rms(pcm[i:i + win], 2) >= IVR_MIN_SPEECH_RMS for i in range(0, len(pcm), win))
        if not loud:
            return ""
        try:
            return transcribe_ivr_pcm(pcm, VOICELINK_RATE, self.ivr.get("language", ""))
        except Exception as e:
            log("IVR", f"call {self.call_id}: STT failed: {e}")
            return ""

    def _hangup(self):
        """Hang up from Eva's side: send VoiceLink's `stop` event with the
        callSid we got in the `start` event, then close the socket as a fallback."""
        call_sid = getattr(self, "call_sid", None)
        if call_sid:
            with self.ws_lock:
                try:
                    self.ws.send(json.dumps({"event": "stop", "stop": {"callSid": call_sid}}))
                    log("IVR", f"call {self.call_id}: stop event sent (call_sid={call_sid})")
                except Exception as e:
                    log("IVR", f"call {self.call_id}: could not send stop event: {e}")
            time.sleep(0.3)   # let VoiceLink process the stop before the socket closes
        try:
            self.ws.close()
        except Exception:
            pass



    def _run(self):
        try:
            try:
                load_ivr_audio(self.audio_url)      # warm the download/cache while we wait for 'start'
            except Exception as e:
                log("IVR", f"call {self.call_id}: could not load IVR audio: {e}")
                self.hangup_reason = "ivr_audio_error"
                return

            if not self.started.wait(timeout=20):
                self.hangup_reason = "no_start_event"
                return

            alaw = load_ivr_audio(self.audio_url, self.codec)   # encode in the codec the line really uses

            if not self._play(alaw):
                self.hangup_reason = "hangup_during_playback"
                return

            self.transcript.append({
                "role": "agent",
                "text": (self.ivr.get("text") or "[IVR recording played]"),
                "ts": time.time(),
            })

            # wait for the caller's reply
            self.listening.set()
            self.stop_event.wait(self.wait_secs)
            extended = 0.0
            while (not self.stop_event.is_set() and extended < IVR_MAX_EXTEND_SECS
                   and self.last_rms >= IVR_MIN_SPEECH_RMS):
                self.stop_event.wait(0.2)
                extended += 0.2
            self.listening.clear()

            text = self._transcribe()
            if text:
                self.answered = True
                self.reply_text = text
                self.lead_status = "cold" if ivr_is_negative(text) else "warm"
                self.transcript.append({"role": "lead", "text": text, "ts": time.time()})
            else:
                self.answered = False
                self.lead_status = "cold"
                self.hangup_reason = "no_reply"
            log("IVR", f"call {self.call_id}: reply={text!r} -> {self.lead_status}")
        except Exception as e:
            log("IVR", f"call {self.call_id}: error {type(e).__name__}: {e!r}")
            self.hangup_reason = "error"
        finally:
            if self.call_started_at:
                self.duration = round(time.time() - self.call_started_at, 1)
            self.listening.clear()
            self._hangup()
            self.done.set()

    def report(self):
        """Hands transcript + result to the same VoiceLink finalize flow
        (waits for recording URL from call.completed, then POSTs to Pravah)."""
        with self._report_lock:
            if self._reported:
                return
            self._reported = True
        with _call_results_lock:
            e = CALL_RESULTS.setdefault(self.call_id, _new_call_result())
            e["callback_url"] = e["callback_url"] or self.callback_url
            e["transcript"] = self.transcript
            e["status"] = "completed" if self.answered else "no_response"
            e["hangup_reason"] = self.hangup_reason
            e["answered"] = bool(e.get("answered")) or self.started.is_set()
            if not e["duration_secs"]:
                e["duration_secs"] = self.duration
            e["extra"] = {
                "call_mode": "ivr",
                "ivr_answered": self.answered,
                "ivr_reply": self.reply_text,
                "ivr_lead_status": self.lead_status,
            }
        _arm_finalize_timer(self.call_id)
        _finalize_call(self.call_id)


def _run_voicelink_ivr(ws, call_id, cfg, start_data=None):
    session = VoiceLinkIVRSession(ws, call_id, cfg)
    VOICELINK_BRIDGES[call_id] = session      # so the call.ended webhook can close it
    session.start()
    if start_data:                            # inbound: the start event was already consumed
        session.on_start(start_data)
    log("VOICELINK", f"call {call_id}: IVR mode, audio={session.audio_url}")

    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            if isinstance(msg, (bytes, bytearray)):
                continue
            try:
                data = json.loads(msg)
            except Exception:
                continue

            event = data.get("event")
            if event == "start":
                session.on_start(data)
            elif event == "media":
                media = data.get("media", {}) or {}
                if media.get("track", "inbound") != "inbound":
                    continue
                payload_b64 = media.get("payload")
                if payload_b64:
                    try:
                        session.feed_alaw(base64.b64decode(payload_b64))
                    except Exception:
                        continue
            elif event == "stop":
                log("VOICELINK", f"call {call_id}: stop event")
                break
    except Exception as e:
        log("VOICELINK", f"IVR ws loop error: {e}")
    finally:
        session.close()
        session.done.wait(timeout=25)
        session.report()
        VOICELINK_BRIDGES.pop(call_id, None)
        with _pending_calls_lock:
            PENDING_CALLS.pop(call_id, None)
        log("VOICELINK", f"IVR call {call_id} disconnected.")


# ============================================================
# VoiceLink INBOUND  (customer dials our DID)
# ============================================================
INBOUND_CALL_MAP = {}            # "sid:<call_sid>" / "pair:<from10>:<to10>" -> {"call_id", "ts"}
_inbound_map_lock = threading.Lock()
INBOUND_MAP_TTL_SECS = 3 * 3600


def _last10(num) -> str:
    return re.sub(r"\D", "", str(num or ""))[-10:]


def _register_inbound_call(call_id, call_sid, from_number, to_number):
    """Remembers which Pravah call_id belongs to a VoiceLink inbound call, so the
    'placeholder' webhooks (call.completed -> recording url) can find it later."""
    now = time.time()
    with _inbound_map_lock:
        for k in [k for k, v in INBOUND_CALL_MAP.items() if now - v["ts"] > INBOUND_MAP_TTL_SECS]:
            INBOUND_CALL_MAP.pop(k, None)
        entry = {"call_id": call_id, "ts": now}
        if call_sid:
            INBOUND_CALL_MAP[f"sid:{call_sid}"] = entry
        INBOUND_CALL_MAP[f"pair:{_last10(from_number)}:{_last10(to_number)}"] = entry
    log("VOICELINK", f"inbound registered call_id={call_id} call_sid={call_sid} from={from_number} to={to_number}")


def _lookup_inbound_call(payload):
    with _inbound_map_lock:
        for key in (payload.get("id"), payload.get("callSid"), payload.get("call_sid")):
            if key:
                e = INBOUND_CALL_MAP.get(f"sid:{key}")
                if e:
                    return e["call_id"]
        e = INBOUND_CALL_MAP.get(f"pair:{_last10(payload.get('from'))}:{_last10(payload.get('to'))}")
        return e["call_id"] if e else None


def fetch_inbound_call_context(did_number, from_number, provider_call_ref=""):
    """Asks Pravah which agent owns this DID; Pravah also creates the lead + call record."""
    if not PRAVAAH_API_BASE_URL or not EVA_API_SECRET:
        return None, "Eva is not configured to talk to PravaahAI"
    try:
        resp = requests.post(
            f"{PRAVAAH_API_BASE_URL}/api/eva-webhook/inbound-call",
            headers={"X-Eva-Secret": EVA_API_SECRET, "Content-Type": "application/json"},
            json={"did_number": did_number, "from_number": from_number,
                  "provider": "voicelink", "provider_call_ref": provider_call_ref},
            timeout=8,
        )
        data = resp.json()
        if resp.status_code >= 400:
            return None, data.get("error", f"HTTP {resp.status_code}")
        return data, None
    except Exception as e:
        return None, str(e)


def _prepare_inbound_greeting(agent: dict, codec: str = None):
    """Returns (audio_bytes | None, text) already encoded in the line's codec. Prefers the
    agent's pre-recorded opening audio (instant, cached after the first call); falls back
    to one quick Sarvam TTS call."""
    line = re.sub(r"\{\{[^}]*\}\}", "", agent.get("opening_line") or "")
    line = re.sub(r'[“”„«»"]', "", line)
    line = re.sub(r"\s+([,.!?।])", r"\1", line)
    line = re.sub(r"\s{2,}", " ", line).strip(" ,")

    url = (agent.get("opening_audio_url") or "").strip()
    text = (agent.get("opening_audio_text") or "").strip() or line
    if url:
        try:
            return load_ivr_audio(url, codec), text
        except Exception as e:
            log("INBOUND", f"could not load greeting recording: {e}")

    if line and SPEAKABLE_RE.search(line) and SARVAM_API_KEY:
        try:
            lang = agent.get("language") if agent.get("language") in SUPPORTED_LANGUAGES else "hi"
            voice = VOICE_MALE if agent.get("gender") == "male" else VOICE_FEMALE
            pcm, sr = sarvam_tts_synthesize(line, lang, voice, VOICELINK_RATE)
            if sr != VOICELINK_RATE:
                pcm, _ = audioop.ratecv(pcm, 2, 1, sr, VOICELINK_RATE, None)
            return _ivr_encode(pcm, codec), line
        except Exception as e:
            log("INBOUND", f"greeting TTS fallback failed: {e}")
    return None, ""


def _voicelink_hangup(ws, call_sid):
    """Cuts the call from our side: VoiceLink `stop` event with callSid, then close the socket."""
    try:
        if call_sid:
            ws.send(json.dumps({"event": "stop", "stop": {"callSid": call_sid}}))
            time.sleep(0.3)
    except Exception:
        pass
    try:
        ws.close()
    except Exception:
        pass


def _run_voicelink_inbound(ws, placeholder_id):
    """Customer dialed one of our VoiceLink DIDs.
      1) wait for VoiceLink's `start` event (has from / to / call_sid)
      2) ask Pravah which agent owns that DID (+ lead + call record)
      3) IVR agent -> same IVR flow as outbound
         AI agent  -> play the greeting recording right away while LiveKit room /
                      STT / LLM / TTS warm up in the background. Caller audio is
                      buffered meanwhile, so nothing they say is lost."""
    start_data = None
    try:
        while True:
            msg = ws.receive()
            if msg is None:
                return
            if isinstance(msg, (bytes, bytearray)):
                continue
            try:
                data = json.loads(msg)
            except Exception:
                continue
            ev = data.get("event")
            if ev == "connected":
                log("VOICELINK", "inbound: connected")
            elif ev == "start":
                start_data = data
                break
            elif ev == "stop":
                return
    except Exception as e:
        log("VOICELINK", f"inbound: ws error before start: {e}")
        return

    start_obj = start_data.get("start") or {}
    stream_sid = (start_obj.get("stream_sid") or start_obj.get("streamSid")
                  or start_data.get("stream_sid") or start_data.get("streamSid"))
    call_sid = (start_obj.get("call_sid") or start_obj.get("callSid")
                or start_data.get("call_sid") or start_data.get("callSid"))
    from_number = str(start_obj.get("from") or "")
    to_number = str(start_obj.get("to") or "")
    log("VOICELINK", f"inbound start: from={from_number} to={to_number} call_sid={call_sid} payload={str(start_data)[:600]}")

    ctx, err = fetch_inbound_call_context(to_number, from_number, call_sid or "")
    if not ctx:
        log("VOICELINK", f"inbound call to {to_number} rejected: {err}")
        _voicelink_hangup(ws, call_sid)
        return

    call_id = ctx["call_id"]
    _register_inbound_call(call_id, call_sid, from_number, to_number)
    cfg = {
        "agent": ctx.get("agent") or {}, "lead": ctx.get("lead") or {},
        "meeting": ctx.get("meeting"), "callback_url": ctx.get("callback_url"),
        "call_mode": ctx.get("call_mode", "ai"), "ivr": ctx.get("ivr") or {},
    }

    # ---------- IVR on this agent: play recording -> wait -> STT -> hang up ----------
    if cfg["call_mode"] == "ivr":
        _run_voicelink_ivr(ws, call_id, cfg, start_data=start_data)
        return

    # ---------- AI agent ----------
    codec = _codec_from_start(start_data)
    log("VOICELINK", f"inbound {call_id}: line codec={codec}")
    greeting_audio, greeting_text = _prepare_inbound_greeting(cfg["agent"], codec)
    bridge = livekit_bridge.VoiceLinkBridge(
        call_id=call_id, ws=ws, agent_cfg=cfg["agent"], lead=cfg["lead"],
        meeting=cfg.get("meeting"), callback_url=cfg.get("callback_url"),
        inbound=True, greeting_audio=greeting_audio, greeting_text=greeting_text,
        call_sid=call_sid, codec=codec,
    )
    bridge.stream_sid = stream_sid
    bridge.started.set()
    VOICELINK_BRIDGES[call_id] = bridge

    # 1) caller hears the greeting immediately
    if greeting_audio:
        threading.Thread(target=bridge.play_greeting, daemon=True, name=f"VLGreeting-{call_id}").start()

    # 2) LiveKit room + agent (STT/LLM/TTS) warm up in the background
    def _warmup():
        try:
            bridge.start()
            log("VOICELINK", f"inbound {call_id}: LiveKit room {bridge.room_name} ready")
        except Exception as e:
            log("VOICELINK", f"inbound {call_id}: bridge failed to start: {type(e).__name__}: {e!r}")
            bridge.close()
            _voicelink_hangup(ws, call_sid)

    threading.Thread(target=_warmup, daemon=True, name=f"VLWarmup-{call_id}").start()
    log("VOICELINK", f"Inbound call {call_id}: greeting={'recording/TTS' if greeting_audio else 'by agent'}, "
                     f"warming up room {bridge.room_name}")

    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            if isinstance(msg, (bytes, bytearray)):
                continue
            try:
                data = json.loads(msg)
            except Exception:
                continue
            event = data.get("event")
            if event == "media":
                media = data.get("media", {}) or {}
                if media.get("track", "inbound") != "inbound":
                    continue
                payload_b64 = media.get("payload")
                if payload_b64:
                    try:
                        bridge.feed_alaw(base64.b64decode(payload_b64))
                    except Exception:
                        continue
            elif event == "stop":
                log("VOICELINK", f"inbound call {call_id}: stop event")
                break
    except Exception as e:
        log("VOICELINK", f"inbound ws loop error: {e}")
    finally:
        bridge.close()
        VOICELINK_BRIDGES.pop(call_id, None)
        with _call_results_lock:
            _e = CALL_RESULTS.setdefault(call_id, _new_call_result())
            _e["callback_url"] = _e["callback_url"] or cfg.get("callback_url")
            _e["answered"] = True
        _arm_finalize_timer(call_id)
        log("VOICELINK", f"Inbound call {call_id} disconnected.")


# ============================================================
# MANUAL DIALER — a person on the dashboard talks through the browser mic,
# VoiceLink carries the call to the lead. No AI involved.
#
#   browser ws  /ws/dialer/<call_id>?token=...   <->  ManualDialerBridge  <->  VoiceLink ws /ws/voicelink/<call_id>
#
# browser -> Eva : PCM16 mono at the browser's own rate (sent in a "hello" message)
# Eva -> browser : PCM16 mono at MANUAL_BROWSER_RATE (phone audio upsampled 8k -> 16k)
# ============================================================
MANUAL_BRIDGES = {}                 # call_id -> ManualDialerBridge
_manual_lock = threading.Lock()
MANUAL_BROWSER_RATE = 16000
MANUAL_RING_TIMEOUT_SECS = int(os.environ.get("EVA_MANUAL_RING_TIMEOUT_SECS", 75))
MANUAL_BRIDGE_KEEP_SECS = 300       # keep a finished bridge around so a late VoiceLink socket can still be hung up


class ManualDialerBridge:
    def __init__(self, call_id, cfg):
        self.call_id = call_id
        self.callback_url = (cfg or {}).get("callback_url")

        self.browser_ws = None
        self.browser_lock = threading.Lock()
        self.browser_rate = MIC_RATE

        self.vl_ws = None
        self.vl_lock = threading.Lock()
        self.codec = _norm_codec(VOICELINK_CODEC)
        self.stream_sid = None
        self.call_sid = None
        self.vl_started = threading.Event()

        self.closed = threading.Event()
        self._finish_lock = threading.Lock()
        self._report_lock = threading.Lock()
        self._reported = False

        self._down_state = None      # browser -> phone resample state
        self._up_state = None        # phone -> browser resample state
        self._out_buf = b""          # re-chunk outgoing audio into 20 ms frames

        self._last_status = "dialing"
        self.rec_phone = bytearray()     # lead's voice, linear16 @ 8 kHz (for transcription)
        self.rec_browser = bytearray()   # dashboard user's voice, linear16 @ 8 kHz
        self.REC_MAX_BYTES = 8000 * 2 * 60 * 45   # cap: 45 min per side
        self.call_started_at = None
        self.answered = False
        self.hangup_reason = "completed"

    # ---------- browser side ----------
    def _send_browser_json(self, obj):
        ws = self.browser_ws
        if not ws:
            return
        with self.browser_lock:
            try:
                ws.send(json.dumps(obj))
            except Exception:
                pass

    def attach_browser(self, ws):
        self.browser_ws = ws
        self._send_browser_json({"type": "ready", "play_rate": MANUAL_BROWSER_RATE})
        self._send_browser_json({"type": "status", "state": self._last_status})

    def set_browser_rate(self, rate):
        if 8000 <= rate <= 96000 and rate != self.browser_rate:
            self.browser_rate = rate
            self._down_state = None

    def send_status(self, state, **extra):
        self._last_status = state
        msg = {"type": "status", "state": state}
        msg.update(extra)
        self._send_browser_json(msg)

    def start_ring_watchdog(self):
        def _check():
            if not self.vl_started.is_set() and not self.closed.is_set():
                log("DIALER", f"call {self.call_id}: nobody answered in {MANUAL_RING_TIMEOUT_SECS}s")
                self.finish("no_answer_timeout")
        t = threading.Timer(MANUAL_RING_TIMEOUT_SECS, _check)
        t.daemon = True
        t.start()

    # ---------- VoiceLink side ----------
    def attach_voicelink(self, ws):
        self.vl_ws = ws

    def on_voicelink_start(self, data):
        start_obj = data.get("start") or {}
        self.stream_sid = (start_obj.get("stream_sid") or start_obj.get("streamSid")
                           or data.get("stream_sid") or data.get("streamSid"))
        self.call_sid = (start_obj.get("call_sid") or start_obj.get("callSid")
                         or data.get("call_sid") or data.get("callSid"))
        self.codec = _codec_from_start(data)
        self.answered = True
        self.call_started_at = time.time()
        self.vl_started.set()
        log("DIALER", f"call {self.call_id}: VoiceLink audio started (call_sid={self.call_sid}, codec={self.codec})")
        if self.closed.is_set():          # the user already hung up while it was ringing
            self._hangup_vl()
            return
        self.send_status("connected")
        if self.browser_ws is None:
            def _no_browser():
                if self.browser_ws is None and not self.closed.is_set():
                    log("DIALER", f"call {self.call_id}: nobody connected the audio websocket, dropping the call")
                    self.finish("browser_not_connected")
            _t = threading.Timer(float(os.environ.get("EVA_MANUAL_BROWSER_GRACE_SECS", 20)), _no_browser)
            _t.daemon = True
            _t.start()

    def _send_vl_media(self, chunk: bytes):
        ws = self.vl_ws
        if not ws:
            return
        with self.vl_lock:
            try:
                ws.send(json.dumps({
                    "event": "media",
                    "stream_sid": self.stream_sid,
                    "media": {"payload": base64.b64encode(chunk).decode("ascii")},
                }))
            except Exception:
                pass

    def feed_browser_audio(self, pcm: bytes):
        """browser mic -> phone"""
        if self.closed.is_set() or not self.vl_started.is_set():
            return
        if len(pcm) % 2:
            pcm = pcm[:-1]
        if not pcm:
            return
        try:
            if self.browser_rate != VOICELINK_RATE:
                pcm, self._down_state = audioop.ratecv(pcm, 2, 1, self.browser_rate, VOICELINK_RATE, self._down_state)
            if len(self.rec_browser) < self.REC_MAX_BYTES:
                self.rec_browser.extend(pcm)
            self._out_buf += _ivr_encode(pcm, self.codec)
        except Exception as e:
            log("DIALER", f"call {self.call_id}: browser audio error {e}")
            return
        while len(self._out_buf) >= 160:
            chunk, self._out_buf = self._out_buf[:160], self._out_buf[160:]
            self._send_vl_media(chunk)

    def feed_phone_audio(self, raw: bytes):
        """phone -> browser speaker"""
        ws = self.browser_ws
        if self.closed.is_set():
            return
        try:
            pcm = _ivr_decode(raw, self.codec)
            if len(self.rec_phone) < self.REC_MAX_BYTES:
                self.rec_phone.extend(pcm)
            if not ws:
                return
            pcm, self._up_state = audioop.ratecv(pcm, 2, 1, VOICELINK_RATE, MANUAL_BROWSER_RATE, self._up_state)
        except Exception:
            return
        with self.browser_lock:
            try:
                ws.send(pcm)
            except Exception:
                pass

    def _hangup_vl(self):
        ws, sid = self.vl_ws, self.call_sid
        if not ws:
            return
        with self.vl_lock:
            try:
                if sid:
                    ws.send(json.dumps({"event": "stop", "stop": {"callSid": sid}}))
            except Exception:
                return
        time.sleep(0.3)
        try:
            ws.close()
        except Exception:
            pass

    # ---------- lifecycle ----------
    def finish(self, reason="completed"):
        """Idempotent: tells the browser the call is over, hangs up VoiceLink, closes the browser socket."""
        with self._finish_lock:
            if self.closed.is_set():
                return
            self.closed.set()
            self.hangup_reason = reason
        log("DIALER", f"call {self.call_id}: finished ({reason})")
        self._send_browser_json({"type": "status", "state": "ended", "reason": reason})
        self._hangup_vl()

        def _close_browser():
            ws = self.browser_ws
            if ws:
                try:
                    ws.close()
                except Exception:
                    pass
        t = threading.Timer(0.6, _close_browser)
        t.daemon = True
        t.start()

        def _forget():
            with _manual_lock:
                MANUAL_BRIDGES.pop(self.call_id, None)
        t2 = threading.Timer(MANUAL_BRIDGE_KEEP_SECS, _forget)
        t2.daemon = True
        t2.start()

    def close(self):
        """Called by the VoiceLink call.ended webhook (via VOICELINK_BRIDGES)."""
        self.finish("call_ended")

    def report(self):
        """Transcribes both sides (you = 'agent', lead = 'lead'), then hands the
        result to the same finalize flow (waits for recording URL -> ONE callback)."""
        with self._report_lock:
            if self._reported:
                return
            self._reported = True
        duration = round(time.time() - self.call_started_at, 1) if self.call_started_at else 0
        with _call_results_lock:
            e = CALL_RESULTS.setdefault(self.call_id, _new_call_result())
            e["callback_url"] = e["callback_url"] or self.callback_url
            e["status"] = "completed" if self.answered else "no_response"
            e["hangup_reason"] = self.hangup_reason
            e["answered"] = bool(e.get("answered")) or self.answered
            if not e["duration_secs"]:
                e["duration_secs"] = duration
            e["extra"] = {"call_mode": "manual"}
        _arm_finalize_timer(self.call_id)
        threading.Thread(target=self._transcribe_and_finalize, daemon=True,
                         name=f"ManualSTT-{self.call_id}").start()

    def _transcribe_and_finalize(self):
        transcript = []
        try:
            if self.answered:
                items = []
                for role, buf in (("agent", bytes(self.rec_browser)), ("lead", bytes(self.rec_phone))):
                    try:
                        for start, text in deepgram_transcribe_segments(buf, VOICELINK_RATE):
                            items.append((start, role, text))
                    except Exception as ex:
                        log("DIALER", f"call {self.call_id}: STT failed for {role}: {ex}")
                items.sort(key=lambda x: x[0])
                base = self.call_started_at or time.time()
                transcript = [{"role": r, "text": t, "ts": base + s} for s, r, t in items]
        finally:
            self.rec_browser = bytearray()
            self.rec_phone = bytearray()
            with _call_results_lock:
                e = CALL_RESULTS.setdefault(self.call_id, _new_call_result())
                e["transcript"] = transcript
            _finalize_call(self.call_id)


def _get_manual_bridge(call_id, cfg):
    with _manual_lock:
        b = MANUAL_BRIDGES.get(call_id)
        if b is None:
            b = ManualDialerBridge(call_id, cfg)
            MANUAL_BRIDGES[call_id] = b
        return b


def _run_voicelink_manual(ws, call_id, cfg):
    """VoiceLink side of a manual call (the lead picked up)."""
    bridge = _get_manual_bridge(call_id, cfg)
    VOICELINK_BRIDGES[call_id] = bridge          # so call.ended webhook can close it
    bridge.attach_voicelink(ws)
    log("VOICELINK", f"call {call_id}: manual dialer mode")
    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            if isinstance(msg, (bytes, bytearray)):
                continue
            try:
                data = json.loads(msg)
            except Exception:
                continue
            event = data.get("event")
            if event == "start":
                bridge.on_voicelink_start(data)
            elif event == "media":
                media = data.get("media", {}) or {}
                if media.get("track", "inbound") != "inbound":
                    continue
                payload_b64 = media.get("payload")
                if payload_b64:
                    try:
                        bridge.feed_phone_audio(base64.b64decode(payload_b64))
                    except Exception:
                        continue
            elif event == "stop":
                log("VOICELINK", f"call {call_id}: stop event")
                break
    except Exception as e:
        log("VOICELINK", f"manual ws loop error: {e}")
    finally:
        bridge.finish("customer_hangup")         # no-op if the browser already ended it
        bridge.report()
        VOICELINK_BRIDGES.pop(call_id, None)
        with _pending_calls_lock:
            PENDING_CALLS.pop(call_id, None)
        log("VOICELINK", f"Manual call {call_id} disconnected.")


@sock.route("/ws/dialer/<call_id>")
def dialer_browser_ws(ws, call_id):
    """The dashboard browser connects here (token comes from Pravah's /api/dialer/start)."""
    import hmac as _hmac
    token = (request.args.get("token") or "").strip()
    with _pending_calls_lock:
        cfg = PENDING_CALLS.get(call_id)
    expected = ((cfg or {}).get("browser_token") or "")
    if (not cfg or cfg.get("call_mode") != "manual" or not token
            or not _hmac.compare_digest(token.encode(), expected.encode())):
        try:
            ws.send(json.dumps({"type": "error", "message": "Call not found or not authorised"}))
        except Exception:
            pass
        return

    bridge = _get_manual_bridge(call_id, cfg)
    if bridge.browser_ws is not None or bridge.closed.is_set():
        try:
            ws.send(json.dumps({"type": "error", "message": "This call is already connected elsewhere or has ended"}))
        except Exception:
            pass
        return

    bridge.attach_browser(ws)
    bridge.start_ring_watchdog()
    log("DIALER", f"browser connected for call {call_id}")

    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            if isinstance(msg, (bytes, bytearray)):
                bridge.feed_browser_audio(bytes(msg))
                continue
            try:
                payload = json.loads(msg)
            except Exception:
                continue
            mtype = payload.get("type")
            if mtype == "hello":
                try:
                    bridge.set_browser_rate(int(payload.get("sample_rate") or 0))
                except (TypeError, ValueError):
                    pass
            elif mtype == "hangup":
                bridge.finish("browser_hangup")
                break
            elif mtype == "ping":
                bridge._send_browser_json({"type": "pong"})
    except Exception as e:
        log("DIALER", f"browser ws loop error: {e}")
    finally:
        bridge.finish("browser_disconnected")    # no-op if already finished
        log("DIALER", f"browser disconnected for call {call_id}")

#did update

@sock.route("/ws/voicelink/<call_id>")
def voicelink_ws(ws, call_id):
    """VoiceLink connects here once the customer answers. Audio is now
    bridged into a LiveKit room instead of going through EvaSession's
    raw Deepgram pipeline - see livekit_bridge.py."""
    with _pending_calls_lock:
        cfg = PENDING_CALLS.get(call_id)
    if not cfg:
        if re.fullmatch(r"[0-9a-f]{24}", call_id or ""):
            log("VOICELINK", f"No pending config for outbound call_id={call_id} (expired?), closing.")
            return
        # not one of our outbound call ids -> the customer dialed our DID
        _run_voicelink_inbound(ws, call_id)
        return

    # IVR calls: play recording -> wait -> STT -> hang up (no LiveKit)
    if cfg.get("call_mode") == "ivr":
        _run_voicelink_ivr(ws, call_id, cfg)
        return

    # Manual dialer calls: bridge VoiceLink audio <-> the user's browser (no AI)
    if cfg.get("call_mode") == "manual":
        _run_voicelink_manual(ws, call_id, cfg)
        return

    bridge = livekit_bridge.VoiceLinkBridge(
        call_id=call_id, ws=ws, agent_cfg=cfg["agent"], lead=cfg["lead"],
        meeting=cfg.get("meeting"), callback_url=cfg["callback_url"],
    )
    try:
        bridge.start()
    except Exception as e:
        log("VOICELINK", f"bridge failed to start for {call_id}: {type(e).__name__}: {e!r}")
        bridge.close()  # best-effort: clean up anything that DID connect
        with _pending_calls_lock:
            PENDING_CALLS.pop(call_id, None)
        return

    VOICELINK_BRIDGES[call_id] = bridge
    log("VOICELINK", f"Outbound call {call_id}: bridged into LiveKit room {bridge.room_name}")

    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            if isinstance(msg, (bytes, bytearray)):
                log("VOICELINK", f"call {call_id}: unexpected binary frame, ignoring")
                continue
            try:
                data = json.loads(msg)
            except Exception:
                continue

            event = data.get("event")
            if event == "connected":
                log("VOICELINK", f"call {call_id}: connected")
            elif event == "start":
                start_obj = data.get("start") or {}
                bridge.stream_sid = (start_obj.get("stream_sid") or start_obj.get("streamSid")
                                     or data.get("stream_sid") or data.get("streamSid"))
                bridge.call_sid = (start_obj.get("call_sid") or start_obj.get("callSid")
                                   or data.get("call_sid") or data.get("callSid"))
                bridge.set_codec_from_start(data)
                bridge.started.set()      # sender greenlet may now push audio
                log("VOICELINK", f"call {call_id}: start event payload={str(data)[:1500]}")
            elif event == "media":
                media = data.get("media", {}) or {}
                if media.get("track", "inbound") != "inbound":
                    continue
                payload_b64 = media.get("payload")
                if payload_b64:
                    try:
                        audio = base64.b64decode(payload_b64)
                    except Exception:
                        continue
                    bridge.feed_alaw(audio)
            elif event == "mark":
                pass
            elif event == "stop":
                log("VOICELINK", f"call {call_id}: stop event")
                break
            else:
                log("VOICELINK", f"call {call_id}: unhandled event {event!r}")
    except Exception as e:
        log("VOICELINK", f"ws loop error: {e}")
    finally:
        bridge.close()
        VOICELINK_BRIDGES.pop(call_id, None)
        with _pending_calls_lock:
            PENDING_CALLS.pop(call_id, None)
        # Keep callback_url alive so transcript (agent.py) + recording URL
        # (call.completed webhook) can still be delivered together afterwards.
        with _call_results_lock:
            _e = CALL_RESULTS.setdefault(call_id, _new_call_result())
            _e["callback_url"] = _e["callback_url"] or cfg.get("callback_url")
        _arm_finalize_timer(call_id)
        log("VOICELINK", f"Outbound call {call_id} disconnected.")

if __name__ == "__main__":
    missing = check_missing_keys()
    if missing:
        print(f"Missing keys in .env: {', '.join(missing)}")
        sys.exit(1)
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
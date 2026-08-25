"""
Eva - Web backend

Flask app that serves:
  - "/"            a landing page (templates/landing.html) with Eva's demo widget
  - "/widget.js"   an embeddable <script> you can paste into ANY html page,
                    which injects a floating "Talk to Eva" widget bottom-right
  - "/ws/eva"      a WebSocket endpoint that runs the real voice pipeline:
                    browser mic (PCM16/16kHz) -> Deepgram STT (streaming)
                    -> Mistral LLM (streaming) -> Sarvam TTS (streaming)
                    -> PCM16/22050Hz audio frames sent back to the browser
  - "/api/calls"   called by PravaahAI to place an outbound campaign call
                    (Eva calls the lead, runs the same voice pipeline over
                    Twilio Media Streams, then POSTs the transcript back to
                    PravaahAI's callback_url when the call ends)

Each WebSocket connection gets its own EvaSession with its own Deepgram
connection + history, so multiple visitors/calls can run at once.

Setup:
    pip install -r requirements.txt

    .env:
        DEEPGRAM_API_KEY=...
        MISTRAL_API_KEY=...
        SARVAM_API_KEY=...
        PORT=8420                    (optional, defaults to 8420)
        EVA_SPEAKER=priya            (optional)
        EVA_API_SECRET=...           (shared secret with PravaahAI)
        PUBLIC_BASE_URL=https://your-eva-tunnel.ngrok-free.app
        EVA_RESPONSE_PAUSE_SECS=3.5  (optional, natural pause before Eva replies)

Run (dev):
    python app.py

Run (prod, inside Docker):
    gunicorn -k gevent -w 1 -b 0.0.0.0:8420 app:app
"""

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
from sarvamai import SarvamAI
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import VoiceResponse, Connect

import websocket as vanisetu_ws_lib   # pip install websocket-client

load_dotenv()

# ---------------- Config ----------------
PORT = int(os.environ.get("PORT", 8420))

MIC_RATE = 16000               # PCM16 the browser sends to us
TTS_SAMPLE_RATE = 22050        # PCM16 we send back to the browser

SENTENCE_END_RE = re.compile(r"([.!?।\n])")
SPEAKABLE_RE = re.compile(r"[A-Za-z0-9\u0900-\u097F]")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
BOOK_MEETING_RE = re.compile(r"BOOK_MEETING:\s*(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})")

MISTRAL_MODEL = "mistral-small-latest"
MISTRAL_REASONING_EFFORT="none"
SARVAM_TTS_MODEL = "bulbul:v3"
DEFAULT_SPEAKER = os.environ.get("EVA_SPEAKER", "priya")
# Speech playback speed. bulbul:v3 accepts 0.5 (slower) to 2.0 (faster);
# 1.0 is normal pace. A touch above 1.0 reads as natural-but-brisk instead
# of sluggish, without tipping into sounding rushed.
TTS_PACE = float(os.environ.get("EVA_TTS_PACE", 1.12))

MAX_HISTORY_MESSAGES = 16

# How long Eva waits, after the user goes quiet, before she actually replies.
# Mimics a natural human turn-taking gap instead of jumping in instantly.
RESPONSE_DELAY_SECS = float(os.environ.get("EVA_RESPONSE_PAUSE_SECS", 0.7))
# Small random jitter added on top of the base pause so Eva doesn't reply
# on the exact same beat every time - a perfectly fixed delay is what
# makes a voice bot feel mechanical.
RESPONSE_DELAY_JITTER_SECS = float(os.environ.get("EVA_RESPONSE_PAUSE_JITTER", 0.25))
# Short acknowledgements ("yes", "okay", "no thanks") get a shorter pause -
# humans reply to quick confirmations faster than to longer statements.
SHORT_UTTERANCE_MAX_WORDS = int(os.environ.get("EVA_SHORT_UTTERANCE_MAX_WORDS", 3))
SHORT_UTTERANCE_DELAY_SECS = float(os.environ.get("EVA_SHORT_UTTERANCE_PAUSE_SECS", 0.35))

BARGE_IN_GRACE_SECS = float(os.environ.get("EVA_BARGE_IN_GRACE_SECS", 1.0))
# VAD (SpeechStarted) fires on ANY audio energy spike - coughs, breathing,
# mic bumps - not just real speech. We no longer interrupt Eva on VAD
# alone; we wait to see if Deepgram actually transcribes real words within
# this window before treating it as a genuine barge-in.
BARGE_IN_CONFIRM_MIN_CHARS = int(os.environ.get("EVA_BARGE_IN_MIN_CHARS", 2))
BARGE_IN_CONFIRM_TIMEOUT_SECS = float(os.environ.get("EVA_BARGE_IN_CONFIRM_TIMEOUT", 0.6))

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

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")

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


def log(stage: str, msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{stage}] {msg}", flush=True)


def is_speakable(text: str) -> bool:
    return bool(SPEAKABLE_RE.search(text))


def detect_lang(text: str) -> str:
    return "hi" if DEVANAGARI_RE.search(text) else "en"


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


def render_call_vars(text: str, lead: dict) -> str:
    """Replace {{name}}, {{business_name}}, etc. with lead field values
    (same merge-tag convention as PravaahAI's templates)."""
    for key in ("name", "business_name", "email", "phone", "website", "description"):
        text = text.replace("{{%s}}" % key, str((lead or {}).get(key, "") or ""))
    return text


app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/static")
sock = Sock(app)


# ============================================================
# One EvaSession per WebSocket connection
# ============================================================
class EvaSession:
    def __init__(self, ws, speaker: str = DEFAULT_SPEAKER, mode: str = "browser",
                 call_id: str = None, agent: dict = None, lead: dict = None,
                 callback_url: str = None, meeting: dict = None,
                 transport: str = "twilio", vanisetu_session_id: int = None):
        agent = agent or {}
        lead = lead or {}
        self.meeting = meeting or {}

        self.ws = ws
        self.speaker = (agent.get("speaker") or speaker or DEFAULT_SPEAKER)
        self.mode = mode                # "browser" or "phone"
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
        self.forced_language = forced_lang if forced_lang in ("en", "hi") else None
        self.max_duration_secs = int(agent.get("max_duration_secs") or 0) or None
        self.min_duration_secs = int(agent.get("min_duration_secs") or 0) or None

        self.user_text_q: "queue.Queue[str]" = queue.Queue()
        self.sentence_q: "queue.Queue[tuple]" = queue.Queue()

        self.history = []

        custom_prompt = (agent.get("system_prompt") or "").strip()
        base_prompt = custom_prompt or (
            "You are Eva, a helpful, concise, warm voice assistant. "
            "Keep replies short and conversational (1-3 sentences) since they "
            "will be spoken aloud."
        )
        if lead:
            base_prompt += (
                f"\n\nYou are speaking with {lead.get('name', 'the lead')} from "
                f"{lead.get('business_name', 'their business')}. Use their name naturally, don't overuse it."
            )
        if self.forced_language == "hi":
            base_prompt += "\nAlways reply in English or hindi as per user speaking written in English script, mixed lightly with English words."
        elif self.forced_language == "en":
            base_prompt += "\nAlways reply in English only."
        else:
            base_prompt += (
                "\nLanguage rule: default to English. Only switch to Hinglish "
                "(Hindi written in English script, mixed lightly with English words) "
                "if the user is clearly speaking Hindi (Devanagari script). "
                "If their message is in English, unclear, or mixed, reply in English. "
                "Never default to Hindi on your own."
            )
        base_prompt += "\nNever reply using only emojis or symbols with no words."
        # Applies unconditionally - even on top of an owner's own custom
        # system_prompt above - since this is a live voice call, not a chat
        # window: a long reply just sits there as dead air while Eva is
        # still talking, and invites the lead to talk over her.
        base_prompt += (
            "\n\nSPEAKING LENGTH RULE (always follow, no exceptions): this "
            "is a live phone/voice conversation. Normally answer in ONE "
            "short sentence. At most 2-3 short sentences for a normal "
            "question. Only go longer than that if the lead explicitly asks "
            "for a real explanation, a walkthrough, or a list of things - "
            "and even then stay as brief as possible while still being "
            "correct. Never pad with extra detail, filler, or repeating "
            "back what they said."
        )

        if self.meeting:
            base_prompt += (
                "\n\nYou can book a meeting for this lead. Meetings are "
                f"{self.meeting.get('duration_minutes', 30)} minutes long. "
                f"Available windows: {self.meeting.get('availability_text', '')}. "
                "The lead's name and phone number are already known to you — never ask for "
                "them again, only ask for their preferred meeting date and time. "
                "Once they confirm one specific date and time, output EXACTLY one line in "
                "this format and nothing else on that line: "
                "BOOK_MEETING: YYYY-MM-DD HH:MM (24-hour clock, UTC). "
                "Do not say this line out loud or explain it to the lead — it is processed "
                "automatically and you will be told right after whether it was confirmed, "
                "so you can relay that to them."
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

        self.sarvam = SarvamAI(api_subscription_key=SARVAM_API_KEY)

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
            encoding, sample_rate = "mulaw", PHONE_RATE
        else:
            encoding, sample_rate = "linear16", MIC_RATE

        options = LiveOptions(
            model="nova-3",
            language="multi",
            smart_format=True,
            interim_results=True,
            utterance_end_ms="1000",
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
            self.dg_connection.send(data)
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
        payload = {"model": MISTRAL_MODEL, "messages": messages, "stream": True}
        # mistral-medium-3.5 accepts reasoning_effort "none" or "high" -
        # kept at "none" by default so it doesn't add a thinking pass (and
        # the latency that comes with it) before Eva starts speaking on a
        # live call. Harmless to send even if a non-reasoning model is
        # configured, since Mistral just ignores it in that case.
        if MISTRAL_REASONING_EFFORT:
            payload["reasoning_effort"] = MISTRAL_REASONING_EFFORT
        headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
        with client.stream("POST", "https://api.mistral.ai/v1/chat/completions",
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
                # Must match the script rule already set in the system
                # prompt (Hinglish = Hindi words in Latin/English script,
                # NOT Devanagari). The old text here said "Hindi
                # (Devanagari)", which directly contradicted that - the
                # model would inconsistently follow one instruction or the
                # other, so replies sometimes came back in Devanagari
                # instead of Hinglish.
                lang_note = {
                    "role": "system",
                    "content": (
                        "(Reply in Hinglish - Hindi words written in "
                        "English/Latin script, mixed lightly with English - "
                        "only.)"
                        if user_lang == "hi"
                        else "(Reply in English only.)"
                    )
                }
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
    def _tts_loop(self):
        while not self.stop_event.is_set():
            try:
                sentence, lang = self.sentence_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if not is_speakable(sentence):
                continue

            # Dropped mid-flight by a barge-in that happened between this
            # sentence being queued and us picking it up - skip it.
            if self.interrupt_flag.is_set():
                continue

            target_language_code = "hi-IN" if lang == "hi" else "en-IN"
            self._send_json({"type": "status", "state": "speaking"})
            self.eva_speaking.set()

            if self.mode == "phone" and self.transport == "vanisetu" and self.vanisetu_session_id is not None:
                vanisetu_client.send_command(self.vanisetu_session_id, {"command": "START_MEDIA_BUFFERING"})

            if self.mode == "phone":
                tts_codec, tts_rate = "mulaw", (VANISETU_RATE if self.transport == "vanisetu" else PHONE_RATE)
            else:
                tts_codec, tts_rate = "linear16", TTS_SAMPLE_RATE

            leftover = b""
            try:
                # NOTE: the streaming endpoint's language param is named
                # "language_code" (NOT "target_language_code" - that name is
                # only valid on the non-streaming .convert() call). Passing
                # the wrong name throws a TypeError and produces zero audio.
                # `pace` is bulbul:v3's speed knob (0.5-2.0, 1.0 = normal).
                for chunk in self.sarvam.text_to_speech.convert_stream(
                    text=sentence,
                    language_code=target_language_code,
                    speaker=self.speaker,
                    model=SARVAM_TTS_MODEL,
                    output_audio_codec=tts_codec,
                    speech_sample_rate=tts_rate,
                    pace=TTS_PACE,
                ):
                    if self.interrupt_flag.is_set():
                        # user started talking mid-sentence - stop right here
                        break
                    if not chunk:
                        continue
                    if self.mode == "phone":
                        # mu-law is 1 byte/sample - no alignment needed, send as-is
                        self._send_audio(chunk)
                        continue
                    # linear16 is 2 bytes/sample - keep frames byte-aligned
                    data = leftover + chunk
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

            # Let the client know this sentence's audio has fully been sent.
            if self.sentence_q.empty() and not self.interrupt_flag.is_set():
                self._send_json({"type": "status", "state": "listening"})
                # Eva has genuinely gone quiet with nothing queued - the
                # turn is over. Next time she speaks it's a fresh turn and
                # gets a fresh grace window (see _enqueue_sentence).
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
            cfg["agent"].get("opening_line") or "Hi, do you have a quick minute?", cfg["lead"],
        )
        opening_lang = "hi" if cfg["agent"].get("language") == "hi" else "en"
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


vanisetu_client = VaniSetuClient()
vanisetu_client.start()


# ============================================================
# Routes
# ============================================================
@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/widget.js")
def widget_js():
    return send_from_directory("static", "widget.js", mimetype="application/javascript")


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

  var css = document.createElement('style');
  css.textContent = `
    #eva-w-bubble{position:fixed;bottom:22px;right:22px;width:62px;height:62px;border-radius:50%;
      background:${color};box-shadow:0 6px 20px rgba(0,0,0,.25);display:flex;align-items:center;
      justify-content:center;cursor:pointer;z-index:999999;transition:transform .2s;}
    #eva-w-bubble:hover{transform:scale(1.06);}
    #eva-w-bubble svg{width:26px;height:26px;fill:#fff;}
    #eva-w-bubble.eva-live{animation:eva-pulse 1.4s infinite;}
    @keyframes eva-pulse{0%{box-shadow:0 0 0 0 ${color}66;}70%{box-shadow:0 0 0 16px ${color}00;}100%{box-shadow:0 0 0 0 ${color}00;}}
    #eva-w-panel{position:fixed;bottom:96px;right:22px;width:340px;max-width:92vw;height:480px;max-height:76vh;
      background:#fff;border-radius:16px;box-shadow:0 12px 40px rgba(0,0,0,.22);display:none;flex-direction:column;
      overflow:hidden;z-index:999999;font-family:-apple-system,Segoe UI,Roboto,sans-serif;}
    #eva-w-panel.open{display:flex;}
    #eva-w-head{background:${color};color:#fff;padding:14px 16px;font-size:14px;font-weight:600;}
    #eva-w-body{flex:1;padding:16px;overflow:auto;font-size:13px;color:#222;}
    #eva-w-form input{width:100%;box-sizing:border-box;margin-bottom:8px;padding:10px 12px;border:1px solid #ddd;
      border-radius:8px;font-size:13px;}
    #eva-w-form button{width:100%;padding:10px;border:none;border-radius:8px;background:${color};color:#fff;
      font-weight:600;cursor:pointer;}
    #eva-w-status{text-align:center;color:#888;font-size:12px;margin-top:10px;}
    #eva-w-mic{width:74px;height:74px;border-radius:50%;background:${color};margin:20px auto;display:flex;
      align-items:center;justify-content:center;cursor:pointer;animation:eva-mic-pulse 1.8s infinite;}
    @keyframes eva-mic-pulse{0%{box-shadow:0 0 0 0 ${color}55;}70%{box-shadow:0 0 0 14px ${color}00;}100%{box-shadow:0 0 0 0 ${color}00;}}
    #eva-w-mic svg{width:30px;height:30px;fill:#fff;}
    #eva-w-transcript{font-size:12.5px;line-height:1.6;}
    #eva-w-transcript .u{color:#111;font-weight:600;}
    #eva-w-transcript .a{color:${color};font-weight:600;}
    @media(max-width:520px){
      #eva-w-bubble{left:50%;right:auto;bottom:18px;transform:translateX(-50%);}
      #eva-w-bubble:hover{transform:translateX(-50%) scale(1.06);}
      #eva-w-panel{left:0;right:0;bottom:0;transform:none;width:100%;height:100%;
        max-height:100%;border-radius:0;max-width:100%;}
    }
  `;
  document.head.appendChild(css);

  var bubble = document.createElement('div'); bubble.id = 'eva-w-bubble';
  bubble.innerHTML = '<svg viewBox="0 0 24 24"><path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2z"/></svg>';
  var panel = document.createElement('div'); panel.id = 'eva-w-panel';
  panel.innerHTML = `
    <div id="eva-w-head">Talk to us</div>
    <div id="eva-w-body">
      <div id="eva-w-form">
        <input id="eva-w-name" placeholder="Your name">
        <input id="eva-w-phone" placeholder="Phone number">
        <input id="eva-w-email" placeholder="Email (optional)">
        <button id="eva-w-start">Start</button>
      </div>
      <div id="eva-w-call" style="display:none;text-align:center;">
        <div id="eva-w-mic"><svg viewBox="0 0 24 24"><path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2z"/></svg></div>
        <div id="eva-w-status">Connecting…</div>
        <div id="eva-w-transcript"></div>
      </div>
    </div>`;
  document.body.appendChild(bubble); document.body.appendChild(panel);

  var ws, audioCtx, mic, processor, playHead = 0;
  bubble.onclick = function(){ panel.classList.toggle('open'); };

  document.getElementById('eva-w-start').onclick = function(){
    var name = document.getElementById('eva-w-name').value.trim();
    var phone = document.getElementById('eva-w-phone').value.trim();
    var email = document.getElementById('eva-w-email').value.trim();
    if(!name || !phone){ alert('Please share your name and phone number'); return; }
    document.getElementById('eva-w-form').style.display = 'none';
    document.getElementById('eva-w-call').style.display = 'block';
    startSession(name, phone, email);
  };

  function startSession(name, phone, email){
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    ws.onopen = function(){
      ws.send(JSON.stringify({type:'lead_info', name:name, phone:phone, email:email}));
      startMic();
      bubble.classList.add('eva-live');
    };
    ws.onmessage = function(ev){
      if(typeof ev.data === 'string'){
        var msg = JSON.parse(ev.data);
        if(msg.type === 'status'){ document.getElementById('eva-w-status').textContent =
          msg.state === 'speaking' ? 'Eva is speaking…' : (msg.state === 'thinking' ? 'Thinking…' : 'Listening…'); }
        if(msg.type === 'assistant_delta'){ appendTranscript('a', msg.text, true); }
        if(msg.type === 'assistant_done'){ appendTranscript('a', '', false); }
        if(msg.type === 'user_transcript' && msg.final){ appendTranscript('u', msg.text, false); }
        if(msg.type === 'error'){ document.getElementById('eva-w-status').textContent = msg.message; }
      } else {
        playAudio(ev.data);
      }
    };
    ws.onclose = function(){ bubble.classList.remove('eva-live'); document.getElementById('eva-w-status').textContent = 'Call ended'; stopMic(); };
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
})();
"""
    return js, 200, {"Content-Type": "application/javascript"}


@sock.route("/ws/widget/<public_id>")
def widget_ws(ws, public_id):
    """A visitor on some customer's website connects here. We look up which
    agent + owner this public_id belongs to, run the normal Eva voice
    pipeline, capture the lead, and bill Eva minutes on close."""
    config, err = fetch_widget_config(public_id)
    if err or not config:
        try:
            ws.send(json.dumps({"type": "error", "message": err or "Widget unavailable"}))
        except Exception:
            pass
        return

    owner_id = config["owner_id"]
    widget_id = config["widget_id"]
    agent = config.get("agent", {})
    require_lead_first = config.get("require_lead_before_chat", True)

    session = EvaSession(ws, mode="browser", agent=agent, lead={})
    session.call_started_at = time.time()  # reused purely for widget duration billing
    if not session.start():
        return

    if require_lead_first:
        session._send_json({"type": "status", "state": "awaiting_lead_info"})
    else:
        session._send_json({"type": "ready"})
        session.speak(agent.get("opening_line") or "Hi! How can I help you today?", "en")

    lead_id_holder = {"lead_id": None}
    log("MAIN", f"Widget visitor connected: {public_id}")

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
                session._send_json({"type": "ready"})
                opening = render_call_vars(agent.get("opening_line") or "Hi {{name}}, how can I help you today?", session.lead)
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
    missing = [n for n, v in [
        ("DEEPGRAM_API_KEY", DEEPGRAM_API_KEY),
        ("MISTRAL_API_KEY", MISTRAL_API_KEY),
        ("SARVAM_API_KEY", SARVAM_API_KEY),
    ] if not v]
    return jsonify({"status": "ok" if not missing else "missing_keys", "missing": missing})


@sock.route("/ws/eva")
def eva_ws(ws):
    missing = [n for n, v in [
        ("DEEPGRAM_API_KEY", DEEPGRAM_API_KEY),
        ("MISTRAL_API_KEY", MISTRAL_API_KEY),
        ("SARVAM_API_KEY", SARVAM_API_KEY),
    ] if not v]
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
    missing = [n for n, v in [
        ("DEEPGRAM_API_KEY", DEEPGRAM_API_KEY),
        ("MISTRAL_API_KEY", MISTRAL_API_KEY),
        ("SARVAM_API_KEY", SARVAM_API_KEY),
    ] if not v]
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
                session.speak("Hi, this is Eva. How can I help you today?", "en")
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

    missing = [n for n, v in [
        ("DEEPGRAM_API_KEY", DEEPGRAM_API_KEY),
        ("MISTRAL_API_KEY", MISTRAL_API_KEY),
        ("SARVAM_API_KEY", SARVAM_API_KEY),
    ] if not v]
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

    missing = [n for n, v in [
        ("DEEPGRAM_API_KEY", DEEPGRAM_API_KEY),
        ("MISTRAL_API_KEY", MISTRAL_API_KEY),
        ("SARVAM_API_KEY", SARVAM_API_KEY),
    ] if not v]
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
                    cfg["agent"].get("opening_line") or "Hi, do you have a quick minute?",
                    cfg["lead"],
                )
                opening_lang = "hi" if cfg["agent"].get("language") == "hi" else "en"
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

    missing = [n for n, v in [
        ("DEEPGRAM_API_KEY", DEEPGRAM_API_KEY),
        ("MISTRAL_API_KEY", MISTRAL_API_KEY),
        ("SARVAM_API_KEY", SARVAM_API_KEY),
    ] if not v]
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

if __name__ == "__main__":
    missing = [n for n, v in [
        ("DEEPGRAM_API_KEY", DEEPGRAM_API_KEY),
        ("MISTRAL_API_KEY", MISTRAL_API_KEY),
        ("SARVAM_API_KEY", SARVAM_API_KEY),
    ] if not v]
    if missing:
        print(f"Missing keys in .env: {', '.join(missing)}")
        sys.exit(1)
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
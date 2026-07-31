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

Each WebSocket connection gets its own EvaSession with its own Deepgram
connection + history, so multiple visitors can talk to Eva at once.

Setup:
    pip install -r requirements.txt

    .env:
        DEEPGRAM_API_KEY=...
        MISTRAL_API_KEY=...
        SARVAM_API_KEY=...
        PORT=8420                 (optional, defaults to 8420)
        EVA_SPEAKER=priya         (optional)

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
import base64
import queue
import threading

import httpx
from flask import Flask, render_template, send_from_directory, jsonify
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

load_dotenv()

# ---------------- Config ----------------
PORT = int(os.environ.get("PORT", 8420))

MIC_RATE = 16000               # PCM16 the browser sends to us
TTS_SAMPLE_RATE = 22050        # PCM16 we send back to the browser

SENTENCE_END_RE = re.compile(r"([.!?।\n])")
SPEAKABLE_RE = re.compile(r"[A-Za-z0-9\u0900-\u097F]")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

MISTRAL_MODEL = "mistral-small-latest"
SARVAM_TTS_MODEL = "bulbul:v3"
DEFAULT_SPEAKER = os.environ.get("EVA_SPEAKER", "priya")

MAX_HISTORY_MESSAGES = 16

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


def log(stage: str, msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{stage}] {msg}", flush=True)


def is_speakable(text: str) -> bool:
    return bool(SPEAKABLE_RE.search(text))


def detect_lang(text: str) -> str:
    return "hi" if DEVANAGARI_RE.search(text) else "en"


app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/static")
sock = Sock(app)


# ============================================================
# One EvaSession per WebSocket connection
# ============================================================
class EvaSession:
    def __init__(self, ws, speaker: str = DEFAULT_SPEAKER, mode: str = "browser"):
        self.ws = ws
        self.speaker = speaker
        self.mode = mode                # "browser" or "phone"
        self.stream_sid = None          # set once Twilio's "start" event arrives (phone mode only)
        self.ws_lock = threading.Lock()
        self.stop_event = threading.Event()

        self.user_text_q: "queue.Queue[str]" = queue.Queue()
        self.sentence_q: "queue.Queue[tuple]" = queue.Queue()

        self.history = []
        self.system_prompt = {
            "role": "system", "content": (
                "You are Eva, a helpful, concise, warm voice assistant. "
                "Keep replies short and conversational (1-3 sentences) since they "
                "will be spoken aloud. "
                "Language rule: always reply in the SAME language the user just used. "
                "If they spoke English, reply only in English. "
                "If they spoke Hindi (Devanagari script), reply only in Hinglish "
                "written in english script - mixed with English words little bit. "
                "Never reply using only emojis or symbols with no words."
            )
        }

        config = DeepgramClientOptions(options={"keepalive": "true"})
        self.deepgram = DeepgramClient(DEEPGRAM_API_KEY, config)
        self.dg_connection = self.deepgram.listen.websocket.v("1")
        self.dg_connection.on(LiveTranscriptionEvents.Open, self._dg_open)
        self.dg_connection.on(LiveTranscriptionEvents.Transcript, self._dg_transcript)
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

    def _send_audio(self, audio_bytes: bytes):
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

    # ---------- Deepgram callbacks ----------
    def _dg_open(self, *_a, **_k):
        log("STT", "Deepgram connection open.")

    def _dg_transcript(self, *_a, result=None, **_k):
        if result is None:
            return
        transcript = result.channel.alternatives[0].transcript
        if not transcript:
            return
        if result.is_final:
            log("STT", f"Final transcript: {transcript}")
            self._send_json({"type": "user_transcript", "text": transcript, "final": True})
            self.user_text_q.put(transcript)
        else:
            self._send_json({"type": "user_transcript", "text": transcript, "final": False})

    def _dg_error(self, *_a, error=None, **_k):
        log("STT", f"ERROR: {error}")

    def _dg_close(self, *_a, **_k):
        log("STT", "Deepgram connection closed.")

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
        return True

    def feed_audio(self, data: bytes):
        try:
            self.dg_connection.send(data)
        except Exception as e:
            log("STT", f"send error: {e}")

    def feed_text(self, text: str):
        """Allow a typed message to skip STT and go straight to the LLM."""
        self._send_json({"type": "user_transcript", "text": text, "final": True})
        self.user_text_q.put(text)

    def speak(self, text: str, lang: str = "en"):
        """Queue a line straight to TTS, bypassing the LLM (e.g. an opening greeting)."""
        self.sentence_q.put((text, lang))

    def close(self):
        self.stop_event.set()
        try:
            self.dg_connection.finish()
        except Exception:
            pass

    # ---------- LLM loop ----------
    def _trim_history(self):
        if len(self.history) > MAX_HISTORY_MESSAGES:
            self.history = self.history[-MAX_HISTORY_MESSAGES:]

    def _stream_chat(self, client: httpx.Client, messages):
        payload = {"model": MISTRAL_MODEL, "messages": messages, "stream": True}
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

                user_lang = detect_lang(user_text)
                lang_note = {
                    "role": "system",
                    "content": f"(Reply in {'Hindi (Devanagari)' if user_lang == 'hi' else 'English'} only.)"
                }
                self.history.append({"role": "user", "content": user_text})
                self._send_json({"type": "status", "state": "thinking"})

                buffer, full_reply = "", ""
                try:
                    messages = [self.system_prompt] + self.history + [lang_note]
                    for delta in self._stream_chat(client, messages):
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
                        if sentence and is_speakable(sentence):
                            self.sentence_q.put((sentence, user_lang))
                        buffer = remainder
                except Exception as e:
                    log("LLM", f"ERROR: {e}")
                    self._send_json({"type": "error", "message": "Eva had trouble thinking that through."})
                    continue

                tail = buffer.strip()
                if tail and is_speakable(tail):
                    self.sentence_q.put((tail, user_lang))

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

            target_language_code = "hi-IN" if lang == "hi" else "en-IN"
            self._send_json({"type": "status", "state": "speaking"})

            if self.mode == "phone":
                tts_codec, tts_rate = "mulaw", PHONE_RATE
            else:
                tts_codec, tts_rate = "linear16", TTS_SAMPLE_RATE

            leftover = b""
            try:
                for chunk in self.sarvam.text_to_speech.convert_stream(
                    text=sentence,
                    target_language_code=target_language_code,
                    speaker=self.speaker,
                    model=SARVAM_TTS_MODEL,
                    output_audio_codec=tts_codec,
                    speech_sample_rate=tts_rate,
                ):
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

            # Let the client know this sentence's audio has fully been sent.
            if self.sentence_q.empty():
                self._send_json({"type": "status", "state": "listening"})


# ============================================================
# Routes
# ============================================================
@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/widget.js")
def widget_js():
    return send_from_directory("static", "widget.js", mimetype="application/javascript")


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
    """Places an outbound call from your Twilio number to MY_PHONE_NUMBER."""
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
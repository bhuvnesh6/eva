"""
Bridges a single VoiceLink phone call into a LiveKit room, so agent.py's
AgentSession pipeline (LiveKit's deepgram.STT, Silero VAD, turn detection,
built-in barge-in) handles the call instead of app.py's hand-rolled
Deepgram/barge-in code.

FIRST PASS: the audio path works end-to-end on paper, but frame sizing,
resampling, and interrupt-clear timing have not been tuned against a real
VoiceLink call yet. Expect to adjust after a live test.
"""
import audioop
import asyncio
import base64
import json
import threading
import uuid

from livekit import rtc, api

from gevent.threadpool import ThreadPool

LIVEKIT_URL = None
LIVEKIT_API_KEY = None
LIVEKIT_API_SECRET = None
AGENT_NAME = None

ROOM_AUDIO_RATE = 8000   # A-law is 8kHz on the wire; publish/subscribe at
                          # the same rate so no manual resampling is needed
                          # on our side (LiveKit resamples internally).


def init(url, api_key, api_secret, agent_name):
    global LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, AGENT_NAME
    LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, AGENT_NAME = url, api_key, api_secret, agent_name


class _BridgeLoop:
    """Runs the bridge's asyncio loop on a dedicated, GENUINE OS thread via
    gevent.threadpool.ThreadPool - NOT via threading.Thread(), which under
    gunicorn's gevent worker (monkey-patched) becomes just another greenlet
    multiplexed onto the SAME single real OS thread as everything else in
    the process, including app.py's existing TTS _AsyncLoopRunner. Sharing
    that thread caused two real problems: (1) an asyncio "two loops on one
    thread" crash when this bridge had its own threading.Thread-based
    loop, and (2) after merging into the shared loop to fix (1), room
    connect() - which goes through livekit-rtc's native FFI layer -
    started timing out because the shared loop was busy with other work
    and couldn't service the FFI callback promptly ("timed out waiting for
    ReadyForRoomEventRequest after ConnectCallback").
    ThreadPool gives us a real, unpatched OS thread: no collision with the
    existing loop, and no risk of FFI-level blocking stalling the rest of
    the app's cooperative scheduling."""
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._pool = ThreadPool(1)
        self._pool.spawn(self._run)

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro, timeout=40):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=timeout)

    def submit(self, coro):
        asyncio.run_coroutine_threadsafe(coro, self.loop)


_bridge_loop = _BridgeLoop()


def _run(coro, timeout=40):
    return _bridge_loop.run(coro, timeout=timeout)


def _submit(coro):
    _bridge_loop.submit(coro)



class VoiceLinkBridge:
    """One instance per VoiceLink call."""

    def __init__(self, call_id: str, ws, agent_cfg: dict, lead: dict,
                 meeting: dict, callback_url: str):
        self.call_id = call_id
        self.ws = ws                      # the flask-sock VoiceLink websocket
        self.ws_lock = threading.Lock()
        self.agent_cfg = agent_cfg
        self.lead = lead
        self.meeting = meeting
        self.callback_url = callback_url
        self.room_name = f"voicelink-{call_id}-{uuid.uuid4().hex[:8]}"

        self.room = None
        self.source = None
        self.local_track = None
        self._closed = threading.Event()

    # ---------------- lifecycle ----------------
    def start(self):
        """Creates the room, dispatches Eva into it, connects a trunk
        participant, and opens an audio source. Blocks until connected.
        40s timeout - room.connect()'s native FFI handshake has been
        observed taking noticeably longer than a plain HTTP round trip."""
        _run(self._async_start(), timeout=40)

    async def _async_start(self):
        lkapi = api.LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        try:
            await lkapi.room.create_room(api.CreateRoomRequest(name=self.room_name))
            print(f"[VOICELINK-BRIDGE] room created: {self.room_name}", flush=True)

            metadata = json.dumps({
                "agent": self.agent_cfg,
                "lead": self.lead,
                "meeting": self.meeting,
                "call_id": self.call_id,
                "callback_url": self.callback_url,
            })
            dispatch_result = await lkapi.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    room=self.room_name, agent_name=AGENT_NAME, metadata=metadata,
                )
            )
            print(f"[VOICELINK-BRIDGE] dispatch created: {dispatch_result!r}", flush=True)
        finally:
            await lkapi.aclose()

        token = (
            api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(f"voicelink-trunk-{self.call_id}")
            .with_name("VoiceLink Trunk")
            .with_grants(api.VideoGrants(room_join=True, room=self.room_name))
            .to_jwt()
        )

        self.room = rtc.Room()
        self.room.on("track_subscribed", self._on_track_subscribed)

        print(f"[VOICELINK-BRIDGE] connecting trunk to {self.room_name}...", flush=True)
        await self.room.connect(LIVEKIT_URL, token, options=rtc.RoomOptions(auto_subscribe=True))
        print(f"[VOICELINK-BRIDGE] trunk connected to {self.room_name}", flush=True)

        self.source = rtc.AudioSource(ROOM_AUDIO_RATE, 1)
        self.local_track = rtc.LocalAudioTrack.create_audio_track("voicelink-in", self.source)
        await self.room.local_participant.publish_track(
            self.local_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        print(f"[VOICELINK-BRIDGE] audio track published for {self.room_name}", flush=True)

    # ---------------- inbound: VoiceLink -> LiveKit room ----------------
    def feed_alaw(self, alaw_bytes: bytes):
        """Call from the VoiceLink websocket thread for every inbound
        'media' frame. Decodes A-law -> linear16 and pushes it into the
        room. Fire-and-forget so the websocket read loop never blocks."""
        if self._closed.is_set() or self.source is None:
            return
        pcm16 = audioop.alaw2lin(alaw_bytes, 2)
        frame = rtc.AudioFrame(
            data=pcm16, sample_rate=ROOM_AUDIO_RATE, num_channels=1,
            samples_per_channel=len(pcm16) // 2,
        )
        _submit(self.source.capture_frame(frame))

    # ---------------- outbound: LiveKit room -> VoiceLink ----------------
    def _on_track_subscribed(self, track, publication, participant):
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        _submit(self._pump_agent_audio(track))

    async def _pump_agent_audio(self, track: rtc.Track):
        stream = rtc.AudioStream(track, sample_rate=ROOM_AUDIO_RATE, num_channels=1)
        async for event in stream:
            if self._closed.is_set():
                break
            alaw = audioop.lin2alaw(bytes(event.frame.data), 2)
            self._send_alaw_to_voicelink(alaw)

    def _send_alaw_to_voicelink(self, alaw_bytes: bytes):
        with self.ws_lock:
            try:
                self.ws.send(json.dumps({
                    "event": "media",
                    "media": {"payload": base64.b64encode(alaw_bytes).decode("ascii")},
                }))
            except Exception:
                pass

    def clear_playback(self):
        """Best-effort barge-in flush signal to VoiceLink."""
        with self.ws_lock:
            try:
                self.ws.send(json.dumps({"event": "clear"}))
            except Exception:
                pass

    # ---------------- teardown ----------------
    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        if self.room:
            _submit(self.room.disconnect())
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
import time
from collections import deque

from livekit import rtc, apiimport, time
from collections import deque

from livekit import rtc, api

import gevent.monkey as _gevent_monkey

# The ACTUAL, pre-monkeypatch threading.Thread class. gunicorn's gevent
# worker monkey-patches threading.Thread into a greenlet that multiplexes
# onto ONE shared real OS thread - using it here caused the earlier
# "Cannot run the event loop while another loop is running" crash when
# combined with app.py's existing TTS loop. Switching to
# gevent.threadpool.ThreadPool got a real OS thread, but ThreadPool's
# worker hub expects each task to RETURN promptly; asyncio's run_forever()
# never returns, so the pool's own internal cross-thread signaling had
# nothing left to wait for and raised "LoopExit: This operation would
# block forever". get_original() sidesteps both: a genuine, permanently-
# running OS thread with no gevent thread-pool lifecycle assumptions
# attached to it.
_RealThread = _gevent_monkey.get_original("threading", "Thread")

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


_task_errors = 0
class _BridgeLoop:
    """Runs the bridge's asyncio loop on a genuine, permanently-running OS
    thread spawned via the real (pre-monkeypatch) threading.Thread - see
    _RealThread comment above for why neither a plain threading.Thread nor
    gevent.threadpool.ThreadPool work for this."""
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = _RealThread(target=self._run, daemon=True, name="LiveKitBridgeLoop")
        self._thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro, timeout=40):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=timeout)

    def submit(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)

        def _log_exc(f):
            global _task_errors
            try:
                exc = f.exception()
            except BaseException:
                return
            if exc is not None and _task_errors < 5:
                _task_errors += _task_errors = 0
                print(f"[VOICELINK-BRIDGE] background task failed: {type(exc).__name__}: {exc!r}", flush=True)

        fut.add_done_callback(_log_exc)


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

        # outbound audio (agent -> phone): filled by the LiveKit loop thread,
        # drained by a gevent greenlet that owns the websocket
        self._out = deque()
        self.started = threading.Event()   # set when VoiceLink's "start" event arrives
        self.stream_sid = None
        self._sent_frames = 0
        self._in_frames = 0

    # ---------------- lifecycle ----------------
    def start(self):
        """Creates the room, connects the trunk participant, publishes the
        audio track, THEN dispatches Eva. Blocks until done."""
        _run(self._async_start(), timeout=40)
        _RealThread(target=self._sender_loop, daemon=True, name="VoiceLinkSender").start()
        _submit(self._heartbeat())
        _submit(self._diagnose_dispatch())

    def _lk_api(self):
        return api.LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)

    async def _async_start(self):
        lkapi = self._lk_api()
        try:
            await lkapi.room.create_room(api.CreateRoomRequest(name=self.room_name))
            print(f"[VOICELINK-BRIDGE] room created: {self.room_name}", flush=True)
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
        self.room.on("participant_connected",
                     lambda p: print(f"[VOICELINK-BRIDGE] participant joined: {p.identity} kind={p.kind}", flush=True))
        self.room.on("participant_disconnected",
                     lambda p: print(f"[VOICELINK-BRIDGE] participant left: {p.identity}", flush=True))

        print(f"[VOICELINK-BRIDGE] connecting trunk to {self.room_name}...", flush=True)
        await self.room.connect(LIVEKIT_URL, token, options=rtc.RoomOptions(auto_subscribe=True))
        print(f"[VOICELINK-BRIDGE] trunk connected to {self.room_name}", flush=True)

        self.source = rtc.AudioSource(ROOM_AUDIO_RATE, 1)
        self.local_track = rtc.LocalAudioTrack.create_audio_track("voicelink-in", self.source)
        await self.room.local_participant.publish_track(
            self.local_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        print(f"[VOICELINK-BRIDGE] audio track published for {self.room_name}", flush=True)

        # Dispatch Eva only now, so she joins a room that already has the caller's mic.
        metadata = json.dumps({
            "agent": self.agent_cfg,
            "lead": self.lead,
            "meeting": self.meeting,
            "call_id": self.call_id,
            "callback_url": self.callback_url,
        })
        lkapi = self._lk_api()
        try:
            dispatch_result = await lkapi.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    room=self.room_name, agent_name=AGENT_NAME, metadata=metadata,
                )
            )
            print(f"[VOICELINK-BRIDGE] dispatch created: id={dispatch_result.id} agent={dispatch_result.agent_name}", flush=True)
        finally:
            await lkapi.aclose()

    async def _diagnose_dispatch(self):
        """4s after dispatch, print whether a job was actually created and who is in the room."""
        try:
            await asyncio.sleep(4)
            who = [p.identity for p in self.room.remote_participants.values()] if self.room else []
            print(f"[VOICELINK-BRIDGE] room participants after 4s: {who}", flush=True)
            lkapi = self._lk_api()
            try:
                dispatches = await lkapi.agent_dispatch.list_dispatch(room_name=self.room_name)
                for d in dispatches:
                    print(f"[VOICELINK-BRIDGE] dispatch {d.id} agent={d.agent_name} jobs={len(d.state.jobs)}", flush=True)
                    for j in d.state.jobs:
                        print(f"[VOICELINK-BRIDGE]   job {j.id} status={j.state.status} "
                              f"error={j.state.error!r} participant={j.state.participant_identity!r}", flush=True)
                    if not d.state.jobs:
                        print("[VOICELINK-BRIDGE]   NO JOB CREATED -> worker did not pick it up (check agent logs)", flush=True)
            finally:
                await lkapi.aclose()
        except Exception as e:
            print(f"[VOICELINK-BRIDGE] diagnose failed: {type(e).__name__}: {e!r}", flush=True)

    # ---------------- inbound: VoiceLink -> LiveKit room ----------------
    
    async def _heartbeat(self):
        """Proves the LiveKit loop thread is alive and shows what it sees."""
        for n in range(1, 8):
            await asyncio.sleep(2)
            if self._closed.is_set():
                return
            who = [p.identity for p in self.room.remote_participants.values()] if self.room else []
            print(f"[VOICELINK-BRIDGE] loop alive t={n*2}s in_frames={self._in_frames} "
                  f"out_queue={len(self._out)} sent={self._sent_frames} remote={who}", flush=True)
    
    def feed_alaw(self, alaw_bytes: bytes):
        """Call from the VoiceLink websocket thread for every inbound
        'media' frame. Decodes A-law -> linear16 and pushes it into the
        room. Fire-and-forget so the websocket read loop never blocks."""
        if self._closed.is_set() or self.source is None:
            return
        self._in_frames += 1
        if self._in_frames == 1:
            print(f"[VOICELINK-BRIDGE] first inbound audio frame ({len(alaw_bytes)} bytes)", flush=True)
        pcm16 = audioop.alaw2lin(alaw_bytes, 2)
        frame = rtc.AudioFrame(
            data=pcm16, sample_rate=ROOM_AUDIO_RATE, num_channels=1,
            samples_per_channel=len(pcm16) // 2,
        )
        _submit(self.source.capture_frame(frame))

    # ---------------- outbound: LiveKit room -> VoiceLink ----------------
    def _on_track_subscribed(self, track, publication, participant):
        print(f"[VOICELINK-BRIDGE] track subscribed: kind={track.kind} from {participant.identity}", flush=True)
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        _submit(self._pump_agent_audio(track))

    async def _pump_agent_audio(self, track: rtc.Track):
        """Runs on the LiveKit loop thread. It must NOT touch the websocket,
        so it only pushes 20ms A-law chunks (160 bytes) into a deque."""
        stream = rtc.AudioStream(track, sample_rate=ROOM_AUDIO_RATE, num_channels=1)
        buf = b""
        first = True
        async for event in stream:
            if self._closed.is_set():
                break
            if first:
                first = False
                print("[VOICELINK-BRIDGE] first agent audio frame received from LiveKit", flush=True)
            buf += audioop.lin2alaw(bytes(event.frame.data), 2)
            while len(buf) >= 160:
                self._out.append(buf[:160])
                buf = buf[160:]

    def _sender_loop(self):
        """Plain OS thread: the only place that writes agent audio to the
        VoiceLink websocket. Waits for VoiceLink's 'start' event first."""
        while not self._closed.is_set():
            if not self.started.is_set() or not self._out:
                time.sleep(0.005)
                continue
            try:
                chunk = self._out.popleft()
            except IndexError:
                continue
            try:
                with self.ws_lock:
                    self.ws.send(json.dumps({
                        "event": "media",
                        "stream_sid": self.stream_sid,
                        "media": {"payload": base64.b64encode(chunk).decode("ascii")},
                    }))
                self._sent_frames += 1
                if self._sent_frames == 1:
                    print("[VOICELINK-BRIDGE] first audio frame SENT to VoiceLink", flush=True)
            except Exception as e:
                print(f"[VOICELINK-BRIDGE] ws send failed: {type(e).__name__}: {e!r}", flush=True)
                time.sleep(0.05)

    def clear_playback(self):
        """Best-effort barge-in flush signal to VoiceLink."""
        self._out.clear()
        with self.ws_lock:
            try:
                self.ws.send(json.dumps({"event": "clear", "stream_sid": self.stream_sid}))
            except Exception:
                pass

    # ---------------- teardown ----------------
    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        if self.room:
            _submit(self.room.disconnect())
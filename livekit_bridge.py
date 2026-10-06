"""
Bridges single VoiceLink phone call into a LiveKit room, so agent.py's
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
import time
import uuid
from collections import deque

from livekit import rtc, api

import gevent.monkey as _gevent_monkey

# The ACTUAL, pre-monkeypat threading.Thread class. gunicorn's gevent
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
                _task_errors += 1
                print(f"[VOICELINK-BRIDGE] background task failed: {type(exc).__name__}: {exc!r}", flush=True)

        fut.add_done_callback(_log_exc)


_bridge_loop = _BridgeLoop()


def _run(coro, timeout=40):
    return _bridge_loop.run(coro, timeout=timeout)


def _submit(coro):
    _bridge_loop.submit(coro)



# ---------------- inbound-call tuning ----------------
PREBUF_MAX_SECS = 15              # max caller audio kept while the agent is still warming up
PREBUF_SPEECH_RMS = 250           # below this = silence / line noise
PREBUF_PRE_ROLL_MS = 300          # keep this much audio before the first spoken word
INBOUND_LIVE_FALLBACK_SECS = 10   # go live anyway if the agent never reports subscribing
GREETING_LEAD_SECS = 0.3          # send greeting audio this far ahead of real time
HANGUP_TAIL_SECS = 0.8            # let the last agent words play before cutting the call


class VoiceLinkBridge:
    """One instance per VoiceLink call (outbound OR inbound)."""

    def __init__(self, call_id: str, ws, agent_cfg: dict, lead: dict,
                 meeting: dict, callback_url: str,
                 inbound: bool = False, greeting_audio: bytes = None,
                 greeting_text: str = "", call_sid: str = None):
        self.call_id = call_id
        self.ws = ws                      # the flask-sock VoiceLink websocket
        self.ws_lock = threading.Lock()
        self.agent_cfg = agent_cfg
        self.lead = lead
        self.meeting = meeting
        self.callback_url = callback_url
        self.inbound = inbound
        self.greeting_audio = greeting_audio   # A-law bytes that Eva itself plays (inbound only)
        self.greeting_text = greeting_text
        self.call_sid = call_sid
        self.room_name = f"voicelink-{call_id}-{uuid.uuid4().hex[:8]}"
        self._trunk_identity = f"voicelink-trunk-{call_id}"

        self.room = None
        self.source = None
        self.local_track = None
        self._closed = threading.Event()

        # outbound audio (agent -> phone): filled by the LiveKit loop thread,
        # drained by a thread that owns the websocket
        self._out = deque()
        self.started = threading.Event()   # set when VoiceLink's "start" event arrives
        self.stream_sid = None
        self._sent_frames = 0
        self._in_frames = 0

        # inbound audio (phone -> room). Everything below is touched ONLY on the
        # LiveKit loop thread, so no locks are needed and frame order is guaranteed.
        self._live = False
        self._prebuf = bytearray()
        self._in_q = None

        # Eva plays the greeting itself; agent audio waits until it is done
        self.greeting_done = threading.Event()
        if not (inbound and greeting_audio):
            self.greeting_done.set()

        self._agent_joined = False
        self._hangup_flag = False
        self._hangup_done = False

    # ---------------- lifecycle ----------------
    def start(self):
        """Creates the room, connects the trunk participant, publishes the
        audio track, THEN dispatches the agent. Blocks until done."""
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
            .with_identity(self._trunk_identity)
            .with_name("VoiceLink Trunk")
            .with_grants(api.VideoGrants(room_join=True, room=self.room_name))
            .to_jwt()
        )

        self.room = rtc.Room()
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("participant_connected", self._on_participant_connected)
        self.room.on("participant_disconnected", self._on_participant_disconnected)
        self.room.on("local_track_subscribed", self._on_local_track_subscribed)

        print(f"[VOICELINK-BRIDGE] connecting trunk to {self.room_name}...", flush=True)
        await self.room.connect(LIVEKIT_URL, token, options=rtc.RoomOptions(auto_subscribe=True))
        print(f"[VOICELINK-BRIDGE] trunk connected to {self.room_name}", flush=True)

        self.source = rtc.AudioSource(ROOM_AUDIO_RATE, 1)
        self.local_track = rtc.LocalAudioTrack.create_audio_track("voicelink-in", self.source)
        await self.room.local_participant.publish_track(
            self.local_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        print(f"[VOICELINK-BRIDGE] audio track published for {self.room_name}", flush=True)

        # ordered pump: caller audio -> room
        self._in_q = asyncio.Queue()
        asyncio.ensure_future(self._pump_inbound())
        if self.inbound:
            # inbound: keep buffering the caller until the agent has actually subscribed
            asyncio.ensure_future(self._live_fallback())
        else:
            # outbound: the agent greets first, so go live right away (old behaviour)
            self._loop_go_live()

        # Dispatch the agent only now, so it joins a room that already has the caller's mic.
        metadata = json.dumps({
            "agent": self.agent_cfg,
            "lead": self.lead,
            "meeting": self.meeting,
            "call_id": self.call_id,
            "callback_url": self.callback_url,
            "inbound": self.inbound,
            "greeting_played": bool(self.greeting_audio),   # Eva already plays it -> agent must not repeat it
            "greeting_text": self.greeting_text or "",
        })
        print(f"[VOICELINK-BRIDGE] agent_cfg from Pravah for call_id={self.call_id}: "
              f"{json.dumps(self.agent_cfg, default=str)}", flush=True)
        print(f"[VOICELINK-BRIDGE] full dispatch metadata for call_id={self.call_id}: {metadata}", flush=True)
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

    async def _heartbeat(self):
        """Proves the LiveKit loop thread is alive and shows what it sees."""
        for n in range(1, 8):
            await asyncio.sleep(2)
            if self._closed.is_set():
                return
            who = [p.identity for p in self.room.remote_participants.values()] if self.room else []
            print(f"[VOICELINK-BRIDGE] loop alive t={n*2}s in_frames={self._in_frames} "
                  f"out_queue={len(self._out)} sent={self._sent_frames} live={self._live} remote={who}", flush=True)

    # ---------------- room events (run on the LiveKit loop thread) ----------------
    def _on_participant_connected(self, p):
        print(f"[VOICELINK-BRIDGE] participant joined: {p.identity} kind={p.kind}", flush=True)
        if p.identity != self._trunk_identity:
            self._agent_joined = True

    def _on_participant_disconnected(self, p):
        print(f"[VOICELINK-BRIDGE] participant left: {p.identity}", flush=True)
        if p.identity != self._trunk_identity and self._agent_joined and not self._closed.is_set():
            # the agent ended the session (caller said bye / max duration / crash) -> cut the phone call
            print("[VOICELINK-BRIDGE] agent left the room -> hanging up the caller", flush=True)
            self._hangup_flag = True

    def _on_local_track_subscribed(self, *args):
        # the agent is now listening to the caller's mic -> flush what the caller said while it warmed up
        print("[VOICELINK-BRIDGE] agent subscribed to caller audio -> going live", flush=True)
        self._loop_go_live()

    async def _live_fallback(self):
        await asyncio.sleep(INBOUND_LIVE_FALLBACK_SECS)
        if not self._live:
            print("[VOICELINK-BRIDGE] agent subscription not seen in time -> going live anyway", flush=True)
            self._loop_go_live()

    # ---------------- inbound: VoiceLink -> LiveKit room ----------------
    def feed_alaw(self, alaw_bytes: bytes):
        """Call from the VoiceLink websocket thread for every inbound 'media'
        frame. Decodes A-law -> linear16 and hands it to the loop thread, which
        either buffers it (agent still warming up) or pushes it into the room."""
        if self._closed.is_set():
            return
        self._in_frames += 1
        if self._in_frames == 1:
            print(f"[VOICELINK-BRIDGE] first inbound audio frame ({len(alaw_bytes)} bytes)", flush=True)
        pcm16 = audioop.alaw2lin(alaw_bytes, 2)
        _bridge_loop.loop.call_soon_threadsafe(self._loop_ingest, pcm16)

    def _loop_ingest(self, pcm16: bytes):
        if self._closed.is_set():
            return
        if not self._live:
            self._prebuf.extend(pcm16)
            cap = ROOM_AUDIO_RATE * 2 * PREBUF_MAX_SECS
            if len(self._prebuf) > cap:
                del self._prebuf[:len(self._prebuf) - cap]
            return
        self._put_frame(pcm16)

    def _put_frame(self, pcm16: bytes):
        if self._in_q is None:
            return
        if len(pcm16) % 2:
            pcm16 = pcm16[:-1]
        if len(pcm16) < 2:
            return
        self._in_q.put_nowait(rtc.AudioFrame(
            data=pcm16, sample_rate=ROOM_AUDIO_RATE, num_channels=1,
            samples_per_channel=len(pcm16) // 2,
        ))

    def _trim_prebuf(self, pcm: bytes) -> bytes:
        """Drops the silent lead-in so the agent isn't fed seconds of nothing, but keeps a
        short pre-roll before the first spoken word. If the caller said nothing yet,
        only a short tail is kept."""
        win = int(ROOM_AUDIO_RATE * 0.1) * 2                      # 100 ms of pcm16
        pre_roll = int(PREBUF_PRE_ROLL_MS / 100) * win // 1       # 300 ms
        if len(pcm) % 2:
            pcm = pcm[:-1]
        first_loud = None
        for i in range(0, len(pcm), win):
            seg = pcm[i:i + win]
            if len(seg) >= 2 and audioop.rms(seg, 2) >= PREBUF_SPEECH_RMS:
                first_loud = i
                break
        if first_loud is None:
            return pcm[-pre_roll:] if pre_roll else b""
        return pcm[max(0, first_loud - pre_roll):]

    def _loop_go_live(self):
        """Loop thread only. Flushes the buffered caller audio, then streams live."""
        if self._live or self._in_q is None:
            return
        buffered = bytes(self._prebuf)
        self._prebuf = bytearray()
        trimmed = self._trim_prebuf(buffered) if buffered else b""
        step = int(ROOM_AUDIO_RATE * 0.02) * 2                    # 20 ms frames
        for i in range(0, len(trimmed), step):
            self._put_frame(trimmed[i:i + step])
        self._live = True
        print(f"[VOICELINK-BRIDGE] live: flushed {len(trimmed)/(ROOM_AUDIO_RATE*2):.2f}s of buffered caller audio "
              f"(buffered {len(buffered)/(ROOM_AUDIO_RATE*2):.2f}s)", flush=True)

    async def _pump_inbound(self):
        while True:
            frame = await self._in_q.get()
            if frame is None or self._closed.is_set():
                break
            try:
                await self.source.capture_frame(frame)
            except Exception as e:
                print(f"[VOICELINK-BRIDGE] capture_frame failed: {type(e).__name__}: {e!r}", flush=True)

    # ---------------- greeting (inbound, played by Eva itself) ----------------
    def play_greeting(self):
        """Plays Eva's own greeting recording straight to the caller, paced at real time,
        while the LiveKit room / STT / LLM / TTS warm up. Agent audio is held back
        until this finishes (greeting_done)."""
        audio = self.greeting_audio
        try:
            if not audio:
                return
            chunk_bytes = max(160, int(ROOM_AUDIO_RATE * 0.1))     # 100 ms per chunk (1 byte/sample)
            t0 = time.time()
            print(f"[VOICELINK-BRIDGE] playing greeting ({len(audio)/ROOM_AUDIO_RATE:.1f}s) for {self.call_id}", flush=True)
            for i in range(0, len(audio), chunk_bytes):
                if self._closed.is_set():
                    return
                wait = (t0 + (i / ROOM_AUDIO_RATE) - GREETING_LEAD_SECS) - time.time()
                if wait > 0:
                    time.sleep(wait)
                self._send_media_chunk(audio[i:i + chunk_bytes])
        except Exception as e:
            print(f"[VOICELINK-BRIDGE] greeting playback failed: {type(e).__name__}: {e!r}", flush=True)
        finally:
            self.greeting_done.set()

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

    def _send_media_chunk(self, chunk: bytes):
        with self.ws_lock:
            self.ws.send(json.dumps({
                "event": "media",
                "stream_sid": self.stream_sid,
                "media": {"payload": base64.b64encode(chunk).decode("ascii")},
            }))

    def _sender_loop(self):
        """Plain OS thread: the only place that writes agent audio to the VoiceLink
        websocket. Waits for VoiceLink's 'start' event and for the greeting to finish."""
        while not self._closed.is_set():
            if self._hangup_flag:
                self._do_hangup()
                return
            if (not self.started.is_set() or not self.greeting_done.is_set() or not self._out):
                time.sleep(0.005)
                continue
            try:
                chunk = self._out.popleft()
            except IndexError:
                continue
            try:
                self._send_media_chunk(chunk)
                self._sent_frames += 1
                if self._sent_frames == 1:
                    print("[VOICELINK-BRIDGE] first audio frame SENT to VoiceLink", flush=True)
            except Exception as e:
                print(f"[VOICELINK-BRIDGE] ws send failed: {type(e).__name__}: {e!r}", flush=True)
                time.sleep(0.05)

    # ---------------- hang up (agent said goodbye / left the room) ----------------
    def request_hangup(self):
        self._hangup_flag = True

    def _do_hangup(self):
        """Flushes the last agent words, then cuts the call with VoiceLink's `stop`
        event (same event the IVR uses) and closes the socket."""
        if self._hangup_done:
            return
        self._hangup_done = True
        try:
            t0 = time.time()
            while self._out and time.time() - t0 < 4:
                try:
                    self._send_media_chunk(self._out.popleft())
                except IndexError:
                    break
            time.sleep(HANGUP_TAIL_SECS)
            if self.call_sid:
                with self.ws_lock:
                    self.ws.send(json.dumps({"event": "stop", "stop": {"callSid": self.call_sid}}))
                print(f"[VOICELINK-BRIDGE] stop event sent (call_sid={self.call_sid})", flush=True)
                time.sleep(0.3)
        except Exception as e:
            print(f"[VOICELINK-BRIDGE] hangup error: {type(e).__name__}: {e!r}", flush=True)
        finally:
            try:
                self.ws.close()
            except Exception:
                pass

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
        self.greeting_done.set()
        if self._in_q is not None:
            try:
                _bridge_loop.loop.call_soon_threadsafe(self._in_q.put_nowait, None)
            except Exception:
                pass
        if self.room:
            _submit(self.room.disconnect())
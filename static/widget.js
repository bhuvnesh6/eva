/*!
 * Eva Voice Widget
 * Paste this on any page:
 *   <script src="https://YOUR-SERVER:8420/widget.js"></script>
 * Optional overrides:
 *   <script src="https://YOUR-SERVER:8420/widget.js"
 *           data-eva-ws="wss://YOUR-SERVER:8420/ws/eva"
 *           data-eva-name="Eva"></script>
 */
(function () {
  "use strict";

  var thisScript = document.currentScript;
  var wsOverride = thisScript && thisScript.getAttribute("data-eva-ws");
  var evaName = (thisScript && thisScript.getAttribute("data-eva-name")) || "Eva";

  function deriveWsUrl() {
    if (wsOverride) return wsOverride;
    var proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    if (thisScript && thisScript.src) {
      try {
        var u = new URL(thisScript.src);
        var p = u.protocol === "https:" ? "wss:" : "ws:";
        return p + "//" + u.host + "/ws/eva";
      } catch (e) {}
    }
    return proto + "//" + window.location.host + "/ws/eva";
  }

  var WS_URL = deriveWsUrl();

  // ---------------------------------------------------------------
  // Styles
  // ---------------------------------------------------------------
  var css = "\n" +
  "@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');\n" +
  ".eva-w{--eva-bg:#0B0F19;--eva-surface:#131826;--eva-surface2:#1B2233;--eva-accent:#7C9EFF;--eva-accent2:#FFB86B;--eva-text:#F5F7FA;--eva-muted:#8A93A6;--eva-font:'Inter',system-ui,sans-serif;--eva-display:'Space Grotesk',system-ui,sans-serif;position:fixed;bottom:22px;right:22px;z-index:2147483000;font-family:var(--eva-font);}\n" +
  ".eva-w *{box-sizing:border-box;}\n" +
  ".eva-bubble{width:64px;height:64px;border-radius:50%;background:radial-gradient(circle at 32% 28%, #9CB4FF, var(--eva-accent) 60%);box-shadow:0 6px 24px rgba(124,158,255,.45),0 2px 8px rgba(0,0,0,.35);border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;position:relative;transition:transform .25s ease;}\n" +
  ".eva-bubble:hover{transform:scale(1.06);}\n" +
  ".eva-bubble:active{transform:scale(.96);}\n" +
  ".eva-bubble .eva-ring{position:absolute;inset:-6px;border-radius:50%;border:2px solid rgba(124,158,255,.45);opacity:0;}\n" +
  ".eva-bubble.eva-live .eva-ring{animation:eva-pulse-ring 1.8s ease-out infinite;}\n" +
  "@keyframes eva-pulse-ring{0%{opacity:.65;transform:scale(1);}100%{opacity:0;transform:scale(1.55);}}\n" +
  ".eva-bubble svg{width:26px;height:26px;fill:#0B0F19;position:relative;z-index:1;}\n" +
  ".eva-badge{position:absolute;bottom:-4px;left:50%;transform:translateX(-50%);background:var(--eva-surface);color:var(--eva-text);font-family:var(--eva-display);font-size:10px;font-weight:600;letter-spacing:.04em;padding:2px 8px;border-radius:20px;border:1px solid rgba(124,158,255,.35);white-space:nowrap;}\n" +
  ".eva-panel{position:absolute;bottom:80px;right:0;width:320px;max-height:0;overflow:hidden;background:linear-gradient(180deg, var(--eva-surface), var(--eva-bg));border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,.5);border:1px solid rgba(124,158,255,.18);opacity:0;transform:translateY(12px) scale(.98);transition:max-height .38s cubic-bezier(.2,.8,.25,1), opacity .28s ease, transform .32s cubic-bezier(.2,.8,.25,1);}\n" +
  ".eva-panel.eva-open{max-height:520px;opacity:1;transform:translateY(0) scale(1);}\n" +
  ".eva-head{display:flex;align-items:center;gap:10px;padding:16px 16px 10px;}\n" +
  ".eva-orb{width:34px;height:34px;border-radius:50%;background:radial-gradient(circle at 32% 28%, #9CB4FF, var(--eva-accent) 65%);flex:0 0 auto;position:relative;}\n" +
  ".eva-orb.eva-speaking{animation:eva-glow 1.1s ease-in-out infinite;}\n" +
  "@keyframes eva-glow{0%,100%{box-shadow:0 0 0 0 rgba(255,184,107,.0);}50%{box-shadow:0 0 0 8px rgba(255,184,107,.18);}}\n" +
  ".eva-title{flex:1;}\n" +
  ".eva-title .eva-name{font-family:var(--eva-display);font-weight:600;color:var(--eva-text);font-size:15px;line-height:1.1;}\n" +
  ".eva-title .eva-status{font-size:11.5px;color:var(--eva-muted);margin-top:2px;letter-spacing:.01em;}\n" +
  ".eva-close{background:none;border:none;color:var(--eva-muted);cursor:pointer;font-size:18px;line-height:1;padding:4px;}\n" +
  ".eva-close:hover{color:var(--eva-text);}\n" +
  ".eva-body{padding:2px 16px 16px;display:flex;flex-direction:column;align-items:center;gap:14px;}\n" +
  ".eva-stage{width:100%;height:120px;display:flex;align-items:center;justify-content:center;position:relative;}\n" +
  ".eva-bars{display:flex;align-items:flex-end;gap:5px;height:64px;}\n" +
  ".eva-bars span{display:block;width:5px;border-radius:3px;background:linear-gradient(180deg,var(--eva-accent),var(--eva-accent2));height:8px;transform-origin:bottom;transition:height .12s ease;}\n" +
  ".eva-log{width:100%;max-height:140px;overflow-y:auto;font-size:12.5px;color:var(--eva-muted);display:flex;flex-direction:column;gap:8px;padding-right:2px;}\n" +
  ".eva-log::-webkit-scrollbar{width:5px;}\n" +
  ".eva-log::-webkit-scrollbar-thumb{background:rgba(124,158,255,.3);border-radius:6px;}\n" +
  ".eva-msg{padding:8px 10px;border-radius:12px;line-height:1.4;}\n" +
  ".eva-msg.eva-user{background:var(--eva-surface2);color:var(--eva-text);align-self:flex-end;border-bottom-right-radius:4px;}\n" +
  ".eva-msg.eva-eva{background:rgba(124,158,255,.12);color:var(--eva-text);align-self:flex-start;border-bottom-left-radius:4px;}\n" +
  ".eva-talk{width:100%;display:flex;align-items:center;justify-content:center;gap:10px;padding:12px 14px;border-radius:14px;border:none;cursor:pointer;font-family:var(--eva-display);font-weight:600;font-size:13.5px;letter-spacing:.02em;background:linear-gradient(135deg,var(--eva-accent),#5C7CFF);color:#0B0F19;transition:transform .15s ease, filter .15s ease;}\n" +
  ".eva-talk:hover{filter:brightness(1.08);}\n" +
  ".eva-talk:active{transform:scale(.98);}\n" +
  ".eva-talk.eva-active{background:linear-gradient(135deg,var(--eva-accent2),#FF9B5C);}\n" +
  ".eva-talk .eva-dot{width:8px;height:8px;border-radius:50%;background:#0B0F19;opacity:.55;}\n" +
  ".eva-talk.eva-active .eva-dot{animation:eva-blink 1s ease-in-out infinite;}\n" +
  "@keyframes eva-blink{0%,100%{opacity:.25;}50%{opacity:1;}}\n" +
  ".eva-hint{font-size:10.5px;color:var(--eva-muted);text-align:center;}\n" +
  "@media (max-width:420px){.eva-panel{width:calc(100vw - 32px);right:-6px;}}\n";

  var styleTag = document.createElement("style");
  styleTag.setAttribute("data-eva-widget", "1");
  styleTag.textContent = css;
  document.head.appendChild(styleTag);

  // ---------------------------------------------------------------
  // Markup
  // ---------------------------------------------------------------
  var root = document.createElement("div");
  root.className = "eva-w";
  root.innerHTML =
    '<div class="eva-panel" id="eva-panel">' +
      '<div class="eva-head">' +
        '<div class="eva-orb" id="eva-orb"></div>' +
        '<div class="eva-title">' +
          '<div class="eva-name">' + evaName + '</div>' +
          '<div class="eva-status" id="eva-status">Tap the mic to start</div>' +
        '</div>' +
        '<button class="eva-close" id="eva-close" aria-label="Close">&times;</button>' +
      '</div>' +
      '<div class="eva-body">' +
        '<div class="eva-stage"><div class="eva-bars" id="eva-bars"></div></div>' +
        '<div class="eva-log" id="eva-log"></div>' +
        '<button class="eva-talk" id="eva-talk">' +
          '<span class="eva-dot"></span><span id="eva-talk-label">Demo &mdash; Talk to ' + evaName + '</span>' +
        '</button>' +
        '<div class="eva-hint">Mic audio is streamed live. Nothing is recorded.</div>' +
      '</div>' +
    '</div>' +
    '<button class="eva-bubble" id="eva-bubble" aria-label="Open ' + evaName + '">' +
      '<div class="eva-ring"></div>' +
      '<svg viewBox="0 0 24 24"><path d="M12 15a3 3 0 003-3V6a3 3 0 10-6 0v6a3 3 0 003 3zm5-3a5 5 0 01-10 0H5a7 7 0 006 6.92V21h2v-2.08A7 7 0 0019 12h-2z"/></svg>' +
      '<div class="eva-badge">' + evaName + '</div>' +
    '</button>';

  document.body.appendChild(root);

  var bubbleBtn = root.querySelector("#eva-bubble");
  var panel = root.querySelector("#eva-panel");
  var closeBtn = root.querySelector("#eva-close");
  var talkBtn = root.querySelector("#eva-talk");
  var talkLabel = root.querySelector("#eva-talk-label");
  var statusEl = root.querySelector("#eva-status");
  var orbEl = root.querySelector("#eva-orb");
  var barsEl = root.querySelector("#eva-bars");
  var logEl = root.querySelector("#eva-log");

  var NUM_BARS = 24;
  for (var i = 0; i < NUM_BARS; i++) {
    var bar = document.createElement("span");
    barsEl.appendChild(bar);
  }
  var barEls = barsEl.querySelectorAll("span");

  function setBars(level) {
    // level 0..1
    for (var i = 0; i < barEls.length; i++) {
      var jitter = 0.35 + Math.random() * 0.65;
      var h = Math.max(6, Math.min(64, level * 64 * jitter));
      barEls[i].style.height = h + "px";
    }
  }
  function idleBars() {
    for (var i = 0; i < barEls.length; i++) barEls[i].style.height = "6px";
  }

  function addMsg(text, who) {
    var div = document.createElement("div");
    div.className = "eva-msg " + (who === "user" ? "eva-user" : "eva-eva");
    div.textContent = text;
    logEl.appendChild(div);
    logEl.scrollTop = logEl.scrollHeight;
    return div;
  }

  function setStatus(text) { statusEl.textContent = text; }

  bubbleBtn.addEventListener("click", function () {
    panel.classList.toggle("eva-open");
  });
  closeBtn.addEventListener("click", function () {
    panel.classList.remove("eva-open");
  });

  // ---------------------------------------------------------------
  // Audio: mic capture (downsample to 16kHz PCM16) + playback (22050 PCM16)
  // ---------------------------------------------------------------
  var ws = null;
  var audioCtx = null;
  var micStream = null;
  var micSource = null;
  var micProcessor = null;
  var isLive = false;
  var isMuted = false; // self-mute while Eva is speaking, to avoid echo
  var muteUntil = 0;

  var playCtx = null;
  var playTime = 0;
  var partialAssistant = null;

  function ensurePlayCtx() {
    if (!playCtx) {
      playCtx = new (window.AudioContext || window.webkitAudioContext)();
      playTime = playCtx.currentTime;
    }
    return playCtx;
  }

  function playPcm16(int16Buf, sampleRate) {
    var ctx = ensurePlayCtx();
    var floatBuf = new Float32Array(int16Buf.length);
    for (var i = 0; i < int16Buf.length; i++) floatBuf[i] = int16Buf[i] / 32768;
    var audioBuffer = ctx.createBuffer(1, floatBuf.length, sampleRate);
    audioBuffer.copyToChannel(floatBuf, 0);
    var src = ctx.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(ctx.destination);
    var startAt = Math.max(ctx.currentTime, playTime);
    src.start(startAt);
    playTime = startAt + audioBuffer.duration;

    // self-mute mic while this chunk (and a small tail) plays
    isMuted = true;
    muteUntil = Math.max(muteUntil, (startAt + audioBuffer.duration - ctx.currentTime) * 1000 + performance.now() + 350);
  }

  function downsampleTo16k(float32, inRate, outRate) {
    if (outRate === inRate) return float32;
    var ratio = inRate / outRate;
    var outLen = Math.round(float32.length / ratio);
    var out = new Float32Array(outLen);
    for (var i = 0; i < outLen; i++) {
      var srcIdx = i * ratio;
      var i0 = Math.floor(srcIdx);
      var i1 = Math.min(i0 + 1, float32.length - 1);
      var frac = srcIdx - i0;
      out[i] = float32[i0] * (1 - frac) + float32[i1] * frac;
    }
    return out;
  }

  function floatToPcm16(float32) {
    var out = new Int16Array(float32.length);
    for (var i = 0; i < float32.length; i++) {
      var s = Math.max(-1, Math.min(1, float32[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
  }

  function connectWs() {
    ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";

    ws.onopen = function () {
      setStatus("Connecting to " + evaName + "...");
    };

    ws.onmessage = function (evt) {
      if (typeof evt.data === "string") {
        var msg;
        try { msg = JSON.parse(evt.data); } catch (e) { return; }
        handleControl(msg);
      } else {
        var int16 = new Int16Array(evt.data);
        playPcm16(int16, 22050);
      }
    };

    ws.onerror = function () {
      setStatus("Connection error");
    };

    ws.onclose = function () {
      setStatus("Disconnected");
      stopTalking(false);
    };
  }

  function handleControl(msg) {
    if (msg.type === "ready") {
      setStatus("Listening\u2026");
    } else if (msg.type === "status") {
      if (msg.state === "thinking") { setStatus(evaName + " is thinking\u2026"); orbEl.classList.remove("eva-speaking"); }
      else if (msg.state === "speaking") { setStatus(evaName + " is speaking\u2026"); orbEl.classList.add("eva-speaking"); }
      else if (msg.state === "listening") { setStatus("Listening\u2026"); orbEl.classList.remove("eva-speaking"); }
    } else if (msg.type === "user_transcript") {
      if (msg.final) {
        addMsg(msg.text, "user");
      }
    } else if (msg.type === "assistant_delta") {
      if (!partialAssistant) partialAssistant = addMsg("", "eva");
      partialAssistant.textContent += msg.text;
      logEl.scrollTop = logEl.scrollHeight;
    } else if (msg.type === "assistant_done") {
      if (partialAssistant) {
        partialAssistant.textContent = msg.text;
        partialAssistant = null;
      } else if (msg.text) {
        addMsg(msg.text, "eva");
      }
    } else if (msg.type === "error") {
      setStatus(msg.message || "Something went wrong");
    }
  }

  var heartbeatTimer = null;

  function startHeartbeat() {
    stopHeartbeat();
    heartbeatTimer = setInterval(function () {
      if (ws && ws.readyState === WebSocket.OPEN) {
        try { ws.send(JSON.stringify({ type: "ping" })); } catch (e) {}
      }
    }, 20000);
  }
  function stopHeartbeat() {
    if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
  }

  function startTalking() {
    if (isLive) return;

    // Mic access requires a secure context (https://, or http://localhost).
    // On a plain http://IP:port page, getUserMedia is unavailable and would
    // otherwise fail silently, leaving the socket open with no audio ever sent.
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("Mic needs HTTPS \u2014 open this page over https://");
      return;
    }

    isLive = true;
    talkBtn.classList.add("eva-active");
    talkLabel.textContent = "Stop talking";
    bubbleBtn.classList.add("eva-live");
    setStatus("Connecting\u2026");
    connectWs();
    startHeartbeat();

    navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } })
      .then(function (stream) {
        micStream = stream;
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        micSource = audioCtx.createMediaStreamSource(stream);
        micProcessor = audioCtx.createScriptProcessor(4096, 1, 1);

        micProcessor.onaudioprocess = function (e) {
          var input = e.inputBuffer.getChannelData(0);

          // level meter
          var sum = 0;
          for (var i = 0; i < input.length; i += 8) sum += Math.abs(input[i]);
          var level = Math.min(1, (sum / (input.length / 8)) * 6);

          if (isMuted && performance.now() < muteUntil) {
            setBars(0.15 * level);
            return; // don't send Eva's own voice back
          }
          isMuted = false;
          setBars(level);

          if (!ws || ws.readyState !== WebSocket.OPEN) return;
          var down = downsampleTo16k(input, audioCtx.sampleRate, 16000);
          var pcm16 = floatToPcm16(down);
          ws.send(pcm16.buffer);
        };

        micSource.connect(micProcessor);
        micProcessor.connect(audioCtx.destination);
      })
      .catch(function (err) {
        setStatus("Mic access denied");
        stopTalking(true);
      });
  }

  function stopTalking(closeSocket) {
    isLive = false;
    stopHeartbeat();
    talkBtn.classList.remove("eva-active");
    talkLabel.textContent = "Demo \u2014 Talk to " + evaName;
    bubbleBtn.classList.remove("eva-live");
    idleBars();
    orbEl.classList.remove("eva-speaking");
    setStatus("Tap the mic to start");

    if (micProcessor) { micProcessor.disconnect(); micProcessor.onaudioprocess = null; micProcessor = null; }
    if (micSource) { micSource.disconnect(); micSource = null; }
    if (micStream) { micStream.getTracks().forEach(function (t) { t.stop(); }); micStream = null; }
    if (audioCtx) { audioCtx.close(); audioCtx = null; }

    if (closeSocket !== false && ws) {
      try { ws.close(); } catch (e) {}
      ws = null;
    }
  }

  talkBtn.addEventListener("click", function () {
    if (isLive) stopTalking(true);
    else startTalking();
  });

  idleBars();
})();
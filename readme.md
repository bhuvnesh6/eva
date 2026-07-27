# Eva &mdash; bilingual voice agent (web version)

Real-time voice agent: browser mic &rarr; Deepgram (STT) &rarr; Mistral (LLM) &rarr; Sarvam AI (TTS) &rarr; browser speakers.
Speak English or Hindi (or mix them) and Eva replies out loud in the same language, in a couple of seconds.

Two ways to use it:

- **`script.py`** &mdash; the original terminal app. Runs locally, uses your computer's mic/speakers directly. Good for quick testing.
- **`app.py`** &mdash; the web version. A Flask backend that any browser can talk to over a WebSocket, plus:
  - `templates/landing.html` &mdash; a demo/marketing page served at `/`
  - `static/widget.js` &mdash; a single `<script>` tag you can paste into **any** website to add a floating "Talk to Eva" widget in the bottom-right corner

```
eva-voice-agent/
├── app.py                 # Flask + WebSocket backend (the deployed service)
├── script.py               # original CLI voice agent (local mic, no server)
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .env.example
├── templates/
│   └── landing.html        # served at "/"
└── static/
    └── widget.js            # served at "/widget.js", embeddable anywhere
```

## 1. Get API keys

- Deepgram &mdash; https://console.deepgram.com
- Mistral &mdash; https://console.mistral.ai
- Sarvam AI &mdash; https://dashboard.sarvam.ai

## 2. Clone and configure

```bash
git clone <your-repo-url> eva-voice-agent
cd eva-voice-agent
cp .env.example .env
# edit .env and paste in your real keys
```

## 3. Run locally (no Docker)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# terminal / local-mic version
python script.py

# OR web version
python app.py
# -> open http://localhost:8420
```

## 4. Run with Docker (on your VPS)

```bash
git clone <your-repo-url> eva-voice-agent
cd eva-voice-agent
cp .env.example .env
nano .env            # fill in DEEPGRAM_API_KEY / MISTRAL_API_KEY / SARVAM_API_KEY

# build
docker build -t eva-voice-agent .

# run (8420 is Eva's default, unique port — change the host side if it's taken)
docker run -d \
  --name eva \
  --restart unless-stopped \
  --env-file .env \
  -p 8420:8420 \
  eva-voice-agent

# check it's alive
curl http://localhost:8420/health
docker logs -f eva
```

Visit `http://YOUR_SERVER_IP:8420/` to see the landing page, or `http://YOUR_SERVER_IP:8420/widget.js` to confirm the widget script is being served.

### Putting it behind HTTPS (recommended)

Browsers require a secure context (`https://`) to use the microphone on any
domain other than `localhost`. Put Nginx/Caddy/Traefik in front of the
container with a TLS certificate (e.g. Let's Encrypt) and reverse-proxy to
`127.0.0.1:8420`, making sure `Upgrade`/`Connection` headers are forwarded so
the `/ws/eva` WebSocket connection works. Example Nginx snippet:

```nginx
location / {
    proxy_pass http://127.0.0.1:8420;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
}
```

## 5. Embed the widget on any website

Paste this before `</body>` on any page:

```html
<script src="https://your-domain.com/widget.js"></script>
```

If the widget is hosted on a different domain than the page it's embedded
in, you can point it at the backend explicitly:

```html
<script
  src="https://your-domain.com/widget.js"
  data-eva-ws="wss://your-domain.com/ws/eva"
  data-eva-name="Eva">
</script>
```

A floating bubble appears bottom-right. Clicking it opens a panel with a
"Demo &mdash; Talk to Eva" button; clicking that asks for mic permission and
starts a live conversation, with a waveform, live transcript, and smooth
listening/thinking/speaking states.

## Notes

- `app.py` never touches your server's own mic/speakers &mdash; audio is
  captured in the visitor's browser and streamed over the WebSocket, so the
  service can hold many simultaneous conversations, one `EvaSession` per
  connection.
- `script.py` is unchanged and still works standalone for local testing with
  your machine's mic — it doesn't need Docker or a browser at all.
- Change the default port via the `PORT` env var (defaults to `8420`) and the
  default voice via `EVA_SPEAKER` (defaults to `priya`).
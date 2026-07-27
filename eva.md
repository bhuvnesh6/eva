"""
Eva - Real-time voice agent:
  Mic -> Deepgram (STT, streaming) -> Mistral (LLM, streaming, REST)
      -> Sarvam AI (TTS, streaming) -> Speakers

Behavior:
  - Eva listens, and while she's speaking the mic is muted (plus a short
    echo-cooldown after she stops), so she doesn't hear/respond to her own
    voice. No barge-in / interruption logic - simple and reliable.
  - Auto language detection per turn: if you speak English, Eva replies in
    English (English voice). If you speak Hindi (Devanagari script), Eva
    replies in Hindi (Hindi voice). No Hinglish mixing - matches whichever
    language you used.
  - Female voice ("priya" by default).
  - Session conversation history kept in memory (resets each run), capped
    so it doesn't grow unbounded.

Setup:
    pip install deepgram-sdk sounddevice numpy python-dotenv sarvamai httpx

    .env:
        DEEPGRAM_API_KEY=your_deepgram_key
        MISTRAL_API_KEY=your_mistral_key
        SARVAM_API_KEY=your_sarvam_key

Run:
    python voice_agent.py
    python voice_agent.py --speaker neha     # try a different female voice
"""
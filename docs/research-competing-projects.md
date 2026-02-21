# Competing TTS Projects: Research Findings

Date: 2026-02-20

## Verdict

**Continue building speak.** No existing project has an audio pipeline comparable to ours. Our gapless playback, two-tier cache, clause-level streaming, and backpressure pacing are unique across all projects analyzed.

## Projects Analyzed

### 1. agent-tts (kiliman)
- **What it is:** File-monitoring React web app that watches agent chat logs and speaks new messages
- **TTS approach:** Kokoro via HTTP API (requires separate Kokoro server), ElevenLabs, or OpenAI
- **Audio pipeline:** Per-message subprocess (`afplay`/`ffplay`), no streaming, no gapless
- **Caching:** Timestamp-keyed file archive (not semantic)
- **Multi-agent:** Profile-based with manual config per agent
- **Notable feature:** Uses Claude Haiku to summarize long messages before speaking (LLM summarization)
- **Stars:** 14 | **Last active:** Dec 2025
- **Gap vs speak:** No gapless playback, no streaming, no semantic cache, HTTP-only Kokoro

### 2. clarvis/lspeak (nickpending)
- **What it is:** Thin Bun hook processor delegating to lspeak Python TTS daemon
- **TTS approach:** Kokoro (full PyTorch), ElevenLabs, system TTS
- **Audio pipeline:** Per-sentence `pygame.mixer.music.load()` + `.play()`, audible gaps between sentences
- **Caching:** Semantic via FAISS + sentence-transformers (clever but heavy deps: PyTorch, FAISS, spaCy)
- **Multi-agent:** None
- **Notable feature:** Jarvis-style LLM summarization with verbosity modes (terse/brief/normal/full)
- **Stars:** 1 | **Last active:** Recent (alpha)
- **Gap vs speak:** Per-sentence playback (not gapless), massive dependency tree, no caller identification

### 3. multi-agent-observability-with-TTS (triepod-ai)
- **What it is:** Full Vue/Bun observability dashboard for Claude Code agents
- **TTS approach:** Subprocess calls to OpenAI API per message
- **Audio pipeline:** Fire-and-forget subprocess per message, no streaming
- **Caching:** None
- **Multi-agent:** 30+ agent type classifications, voice-per-event-type (not per-agent)
- **Notable feature:** Comprehensive Claude Code hook integration (all 9 events), real-time web dashboard
- **Stars:** 0 | **Last active:** Oct 2025 (abandoned)
- **Gap vs speak:** No gapless, no streaming, no local TTS, no audio engineering. "2.3M ops/sec" measures Python heap ops, not audio throughput

### 4. AgentVibes (paulpreibisch)
- **What it is:** UX/config layer for making AI agents audible across platforms
- **TTS approach:** Piper TTS, macOS `say`, Soprano neural engine
- **Audio pipeline:** One-shot generate-file-play-file with 12s timeout
- **Caching:** None
- **Multi-agent:** BMAD agent auto-detection, 914 voice variants
- **Notable feature:** Voice browser console app, reverb effects, background music mixing, 180+ security tests for custom audio
- **Gap vs speak:** No caching, no streaming, no gapless, no queue management

### 5. VoiceMode (mbailey)
- **What it is:** Full bidirectional voice conversation system as MCP server
- **TTS approach:** Kokoro via kokoro-fastapi HTTP, OpenAI cloud
- **Audio pipeline:** `sounddevice` NonBlockingAudioPlayer with threading, no gapless transitions
- **Caching:** None
- **Multi-agent:** Per-agent config dirs, but basic voice differentiation
- **Notable feature:** Whisper STT + TTS bidirectional, operator agent system running headless Claude in tmux, iOS companion app
- **Gap vs speak:** No audio cache, Kokoro via HTTP (not in-process), no gapless, no caller tones

### 6. cc-hooks (husniadil)
- **What it is:** Per-session FastAPI server for Claude Code hook audio feedback
- **TTS approach:** Prerecorded sound effects (zero latency), gTTS (cloud), ElevenLabs (cloud)
- **Audio pipeline:** Independent per-event processing, no queue awareness
- **Caching:** Basic TTS cache flag
- **Multi-agent:** Excellent per-session isolation (port-per-instance), SQLite with atomic claims
- **Notable feature:** SQLite event persistence with retry logic (PENDING -> PROCESSING -> COMPLETED/FAILED), per-event-type sound effects
- **Gap vs speak:** No streaming, no local neural TTS, no gapless. But superior event persistence and session isolation

### 7. claude-mlx-tts (aperepel)
- **What it is:** Claude Code plugin for voice-cloned notification summaries
- **TTS approach:** MLX Chatterbox on Apple Silicon (~4GB GPU), macOS `say` fallback
- **Audio pipeline:** Non-blocking subprocess per notification
- **Caching:** None
- **Multi-agent:** None
- **Notable feature:** Voice cloning from WAV samples, AI-summarized ~12-word notifications, threshold-based triggering (only speaks on 15s+ work), TUI config interface
- **Gap vs speak:** No queue, no cache, no gapless. Apple Silicon only

### 8. RealtimeTTS (KoljaB)
- **What it is:** General-purpose streaming TTS library (not agent-specific)
- **TTS approach:** 13 backends including Kokoro, Piper, Coqui, StyleTTS2, Orpheus, cloud engines
- **Audio pipeline:** PyAudio with threading, sentence-level streaming, no gapless transitions
- **Caching:** None
- **Multi-agent:** None
- **Notable feature:** Voice formula blending (`0.3*af_sarah + 0.7*am_adam`), NLTK/Stanza sentence tokenization, word-level timing callbacks, 13 engine backends
- **Gap vs speak:** Library not daemon, PyAudio dependency, no caching, maintainer stepped back

### 9. AgentVoices (Benny Cheung)
- **What it is:** Reference implementation for hearing AI agents work
- **TTS approach:** ElevenLabs cloud API only
- **Audio pipeline:** Temp file per utterance, `afplay` playback
- **Multi-agent:** Per-agent-type ElevenLabs voice mapping
- **Notable feature:** COMPLETED convention - agents output standardized ~12-word past-tense summaries
- **Gap vs speak:** Minimal prototype, cloud-only, no queue, no caching

### 10. slaygent-communication (breakshit.blog)
- **What it is:** Tmux-centric multi-agent communication with Piper TTS
- **TTS approach:** Piper TTS via FastAPI server (6 voice models preloaded)
- **Audio pipeline:** WAV to temp file, system player
- **Multi-agent:** Tmux-based visible inter-agent messaging, agent discovery server
- **Notable feature:** Agents communicate by writing to each other's tmux panes, making coordination visible
- **Gap vs speak:** Prototype quality, no caching, no streaming, no queue

## Feature Comparison Matrix

| Feature | speak | agent-tts | clarvis | observability | AgentVibes | VoiceMode | cc-hooks | mlx-tts | RealtimeTTS |
|---------|-------|-----------|---------|---------------|------------|-----------|----------|---------|-------------|
| Gapless playback | Yes | No | No | No | No | No | No | No | No |
| Clause-level streaming | Yes | No | No | No | No | No | No | Partial | Sentence |
| Two-tier audio cache | Yes | No | FAISS | No | No | No | Basic | No | No |
| Word-level cache | Yes | No | No | No | No | No | No | No | No |
| Local neural TTS | Yes | HTTP only | Yes | No | Yes | HTTP only | No | Yes | Yes |
| Caller identification | Yes | Profiles | No | Event-type | BMAD | Config | Session | No | No |
| Caller tones | Yes | No | No | No | No | No | No | No | No |
| Backpressure pacing | Yes | No | No | No | No | No | No | No | No |
| State event file | Yes | WebSocket | No | WebSocket | No | No | No | No | No |
| Hook integration | No | File watch | Stop hook | All 9 hooks | MCP | MCP | All hooks | Stop hook | N/A |
| LLM summarization | No | Yes | Yes | No | No | No | Optional | Yes | No |
| Voice blending | No | No | No | No | No | No | No | Clone | Yes |

## Actionable Features to Adopt (Ideas, Not Code)

Filed as beads issues with `research-finding` label:

1. **P1 speak-auf:** Fix ffplay stale audio (reliability)
2. **P2 speak-5cb:** SQLite event persistence with retry (from cc-hooks)
3. **P2 speak-enf:** Claude Code hook integration (from clarvis, cc-hooks, observability)
4. **P3 speak-jr7:** LLM summarization of verbose output (from agent-tts, clarvis)
5. **P3 speak-69m:** COMPLETED convention for brief summaries (from Benny Cheung)
6. **P3 speak-1xv:** NLTK sentence tokenization (from RealtimeTTS)
7. **P3 speak-0vs:** Per-event-type sound effects (from cc-hooks)
8. **P4 speak-2w6:** Voice formula blending (from RealtimeTTS)

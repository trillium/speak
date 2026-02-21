# speak

Local text-to-speech CLI powered by [Kokoro](https://github.com/hexgrad/kokoro) (with Piper as a fallback engine). A background daemon keeps the model loaded so subsequent calls are fast.

## Quick start

```bash
speak "Hello world"
echo "piped text" | speak
```

The daemon starts automatically on first use. It shuts down after 5 minutes of inactivity.

## Requirements

- **uv** (Python package runner)
- **ffmpeg** (`ffplay` command for audio output)
- **jq** (for hook scripts)
- Kokoro model files in `~/.local/share/speak/kokoro/`:
  - `kokoro-v1.0.onnx`
  - `voices-v1.0.bin`

## Options

| Flag | Description |
|------|-------------|
| `--engine ENGINE` | TTS engine: `kokoro` (default) or `piper` |
| `--voice NAME` | Voice name (default: `af_heart`) |
| `--speed FLOAT` | Speech speed (default: `1.26`) |
| `--save FILE` | Save to WAV file instead of playing |
| `--voices` | List available voices |
| `--daemon` | Show daemon status / start it |
| `--stop` | Stop the daemon |
| `--enqueue` | Queue text and return immediately (non-blocking) |
| `--queue` | Show playback queue status |
| `--skip` | Skip the currently playing queued item |
| `--clear` | Clear all pending items in the queue |
| `--replay` | Replay the last queued item |
| `--stats` | Show daemon stats (uptime, cache, queue totals) |
| `--caller NAME` | Caller identity (plays a unique tone per caller) |

## Playback modes

### Sync (default)

Blocks until playback finishes. Audio streams from the daemon to the client, which pipes it to `ffplay`.

```bash
speak "This blocks until done."
```

### Queue (fire-and-forget)

Returns immediately. The daemon synthesizes and plays items sequentially through a persistent `ffplay` process. No overlap — items play one at a time in FIFO order. PCM audio is written in small chunks with backpressure-based pacing for gapless playback.

```bash
speak --enqueue "First thing to say"
speak --enqueue "Second thing to say"   # returns instantly, plays after first
echo "summary of results" | speak --enqueue
```

Manage the queue:

```bash
speak --queue    # show what's playing and what's pending
speak --skip     # skip the current item, advance to next
speak --clear    # drop all pending items (current keeps playing)
speak --replay   # replay the last completed item
```

`--skip` + `--clear` together stops everything:

```bash
speak --clear && speak --skip
```

## Caller identification

When multiple agents or tools use speak, each caller gets a distinct audio identity:

```bash
speak --enqueue --caller myproject "Status update"
```

If `--caller` is not specified, it auto-detects from the git repo name (or falls back to the current directory basename). Set `SPEAK_CALLER=name` to override globally.

Each caller gets:
- **Unique tone** — 1-3 beeps from a pentatonic scale, derived from the caller name hash
- **Distinct voice** — configurable per-caller in the daemon's `CALLER_VOICES` dict
- **Volume normalization** — per-voice gain adjustment
- **Start/end tones** — caller tone plays before and after each message
- **Silence gaps** — 1-second gap between different callers, separator chime between same-caller items

Default voice assignments:

| Caller | Voice | Gain |
|--------|-------|------|
| `speak` | `af_heart` | 1.0 |
| `happy` | `am_adam` | 1.0 |
| `ops` | `af_nova` | 1.5 |

## Text processing pipeline

Two tools process text before it reaches the TTS engine:

### `bin/summarize`

Standalone LLM summarizer. Reads text from stdin, outputs a brief summary to stdout. Uses `claude -p` with Haiku by default. If text is already short enough, passes it through unchanged.

```bash
echo "long verbose text..." | summarize
echo "long verbose text..." | summarize --max-words 20
echo "long verbose text..." | summarize --model claude-haiku-4-5-20251001
```

Swap the LLM backend by editing this one file — no other tooling changes needed.

### `bin/speak-summarize`

Pronunciation fixes and phrase rewrites for TTS. Reads text from stdin, applies transforms from `config/rewrites.json`, writes to stdout. No LLM calls — pure text transformation.

```bash
echo "The API daemon handles async requests" | speak-summarize
# → "The eh pee eye day-mon handles eh-sink requests"
```

### `config/rewrites.json`

Two sections:

**`pronunciation`** — word-level substitutions to fix TTS mispronunciations:
```json
{
  "daemon": "day-mon",
  "API": "eh pee eye",
  "async": "eh-sink",
  "JSON": "jason",
  "startup": "start-up"
}
```

**`phrase_rewrites`** — strip AI filler phrases or map them to better alternatives:
```json
{
  "You're absolutely right": "",
  "I'd be happy to": "I'll",
  "Fantastic!": ""
}
```

## Claude Code hooks

Hook scripts in `.claude/hooks/` integrate with Claude Code lifecycle events. Configured in `~/.claude/settings.json`:

### Setup

The hooks are registered globally in `~/.claude/settings.json` under the `hooks` key. They fire asynchronously (never block Claude) on these events:

| Event | What it does |
|-------|-------------|
| **Stop** | Summarizes Claude's response with Haiku, applies pronunciation/phrase rewrites, speaks the summary |
| **Notification** | Speaks the notification message (permission prompts, idle alerts) with rewrites applied |
| **SubagentStop** | Summarizes subagent output and speaks it |

### Pipeline

```
Claude finishes → hook receives JSON on stdin
  → extract last_assistant_message
  → bin/summarize (LLM → 1-2 sentence summary)
  → bin/speak-summarize (pronunciation + phrase rewrites)
  → speak --enqueue --caller claude
```

### Testing hooks manually

```bash
echo '{"hook_event_name":"Stop","stop_hook_active":false,"last_assistant_message":"I refactored the auth module and added tests.","session_id":"test","cwd":"/tmp","permission_mode":"default","transcript_path":"/dev/null"}' | .claude/hooks/speak-hook.sh
```

### Notes

- Hooks are snapshotted at session startup — restart Claude Code after changing `settings.json`
- Stop hooks check `stop_hook_active` to prevent infinite loops
- All hooks run with `async: true` so they never block Claude

## Voices

Kokoro voice naming convention:

| Prefix | Meaning |
|--------|---------|
| `af_` | American female |
| `am_` | American male |
| `bf_` | British female |
| `bm_` | British male |

Some voices: `af_heart`, `af_sarah`, `af_bella`, `af_nova`, `af_sky`, `am_adam`, `am_michael`, `am_eric`, `am_liam`, `am_onyx`, `am_puck`.

```bash
speak --voice am_adam "Hello in a male voice"
speak --speed 1.3 "A bit faster"
speak --voices   # list all available voices
```

## Architecture

```
speak (bash)  →  speak-client (python)  →  speak-daemon (python, Unix socket)
                      │                          │
                      │ streams PCM              │ Kokoro model (loaded once)
                      ↓                          │ Two-tier audio cache
                 ffplay (pipe)                   │ PlaybackQueue (for --enqueue)
                                                 ↓
                                            ffplay (persistent, daemon-owned)
```

- **speak** — CLI entry point, parses flags, manages daemon lifecycle, auto-detects caller
- **speak-client** — connects to daemon over Unix socket (`/tmp/speak-$USER.sock`), sends JSON request, receives length-prefixed PCM chunks
- **speak-daemon** — loads Kokoro model once, serves TTS requests, manages audio cache and playback queue
- **summarize** — LLM summarizer (claude -p with Haiku), swappable backend
- **speak-summarize** — pronunciation and phrase rewrites from `config/rewrites.json`

### Audio cache

Two-tier disk cache in `/tmp/speak-cache-$USER/` (3-day TTL):

1. **Clause cache** — keyed by full clause text, highest quality
2. **Word cache** — keyed by per-word phonemes, enables fast assembly of novel clauses from previously-heard words. Background synthesis upgrades assembled results to clause cache.

### State events

The daemon publishes state to `/tmp/speak-$USER.state.json` on every change (enqueued, playing, item_done, skipped, cleared, idle). External tools (Talon, Hammerspoon, etc.) can watch this file.

```json
{"event": "playing", "playing": {"id": 2, "caller": "claude", "voice": "af_heart", "text": "..."}, "pending": 0, "queue": [], "timestamp": 1771632606.5}
```

### Daemon management

```bash
speak --daemon   # check status or start
speak --stop     # stop the daemon
```

The daemon auto-starts when you run `speak` and auto-stops after 5 minutes idle (unless the playback queue is active).

### Logging

The daemon logs to `/tmp/speak-$USER.sock.log`. Queued items produce per-clause timing data:

- **SYN** = full synthesis, **HIT** = cache hit, **ASM** = assembled from word cache
- **synth** = synthesis time (ms)
- **write** = time writing PCM to ffplay (high values mean backpressure — we're ahead of playback)
- **audio** = seconds of audio produced
- **speed** = speed value used

## Socket protocol

Length-prefixed messages over Unix socket. Each message is `[4-byte big-endian length][payload]`. A zero-length message signals end of stream.

**Sync request:**
```json
{"text": "hello", "voice": "af_heart", "speed": 1.0, "lang": "en-us"}
```
Response: sequence of length-prefixed PCM chunks, terminated by zero-length chunk.

**Enqueue request:**
```json
{"enqueue": true, "text": "hello", "voice": "af_heart", "speed": 1.0, "caller": "myproject"}
```
Response: `{"ok": true, "position": 1}` (length-prefixed + zero terminator).

**Command request:**
```json
{"command": "queue_status"}
{"command": "skip"}
{"command": "clear"}
{"command": "replay"}
{"command": "stats"}
```
Response: JSON result (length-prefixed + zero terminator).

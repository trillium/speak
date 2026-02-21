# Pronunciation & Phrase Rewrites: Agent Guide

This document describes how to interact with the speak tool's pronunciation and phrase rewrite review system. Use this to test, accept, or reject text transformations that are applied before text-to-speech.

## What this system does

The speak tool transforms text before sending it to the TTS engine:

1. **Pronunciation fixes** — technical words that TTS mispronounces get rewritten to phonetic equivalents (e.g. `daemon` → `day-mon`, `API` → `eh pee eye`)
2. **Phrase rewrites** — common filler phrases get stripped or replaced (e.g. `You're absolutely right` → `I'm a cat meow`)

All entries live in `config/rewrites.json`. Review status is tracked in `config/rewrites-review.json`.

## Commands

All commands go through the `speak` CLI:

### List all rewrites with their review status
```bash
speak --rewrites list
```
Output shows `✓` for accepted, `✗` for rejected, `?` for pending.

### Show only unreviewed entries
```bash
speak --rewrites pending
```

### Hear how a rewrite sounds (speaks the before→after)
```bash
speak --rewrites test daemon
speak --rewrites test API
speak --rewrites test "You're absolutely right"
```
This queues a spoken comparison into the speak playback queue.

### Speak all unreviewed entries
```bash
speak --rewrites test-all
```

### Accept a rewrite (mark as confirmed good)
```bash
speak --rewrites accept daemon
speak --rewrites accept API
speak --rewrites accept "You're absolutely right"
```

### Reject a rewrite (mark as needs fixing)
```bash
speak --rewrites reject CLI
speak --rewrites reject "Great question"
```

### Review past decisions by category
```bash
speak --rewrites review accepted     # re-hear all accepted entries
speak --rewrites review rejected     # re-hear all rejected entries
speak --rewrites review unnecessary  # re-hear entries marked unnecessary
speak --rewrites review pending      # re-hear pending entries (same as test-all)
speak --rewrites review              # re-hear everything (default: all)
```
Speaks each matching entry so you can re-evaluate. Use accept/reject/fix afterward to change any entry.

### Regenerate all rejected pronunciations
```bash
speak --rewrites regenerate
```
Finds all rejected pronunciation entries, uses an LLM to generate a new phonetic spelling for each (avoiding previously rejected values), updates `rewrites.json`, and resets status to pending. Run `test-all` afterward to review the new values.

## Workflow

1. Run `speak --rewrites test-all` to hear all unreviewed entries
2. For each one you hear, run either:
   - `speak --rewrites accept <key>` if it sounds correct
   - `speak --rewrites reject <key>` if it sounds wrong
3. Run `speak --rewrites regenerate` to get LLM-suggested replacements for all rejected entries
4. Repeat from step 1 — rejected values are tracked internally so the LLM won't repeat them
5. Run `speak --rewrites pending` to see what's left
6. Run `speak --rewrites list` to see the full picture

## Key matching

Keys are matched by exact name first, then case-insensitive. For phrase rewrites that contain spaces or punctuation, wrap in quotes:
```bash
speak --rewrites test "I'd be happy to"
speak --rewrites accept "I'd be happy to"
```

## Review data

Review status is stored in `config/rewrites-review.json` as a JSON object keyed by `section:word`. Each entry has:
- `word` — the key from rewrites.json
- `value` — the replacement value at time of review
- `section` — `pronunciation` or `phrase_rewrites`
- `status` — `accepted`, `rejected`, `pending`, or `unnecessary`
- `reviewed_at` — Unix timestamp (present on accept/reject)
- `rejected_alts` — (internal) list of previously rejected values, fed to LLM during regenerate to avoid repeats

## Config file format

`config/rewrites.json` has two sections:

```json
{
  "pronunciation": {
    "daemon": "day-mon",
    "API": "eh pee eye"
  },
  "phrase_rewrites": {
    "You're absolutely right": "I'm a cat meow",
    "Fantastic!": ""
  }
}
```

Pronunciation entries replace individual words. Phrase rewrite entries replace full phrases (empty string `""` means remove the phrase entirely).

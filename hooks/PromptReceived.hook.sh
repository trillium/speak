#!/usr/bin/env bash
set -euo pipefail

# PromptReceived.hook.sh — Play a tone when Claude Code receives user input
#
# TRIGGER: UserPromptSubmit
#
# Plays the current session's input tone (pluck waveform) via the speak daemon.
# Same pitches as the session's caller tone but different timbre, so you can
# tell which window received your text by ear.

# Read stdin (hook payload) — must consume it even if unused
INPUT=""
if ! INPUT=$(timeout 1 cat 2>/dev/null); then
    exit 0
fi

# Verify this is actually a UserPromptSubmit event
EVENT=$(echo "$INPUT" | jq -r '.hook_event_name // empty' 2>/dev/null)
if [[ "${EVENT}" != "UserPromptSubmit" ]]; then
    exit 0
fi

SPEAK="${HOME}/code/speak/bin/speak"

# Play the input tone for this session (pluck waveform, session-keyed pitches)
"${SPEAK}" --play-tone &

exit 0

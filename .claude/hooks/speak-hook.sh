#!/usr/bin/env bash
set -euo pipefail

# Claude Code lifecycle hook that speaks summaries via speak.
# Adapted from patterns in cc-hooks and clarvis.
# Reads JSON from stdin, summarizes via bin/summarize, speaks via speak --enqueue.

# 1-second timeout on stdin read (robustness, from clarvis)
INPUT=""
if ! INPUT=$(timeout 1 cat 2>/dev/null); then
    exit 0
fi

EVENT=$(echo "$INPUT" | jq -r '.hook_event_name // empty')
if [[ -z "${EVENT}" ]]; then
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
BIN="${SCRIPT_DIR}/../../bin"
SPEAK="${BIN}/speak"
SUMMARIZE="${BIN}/summarize"
REWRITE="${BIN}/speak-summarize"

# Helper: rewrite pronunciation/phrases then speak
_speak() {
    python3 "${REWRITE}" | "${SPEAK}" --enqueue --caller claude
}

# Helper: summarize then rewrite then speak
_summarize_and_speak() {
    "${SUMMARIZE}" | _speak
}

case "${EVENT}" in
    Stop)
        ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false')
        if [[ "${ACTIVE}" == "true" ]]; then
            exit 0
        fi
        MSG=$(echo "$INPUT" | jq -r '.last_assistant_message // empty')
        if [[ -n "${MSG}" ]]; then
            printf '%s' "${MSG}" | _summarize_and_speak
        fi
        ;;

    Notification)
        MSG=$(echo "$INPUT" | jq -r '.message // empty')
        if [[ -n "${MSG}" ]]; then
            # Notifications are already short, just rewrite
            printf '%s' "${MSG}" | _speak
        fi
        ;;

    SubagentStop)
        ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false')
        if [[ "${ACTIVE}" == "true" ]]; then
            exit 0
        fi
        MSG=$(echo "$INPUT" | jq -r '.last_assistant_message // empty')
        AGENT_TYPE=$(echo "$INPUT" | jq -r '.agent_type // "agent"')
        if [[ -n "${MSG}" ]]; then
            printf '%s' "${AGENT_TYPE} agent finished. ${MSG}" | _summarize_and_speak
        fi
        ;;
esac

exit 0

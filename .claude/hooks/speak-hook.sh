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

# --- Failure logging ---
# Append a JSON line naming the pipeline stage that failed, using the same
# {ts, stage, caller, exit_code, text_head} shape bin/speak's _log_failure
# writes. `set -o pipefail` is already active (set -euo pipefail above), so a
# failed stage surfaces its exit code; this records which stage it was so
# silence stays observable. Best-effort — never breaks the hook.
FAILURES_DIR="${HOME}/.local/share/speak"
FAILURES_FILE="${FAILURES_DIR}/failures.jsonl"

_log_stage_failure() {
    local stage="$1"
    local exit_code="${2:-}"
    local text="${3:-}"

    mkdir -p "${FAILURES_DIR}" 2>/dev/null || return 0

    # Rotate if over 5 MB (mirrors bin/speak's _log_failure)
    if [[ -f "${FAILURES_FILE}" ]] && \
       [[ $(stat -f%z "${FAILURES_FILE}" 2>/dev/null || echo 0) -gt 5242880 ]]; then
        mv "${FAILURES_FILE}" "${FAILURES_FILE}.1" 2>/dev/null || true
    fi

    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    [[ "${exit_code}" =~ ^-?[0-9]+$ ]] || exit_code=-1
    local text_head="${text:0:80}"
    text_head="${text_head//$'\n'/ }"

    if command -v jq >/dev/null 2>&1; then
        jq -cn \
            --arg ts "${ts}" \
            --arg stage "${stage}" \
            --arg caller "claude" \
            --argjson exit_code "${exit_code}" \
            --arg text_head "${text_head}" \
            '{ts:$ts, stage:$stage, caller:$caller, exit_code:$exit_code, text_head:$text_head}' \
            >> "${FAILURES_FILE}" 2>/dev/null || true
    else
        local esc_head="${text_head//\\/\\\\}"; esc_head="${esc_head//\"/\\\"}"
        printf '{"ts":"%s","stage":"%s","caller":"claude","exit_code":%s,"text_head":"%s"}\n' \
            "${ts}" "${stage}" "${exit_code}" "${esc_head}" \
            >> "${FAILURES_FILE}" 2>/dev/null || true
    fi
    return 0
}

# Helper: rewrite pronunciation/phrases then speak. Each stage is run
# separately so a failure can be attributed to rewrite vs. speak.
_speak() {
    local input rewritten rc
    input="$(cat)"

    rc=0
    rewritten="$(printf '%s' "${input}" | python3 "${REWRITE}")" || rc=$?
    if [[ ${rc} -ne 0 ]]; then
        _log_stage_failure "hook:rewrite" "${rc}" "${input}"
        return "${rc}"
    fi

    printf '%s' "${rewritten}" | "${SPEAK}" --enqueue --caller claude || {
        rc=$?
        _log_stage_failure "hook:speak" "${rc}" "${rewritten}"
        return "${rc}"
    }
}

# Helper: summarize then rewrite then speak. Summarize is captured separately
# so its failure is attributed to the summarize stage.
_summarize_and_speak() {
    local input summarized rc
    input="$(cat)"

    rc=0
    summarized="$(printf '%s' "${input}" | "${SUMMARIZE}")" || rc=$?
    if [[ ${rc} -ne 0 ]]; then
        _log_stage_failure "hook:summarize" "${rc}" "${input}"
        return "${rc}"
    fi

    printf '%s' "${summarized}" | _speak
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
            printf '%s' "claude background agent here. ${MSG}" | _summarize_and_speak
        fi
        ;;
esac

exit 0

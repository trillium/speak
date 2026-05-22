#!/usr/bin/env bun
/**
 * SpeakResponse.hook.ts — Speak last assistant response via TTS
 *
 * TRIGGER: Stop
 *
 * Pipes the last assistant response through ~/code/speak/bin/speak
 * with --caller da so Trillium hears the response without reading the screen.
 *
 * Extracts the summary/key content — not the full response. Looks for:
 *   1. SUMMARY section (Algorithm mode)
 *   2. TASK + CHANGE + VERIFY lines (Native mode)
 *   3. Falls back to last_assistant_message truncated to ~500 chars
 *
 * Depends on: ~/code/speak/bin/speak (Kokoro TTS daemon)
 */

import { readHookInput, parseTranscriptFromInput } from './lib/hook-io';
import { spawnSync } from 'child_process';

const SPEAK = `${process.env.HOME}/code/speak/bin/speak`;
const MAX_CHARS = 500;

function extractSpeakableText(text: string): string {
  // Strip markdown formatting for speech
  const clean = (s: string) => s
    .replace(/```[\s\S]*?```/g, '')       // code blocks
    .replace(/`[^`]+`/g, '')              // inline code
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // links → text
    .replace(/[#*_~>|═━]/g, '')           // markdown decoration
    .replace(/\n{3,}/g, '\n\n')           // collapse whitespace
    .trim();

  // Try Algorithm SUMMARY section
  const summaryMatch = text.match(/(?:📋\s*SUMMARY|SUMMARY)[:\s]*\n([\s\S]*?)(?:\n(?:🖊|🗣|═|---)|$)/i);
  if (summaryMatch) return clean(summaryMatch[1]).slice(0, MAX_CHARS);

  // Try STORY section (Algorithm closing)
  const storyMatch = text.match(/(?:🖊️?\s*STORY)[:\s]*\n([\s\S]*?)(?:\n(?:🗣|═|---)|$)/i);
  if (storyMatch) return clean(storyMatch[1]).slice(0, MAX_CHARS);

  // Try Native mode TASK line
  const taskMatch = text.match(/🗒️\s*TASK:\s*(.+)/);
  if (taskMatch) {
    let result = taskMatch[1].trim();
    // Append CHANGE bullets if present
    const changeMatch = text.match(/🔧\s*CHANGE:\s*\n([\s\S]*?)(?:\n✅|$)/);
    if (changeMatch) {
      const bullets = changeMatch[1]
        .split('\n')
        .map(l => l.replace(/^[-•]\s*/, '').trim())
        .filter(Boolean)
        .join('. ');
      result += '. ' + bullets;
    }
    return clean(result).slice(0, MAX_CHARS);
  }

  // Fallback: first paragraph of meaningful text
  const lines = clean(text).split('\n').filter(l => l.trim().length > 10);
  return lines.slice(0, 5).join(' ').slice(0, MAX_CHARS);
}

async function main() {
  const input = await readHookInput();
  if (!input) process.exit(0);

  let lastResponse = input.last_assistant_message;
  if (!lastResponse) {
    const parsed = await parseTranscriptFromInput(input);
    lastResponse = parsed.lastMessage;
  }

  if (!lastResponse || lastResponse.trim().length < 10) process.exit(0);

  const speakable = extractSpeakableText(lastResponse);
  if (!speakable) process.exit(0);

  // Fire and forget — speak returns immediately (enqueues to daemon)
  spawnSync(SPEAK, ['--caller', 'da', speakable], {
    timeout: 5000,
    stdio: 'ignore',
  });

  process.exit(0);
}

main().catch(() => process.exit(0));

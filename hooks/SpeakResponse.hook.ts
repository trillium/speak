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
const SPEAK_SUMMARIZE = `${process.env.HOME}/code/speak/bin/speak-summarize`;

function tablesToSpeech(text: string): string {
  const lines = text.split('\n');
  const result: string[] = [];
  let headers: string[] = [];
  let inTable = false;

  for (const line of lines) {
    const stripped = line.trim();
    if (stripped.startsWith('|') && stripped.endsWith('|')) {
      const cells = stripped.slice(1, -1).split('|').map(c => c.trim());
      // Separator row (|---|---|) — skip
      if (cells.every(c => /^[-: ]+$/.test(c))) continue;
      // Header row
      if (!inTable) {
        headers = cells;
        inTable = true;
        continue;
      }
      // Data row — pair with headers
      if (headers.length && cells.length === headers.length) {
        const parts = headers.map((h, i) => cells[i] ? `${h}, ${cells[i]}` : '').filter(Boolean);
        result.push(parts.join('. ') + '.');
      } else {
        result.push(cells.filter(Boolean).join(', ') + '.');
      }
    } else {
      if (inTable) { inTable = false; headers = []; }
      result.push(line);
    }
  }
  return result.join('\n');
}

// Acronyms/initialisms that should be spelled out letter-by-letter
const ACRONYM_SPEECH: Record<string, string> = {
  'cli': 'C L I',
  'api': 'A P I',
  'url': 'U R L',
  'uri': 'U R I',
  'id': 'I D',
  'ids': 'I D s',
  'ui': 'U I',
  'ux': 'U X',
  'db': 'D B',
  'sql': 'sequel',
  'os': 'O S',
  'io': 'I O',
  'pr': 'P R',
  'ci': 'C I',
  'cd': 'C D',
  'sdk': 'S D K',
  'ssr': 'S S R',
  'csr': 'C S R',
  'tts': 'T T S',
  'llm': 'L L M',
  'pcm': 'P C M',
};

// Extension → spoken form mapping
const EXT_SPEECH: Record<string, string> = {
  '.py': ' dot pie',
  '.ts': ' dot T S',
  '.tsx': ' dot T S X',
  '.js': ' dot J S',
  '.jsx': ' dot J S X',
  '.json': ' dot jason',
  '.jsonl': ' dot jason L',
  '.yaml': ' dot yammel',
  '.yml': ' dot yammel',
  '.toml': ' dot tom L',
  '.md': ' dot M D',
  '.sh': ' dot S H',
  '.bash': ' dot bash',
  '.css': ' dot C S S',
  '.html': ' dot H T M L',
  '.sql': ' dot sequel',
  '.rs': ' dot R S',
  '.go': ' dot go',
  '.c': ' dot C',
  '.h': ' dot H',
  '.env': ' dot env',
  '.lock': ' dot lock',
  '.txt': ' dot text',
  '.log': ' dot log',
  '.wav': ' dot wave',
  '.mp3': ' dot M P 3',
  '.png': ' dot P N G',
  '.jpg': ' dot J peg',
  '.svg': ' dot S V G',
  '.onnx': ' dot onyx',
  '.wasm': ' dot wasm',
  '.db': ' dot D B',
};

function filenameToSpeech(filename: string): string {
  const dotIdx = filename.lastIndexOf('.');
  if (dotIdx <= 0) return ACRONYM_SPEECH[filename.toLowerCase()] ?? filename;
  const ext = filename.slice(dotIdx).toLowerCase();
  const name = filename.slice(0, dotIdx);
  const spokenName = ACRONYM_SPEECH[name.toLowerCase()] ?? name;
  const spokenExt = EXT_SPEECH[ext] ?? (' dot ' + ext.slice(1));
  return spokenName + spokenExt;
}

type BlobRule = {
  name: string;
  pattern: RegExp;
  replacement: string | ((match: string, ...groups: string[]) => string);
};

const HEX_WORD_EXCEPTIONS = new Set(['effaced', 'defaced']);

function sys(label: string): string {
  return `<<sys:${label.replace(/ /g, '_').replace(/-/g, '_')}>>`;
}

const STRIPE_PREFIX_MAP: Record<string, string> = {
  pi: sys('payment intent ID'), ch: sys('charge ID'), sub: sys('subscription ID'),
  cus: sys('customer ID'), prod: sys('product ID'), price: sys('price ID'),
  in: sys('invoice ID'), re: sys('refund ID'), evt: sys('event ID'),
  req: sys('request ID'), tok: sys('token ID'), card: sys('card ID'),
  ba: sys('bank account ID'), acct: sys('account ID'),
};

const BLOB_RULES: BlobRule[] = [
  // 1. PEM blocks (multiline, dotAll)
  {
    name: 'pem-block',
    pattern: /-----BEGIN ([A-Z ]+)-----[\s\S]*?-----END \1-----/gm,
    replacement: (_m: string, type: string) => {
      if (/PRIVATE KEY/.test(type)) return sys('private key block');
      if (/CERTIFICATE REQUEST/.test(type)) return sys('certificate request');
      if (/CERTIFICATE/.test(type)) return sys('certificate block');
      if (/PUBLIC KEY/.test(type)) return sys('public key block');
      return sys('key block');
    },
  },

  // 2. JWT tokens
  {
    name: 'jwt',
    pattern: /\beyJ[A-Za-z0-9_-]+=*\.[A-Za-z0-9_-]+=*\.[A-Za-z0-9_-]+=*/g,
    replacement: sys('JWT token'),
  },

  // 3. API keys by known prefix
  {
    name: 'api-key',
    pattern: /\b(sk-(?:proj-|ant-)?[A-Za-z0-9\-_]{20,}|sk_(?:live|test)_[A-Za-z0-9]{20,}|pk_(?:live|test)_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{50,}|xoxb-[A-Za-z0-9\-]+|xapp-[A-Za-z0-9\-]+|AIza[A-Za-z0-9_\-]{35}|hf_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16})\b/g,
    replacement: (m: string) => {
      if (m.startsWith('sk_live_') || m.startsWith('sk_test_')) return sys('Stripe secret key');
      if (m.startsWith('pk_live_') || m.startsWith('pk_test_')) return sys('Stripe publishable key');
      if (m.startsWith('ghp_') || m.startsWith('github_pat_')) return sys('GitHub token');
      if (m.startsWith('xoxb-')) return sys('Slack bot token');
      if (m.startsWith('xapp-')) return sys('Slack app token');
      if (m.startsWith('AIza')) return sys('Google API key');
      if (m.startsWith('hf_')) return sys('Hugging Face token');
      if (m.startsWith('AKIA')) return sys('AWS access key');
      return sys('API key');
    },
  },

  // 4. Env var assignments with secret values (KEY=<long-opaque-value>)
  {
    name: 'env-var-secret',
    pattern: /\b([A-Z][A-Z0-9_]{2,})=([A-Za-z0-9+\/\-_=.]{20,})/g,
    replacement: (_m: string, name: string) => `env var ${name} set`,
  },

  // 5. UUIDs
  {
    name: 'uuid',
    pattern: /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/gi,
    replacement: sys('UUID'),
  },

  // 6. Full SHA-1 (exactly 40 hex chars)
  {
    name: 'sha1-full',
    pattern: /\b[0-9a-f]{40}\b/gi,
    replacement: sys('commit hash'),
  },

  // 7. Short SHA (7–12 hex chars) — replaces existing silent strip
  {
    name: 'sha-short',
    // Anchor on word boundaries and require at least one digit so hex-lettered
    // English words (acceded, defaced) don't match; HEX_WORD_EXCEPTIONS stays
    // as belt-and-suspenders.
    pattern: /\(?\b(?=[0-9a-f]*\d)[0-9a-f]{7,12}\b\)?/gi,
    replacement: (m: string) => {
      const word = m.replace(/[()]/g, '').toLowerCase();
      if (HEX_WORD_EXCEPTIONS.has(word)) return m;
      return sys('hash');
    },
  },

  // 8. Stripe / service-prefixed IDs
  {
    name: 'stripe-id',
    pattern: /\b(pi|ch|sub|cus|prod|price|in|re|evt|req|tok|card|ba|acct)_[A-Za-z0-9]{14,}\b/g,
    replacement: (m: string, prefix: string) => STRIPE_PREFIX_MAP[prefix] ?? sys('service ID'),
  },

  // 9. IPv6 addresses (before IPv4)
  {
    name: 'ipv6',
    pattern: /\b(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?:%[a-z0-9]+)?(?:\/\d+)?\b/gi,
    replacement: sys('IPv6 address'),
  },

  // 10. IPv4 addresses
  {
    name: 'ipv4',
    pattern: /\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3}(?:\/\d{1,2})?\b/g,
    replacement: sys('IP address'),
  },

  // 11. MAC addresses
  {
    name: 'mac',
    pattern: /\b[0-9a-f]{2}(?::[0-9a-f]{2}){5}\b|\b[0-9a-f]{2}(?:-[0-9a-f]{2}){5}\b|\b[0-9a-f]{4}(?:\.[0-9a-f]{4}){2}\b/gi,
    replacement: sys('MAC address'),
  },

  // 12. Hex dump lines (collapse block to single spoken form)
  {
    name: 'hex-dump',
    pattern: /(?:^[0-9a-f]{4,}[:\s]+(?:[0-9a-f]{2} ?)+.*$\n?){2,}/gim,
    replacement: sys('hex dump'),
  },

  // 13. Large numeric IDs (11+ digits)
  {
    name: 'large-numeric-id',
    pattern: /\b\d{11,}\b/g,
    replacement: (m: string) => {
      if (m.length >= 16 && m.length <= 19) return sys('snowflake ID');
      if (m.length >= 20) return sys('large numeric ID');
      return sys('large ID');
    },
  },

  // 14. Base64 blobs (40+ chars, with entropy check for 40–80 char range)
  {
    name: 'base64',
    pattern: /(?:data:[a-z]+\/[a-z]+;base64,)?[A-Za-z0-9+\/]{40,}={0,2}/g,
    replacement: (m: string) => {
      // Entropy check for 40–80 char matches: require mixed case + digit
      if (m.length <= 80) {
        const hasUpper = /[A-Z]/.test(m);
        const hasLower = /[a-z]/.test(m);
        const hasDigit = /[0-9]/.test(m);
        if (!hasUpper || !hasLower || !hasDigit) return m; // skip — likely a word
      }
      if (m.startsWith('data:')) return sys('base64 image');
      return sys('base64 data');
    },
  },

  // 15. Generic long hex (16+ chars, non-SHA lengths)
  {
    name: 'hex-generic',
    pattern: /\b[0-9a-f]{16,}\b/gi,
    replacement: (m: string) => {
      if (m.length === 32) return sys('MD5 hash');
      if (m.length === 64) return sys('SHA-256 hash');
      if (m.length === 128) return sys('SHA-512 hash');
      return sys('hex string');
    },
  },
];

function blobToSpeech(text: string): string {
  for (const { pattern, replacement } of BLOB_RULES) {
    text = text.replace(pattern, replacement as any);
  }
  return text;
}

function extractSpeakableText(text: string): string {
  // Convert tables before cleaning strips the | characters
  text = tablesToSpeech(text);

  // Strip markdown formatting for speech
  const clean = (s: string) => s
    .replace(/```(?:\w+\n)?([\s\S]*?)```/g, (_m, code: string) => code.trim()) // code blocks → content only
    .replace(/`([^`]+)`/g, (_m, code: string) => {  // inline code → spoken form
      const c = code.trim();
      // File paths → spoken basename with extension
      if (c.includes('/')) {
        const base = c.split('/').filter(Boolean).pop() || 'file';
        return filenameToSpeech(base);
      }
      // CLI flags → strip
      if (c.startsWith('--') || c.startsWith('-')) return '';
      // Filenames (has a dot with extension) → spoken form
      if (/\.\w{1,5}$/.test(c)) return filenameToSpeech(c);
      // Short identifiers → keep as-is (variable names, function names)
      return c;
    })
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // links → text
    .replace(/\[([^\]]+)\]/g, 'box wrapped $1') // [thing] → box wrapped thing
    // PAI format emojis → spoken equivalents
    .replace(/🗒️?\s*TASK:\s*/g, 'Task: ')
    .replace(/🔄\s*ITERATION on:\s*/g, 'Follow-up: ')
    .replace(/📃\s*CONTENT:\s*/g, 'Content: ')
    .replace(/🔧\s*CHANGE:\s*/g, 'Changes: ')
    .replace(/✅\s*VERIFY:\s*/g, 'Verified: ')
    .replace(/📋\s*SUMMARY:\s*/g, 'Summary: ')
    .replace(/🖊️?\s*STORY:\s*/g, 'Story: ')
    .replace(/🗣\s*VOICE:\s*/g, '')           // meta line, don't speak
    .replace(/[═━]{2,}/g, '')                 // divider bars
    .replace(/[🗒📃🔧✅📋🖊🗣🔄]/g, '')       // any remaining format emojis
    .replace(/[#*_~>|]/g, '')                 // markdown decoration
    .replace(/\n{3,}/g, '\n\n')              // collapse whitespace
    .replace(/→/g, 'to')
    .replace(/(?<=\w)\/(?=\w)/g, ' slash ') // word/word → word slash word
    .replace(/([a-z])([A-Z])/g, '$1 $2')   // camelCase → camel case (splits before each uppercase run)
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2') // ACRONYMWord → ACRONYM Word
    .replace(/\b(\d+(?:\.\d+)?)ms\b/g, '$1 milliseconds')
    .replace(/\b(\d+(?:\.\d+)?)s\b/g, '$1 seconds')
    .replace(/\b(\d+(?:\.\d+)?)m\b/g, '$1 minutes')
    .replace(/\b(\d+(?:\.\d+)?)h\b/g, '$1 hours')
    .replace(/  +/g, ' ')                    // collapse double spaces from removals
    .trim();

  // Clean the entire response — no truncation, no section extraction.
  // The daemon handles long text fine (streams clause by clause).
  // blobToSpeech runs after clean() so markdown is already stripped.
  return blobToSpeech(clean(text));
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

  // Pipe through speak-summarize for pronunciation rewrites, then enqueue
  const rewritten = spawnSync('python3', [SPEAK_SUMMARIZE], {
    input: speakable,
    encoding: 'utf8',
    timeout: 5000,
  });
  const finalText = rewritten.stdout?.trim() || speakable;

  spawnSync(SPEAK, ['--caller', 'da', finalText], {
    timeout: 5000,
    stdio: 'ignore',
  });

  process.exit(0);
}

main().catch(() => process.exit(0));

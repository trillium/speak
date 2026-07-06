"""
split_inspect.py — Visualize clause splits on the test string.

Shows exactly where split_clauses() breaks the text, what triggered each
split, and stats per clause. No synthesis, no audio, no DB.

Usage:
    python test/split_inspect.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from speakd.text import SPLIT_CHARS, split_clauses

TEST_TEXT = """\
Found it. Here's the full picture:

b69642e (02:58 AM) — commit message says it added T8 files, but the actual diff shows it deleted night-summary.json, T11, T12, and T8 files. pai-bridge faithfully committed deletion events for files that got removed from disk between 02:55 and 02:58. That's the direct cause.

But the multi-machine corruption vector is CheckpointPerISC in CheckpointPerISC.hook dot ts:113 — it runs git add -A on .claude without pulling first. Here's the danger:

1. mini2 writes files X, Y, Z → pai-bridge commits them, pushes
2. MacBook's CheckpointPerISC fires (ISC transition) → runs git add -A WITHOUT pulling first
3. MacBook doesn't have X, Y, Z locally yet (pull hasn't happened)
4. git add -A stages X, Y, Z as deletions
5. MacBook commits + pushes → X, Y, Z gone from bare repo
6. mini2's next pull deletes them locally

The fix is to add a git pull --ff-only before the git add -A in CheckpointPerISC. Want me to patch that?

Also worth noting: the immediate NightShift problem (results deleted) is already fixed by the path move. But CheckpointPerISC is still a live landmine for anything inside .claude that one machine writes and the other doesn't immediately have.\
"""

def get_split_char(text: str) -> str:
    return text[-1] if text and text[-1] in SPLIT_CHARS else ""

def split_char_name(ch: str) -> str:
    names = {
        ".": "period", "!": "exclaim", "?": "question",
        ",": "comma", ";": "semicolon", ":": "colon",
        "-": "dash", "\u2014": "em-dash", "\u2013": "en-dash",
    }
    return names.get(ch, repr(ch))

def main():
    clauses = split_clauses(TEST_TEXT)

    print(f"Input: {len(TEST_TEXT)} chars")
    print(f"Clauses: {len(clauses)}")
    print()

    flags = []
    for i, clause in enumerate(clauses):
        ch = get_split_char(clause)
        words = len(clause.split())
        chars = len(clause)

        # Flag clauses that look problematic
        problems = []
        if chars < 8:
            problems.append("VERY SHORT")
        if words == 1:
            problems.append("SINGLE WORD")
        if "\n" in clause:
            problems.append("HAS NEWLINE")

        flag_str = f"  ⚠ {', '.join(problems)}" if problems else ""
        split_str = f"[{split_char_name(ch)}]" if ch else "[none]"

        print(f"  {i:02d}  {split_str:<12}  {chars:>3}ch  {words:>2}w  {clause!r}{flag_str}")

    # Summary
    short = [c for c in clauses if len(c) < 8]
    single = [c for c in clauses if len(c.split()) == 1]
    newlined = [c for c in clauses if "\n" in c]

    if short or single or newlined:
        print()
        print(f"  ⚠  {len(short)} very short (<8 chars), {len(single)} single-word, {len(newlined)} with embedded newlines")

if __name__ == "__main__":
    main()

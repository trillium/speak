# speak demo

Copy-paste commands one at a time (or a whole section) into your terminal.

## Setup

Make sure the daemon is running:

```bash
speak --daemon
```

Check the log file for timing data after each test:

```bash
cat /tmp/speak-$USER.sock.log | grep "q#"
```

## 1. Basic enqueue

```bash
speak --enqueue "Hello, welcome to the speak demo."
```

## 2. Sequential playback (no gaps between items)

```bash
speak --enqueue "The quick brown fox jumps over the lazy dog."
speak --enqueue "She sells sea shells by the sea shore."
speak --enqueue "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"
```

## 3. Pipe mode

```bash
echo "This text was piped through standard input." | speak --enqueue
```

## 4. Different voices

```bash
speak --enqueue --voice am_adam "This is Adam, an American male voice."
speak --enqueue --voice af_nova "This is Nova, an American female voice."
speak --enqueue --voice bm_daniel "This is Daniel, a British male voice."
```

## 5. Speed comparison

Default speed is 1.26. These should sound obviously different:

```bash
speak --enqueue --speed 1.0 "This is speed one point oh. It should be noticeably slower."
speak --enqueue --speed 1.8 "This is speed one point eight. It should be noticeably faster."
```

## 6. Long multi-clause passage

```bash
speak --enqueue "Here is a longer passage to demonstrate multi-clause streaming. The daemon splits this at punctuation marks, synthesizes each clause, and feeds them into the player. You should hear natural pauses at commas and periods, just like normal speech."
```

## 7. Queue management

Check what's playing and what's pending:

```bash
speak --queue
```

Skip the currently playing item:

```bash
speak --skip
```

Clear all pending items:

```bash
speak --clear
```

Stop everything (clear + skip):

```bash
speak --clear && speak --skip
```

## 8. Sync mode (blocking)

This blocks until playback finishes — the original behavior:

```bash
speak "This blocks until done."
```

## 9. Check the logs

Every queued item logs timing data to the daemon log:

```bash
cat /tmp/speak-$USER.sock.log | tail -20
```

Fields per clause:
- **SYN** = full synthesis, **HIT** = cache hit, **ASM** = assembled from word cache
- **synth** = synthesis time (ms)
- **write** = time writing to play stdin (high = play buffer full, we're ahead of playback)
- **audio** = seconds of audio produced
- **speed** = speed value used
- **wait** = time waiting for playback to catch up before next item

## 10. Restart fresh

Clear the cache and restart the daemon:

```bash
speak --stop
rm -rf /tmp/speak-cache-$USER/
speak --daemon
```

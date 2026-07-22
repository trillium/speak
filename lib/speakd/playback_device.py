"""Audio output via sounddevice (PortAudio).

sounddevice.RawOutputStream opens the device synchronously, so when start()
returns, the device IS ready. No silence priming needed.

Following the system default across changes
-------------------------------------------
PortAudio snapshots the device list (and the default device) at
``Pa_Initialize`` time and does NOT observe a later change to the macOS default
output device. So once the daemon opens ``device=None`` it stays pinned to
whatever was default at process start, even after the user switches output
device in System Settings / Control Center. Proven empirically: with a stream
process alive, flipping the CoreAudio default and re-reading
``sd.query_devices(kind='output')`` still reports the OLD device until
``sd._terminate(); sd._initialize()`` refreshes PortAudio's snapshot.

Fix: follow the CURRENT default on every playback request. A cheap native
CoreAudio probe (``_current_default_output_uid``, ~0.1ms) tells us the live
default's UID without touching PortAudio. When it differs from the device the
open stream is bound to, we drop the stream and reopen — refreshing PortAudio's
snapshot first so ``device=None`` resolves to the new default. When the default
hasn't moved we keep the open stream, so steady-state playback stays seamless
and never pays the ~260ms reopen cost.
"""

import asyncio
import struct
import sys
import time

import sounddevice as sd

from .config import SAMPLE_RATE

# Write PCM in small chunks so PortAudio backpressure naturally paces us.
WRITE_CHUNK_BYTES = int(SAMPLE_RATE * 2 * 0.25)  # 0.25s of audio per write

# Always-present built-in output. When the configured/system-default device
# can't be opened (classic case: the system default is a disconnected
# Bluetooth headset, which PortAudio still reports as default and then fails to
# open with -9986), we fall back to this so audio never silently dies. Matched
# by substring, so it survives shifting device indices.
FALLBACK_DEVICE = "MacBook Pro Speakers"


# --- CoreAudio: live default-output UID, independent of PortAudio's snapshot ---

def _init_coreaudio():
    """Build a fast probe of the current macOS default output device UID.

    Returns a zero-arg callable returning the UID string (e.g.
    "70-BF-92-36-BC-02:output"), or None when it can't be determined. On any
    non-Darwin platform or load failure, returns a callable that always yields
    None so device-change detection simply no-ops (behavior degrades to
    "reopen only when the stream dies").
    """
    if sys.platform != "darwin":
        return lambda: None
    try:
        import ctypes

        ca = ctypes.CDLL(
            "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"
        )
        cf = ctypes.CDLL(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )

        def fourcc(s):
            return struct.unpack(">I", s.encode())[0]

        class Addr(ctypes.Structure):
            _fields_ = [
                ("sel", ctypes.c_uint32),
                ("scope", ctypes.c_uint32),
                ("elem", ctypes.c_uint32),
            ]

        kSystemObject = 1
        kDefaultOutput = fourcc("dOut")  # kAudioHardwarePropertyDefaultOutputDevice
        kScopeGlobal = fourcc("glob")    # kAudioObjectPropertyScopeGlobal
        kDeviceUID = fourcc("uid ")      # kAudioDevicePropertyDeviceUID

        ca.AudioObjectGetPropertyData.restype = ctypes.c_int32
        cf.CFStringGetCString.restype = ctypes.c_bool
        kCFStringEncodingUTF8 = 0x08000100

        def probe():
            try:
                # 1) current default output device id
                addr = Addr(kDefaultOutput, kScopeGlobal, 0)
                dev = ctypes.c_uint32(0)
                sz = ctypes.c_uint32(4)
                rc = ca.AudioObjectGetPropertyData(
                    kSystemObject, ctypes.byref(addr), 0, None,
                    ctypes.byref(sz), ctypes.byref(dev),
                )
                if rc != 0 or dev.value == 0:
                    return None
                # 2) that device's persistent UID string
                addr2 = Addr(kDeviceUID, kScopeGlobal, 0)
                ref = ctypes.c_void_p(0)
                sz2 = ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
                rc = ca.AudioObjectGetPropertyData(
                    dev.value, ctypes.byref(addr2), 0, None,
                    ctypes.byref(sz2), ctypes.byref(ref),
                )
                if rc != 0 or not ref.value:
                    return None
                buf = ctypes.create_string_buffer(512)
                ok = cf.CFStringGetCString(ref, buf, 512, kCFStringEncodingUTF8)
                cf.CFRelease(ref)
                return buf.value.decode("utf-8", "replace") if ok else None
            except Exception:
                return None

        return probe
    except Exception:
        return lambda: None


_current_default_output_uid = _init_coreaudio()


class AudioOutputStream:
    """Manages a sounddevice RawOutputStream for continuous PCM streaming."""

    def __init__(self, subscriber_manager=None, device=None):
        self._stream: sd.RawOutputStream | None = None
        self._subscriber_manager = subscriber_manager
        self._device = device  # int index, str name substring, or None for default
        # CoreAudio UID of the default device the open stream is bound to, when
        # following the system default. None means "not bound to the current
        # default" — either no stream, or parked on the fallback — which forces
        # the next request to re-attempt the real current default (non-sticky
        # fallback).
        self._bound_uid: str | None = None

    @property
    def is_alive(self) -> bool:
        return self._stream is not None and self._stream.active

    async def set_device(self, device) -> None:
        """Switch to a different audio device. Kills current stream so next write reopens."""
        self._device = device
        self._bound_uid = None
        await self.kill()

    async def refresh_default(self) -> None:
        """Re-point at the CURRENT system default before a playback request.

        No-op when a specific device is pinned (device is not None) or when the
        open stream is already bound to the live default. Otherwise drops the
        stream so the next write reopens on the current default. Cheap: one
        ~0.1ms native probe in the common (unchanged) case.
        """
        if self._device is not None:
            return  # explicit pin overrides default-following
        cur = _current_default_output_uid()
        if cur is None:
            return  # can't determine the default; leave the stream as-is
        if self._stream is not None and self._stream.active and cur == self._bound_uid:
            return  # already on the current default — keep the stream seamless
        # Default moved (or we're parked on the fallback, or no stream): drop the
        # stale stream so the next write reopens fresh on the current default.
        if self._stream is not None:
            await self.kill()
        self._bound_uid = None

    async def kill(self, force: bool = False) -> None:
        """Shut down the audio stream.

        force=True: abort() — discard buffer, immediate stop (for skip).
        force=False: stop() — drain buffer, then close.
        """
        if self._stream is not None:
            try:
                loop = asyncio.get_event_loop()
                if force:
                    await loop.run_in_executor(None, self._stream.abort)
                else:
                    await loop.run_in_executor(None, self._stream.stop)
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    async def ensure_running(self) -> None:
        if self._stream is not None and self._stream.active:
            return
        # Clean up stale stream
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        loop = asyncio.get_event_loop()

        # When following the system default, capture the live default UID now so
        # we can record what the stream is bound to after it opens.
        following_default = self._device is None
        default_uid = _current_default_output_uid() if following_default else None

        # Try the configured device first (None = system default). If it can't
        # be opened, fall back to the built-in speakers so a dead default
        # device (e.g. a disconnected Bluetooth headset) never silences speak.
        candidates = [self._device]
        if self._device != FALLBACK_DEVICE:
            candidates.append(FALLBACK_DEVICE)

        def _try_open(cand):
            s = sd.RawOutputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                latency="low",
                device=cand,
            )
            s.start()
            return s

        def _open():
            # Refresh PortAudio's device snapshot so device=None resolves to the
            # CURRENT default, not the one captured at process start. Required:
            # PortAudio caches devices at initialize and never re-reads the
            # macOS default on its own.
            if following_default:
                try:
                    sd._terminate()
                    sd._initialize()
                except Exception:
                    pass
            last_err: Exception | None = None
            for i, cand in enumerate(candidates):
                # Opening a device right after a default switch can transiently
                # fail (CoreAudio AUHAL -10851 while the device is mid-
                # transition). Give the primary candidate one short retry before
                # dropping to the fallback, so a device change doesn't spuriously
                # bounce one utterance to the built-in speakers.
                attempts = 2 if i == 0 else 1
                for attempt in range(attempts):
                    try:
                        return _try_open(cand), cand
                    except Exception as e:  # noqa: BLE001 — retry, then next candidate
                        last_err = e
                        if attempt + 1 < attempts:
                            time.sleep(0.15)
            raise last_err

        self._stream, opened = await loop.run_in_executor(None, _open)
        if opened == self._device:
            # Opened the requested device (the system default when following it).
            self._bound_uid = default_uid if following_default else None
        else:
            # Fell back off the requested/default device. Leave _bound_uid None
            # so the next request re-attempts the real current default rather
            # than staying parked on the fallback.
            self._bound_uid = None
            print(
                f"speak-daemon: audio device {self._device!r} unavailable, "
                f"fell back to {opened!r}",
                file=sys.stderr,
            )

    async def write_pcm(self, pcm: bytes, skip_flag_fn=None) -> float:
        """Write PCM to the audio device in small chunks for backpressure pacing.

        Returns duration in seconds of what was written.

        skip_flag_fn: optional callable returning bool, checked each chunk
        to allow early exit.
        """
        await self.ensure_running()
        loop = asyncio.get_event_loop()
        offset = 0
        while offset < len(pcm):
            if skip_flag_fn and skip_flag_fn():
                break
            chunk = pcm[offset : offset + WRITE_CHUNK_BYTES]
            try:
                await loop.run_in_executor(None, self._stream.write, chunk)
            except sd.PortAudioError as e:
                print(f"speak-daemon: PortAudio error, reopening: {e}", file=sys.stderr)
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
                await self.ensure_running()
                await loop.run_in_executor(None, self._stream.write, chunk)
            if self._subscriber_manager:
                self._subscriber_manager.broadcast_audio(chunk)
            offset += len(chunk)
        n_samples = len(pcm) // 2  # int16
        return n_samples / SAMPLE_RATE

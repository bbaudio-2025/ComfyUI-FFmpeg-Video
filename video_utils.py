"""
Video utility functions for FFmpeg-based video processing in ComfyUI.

This module provides helper functions to:
- Extract file paths from ComfyUI VIDEO type inputs (VideoInput objects)
- Probe video/audio metadata using ffprobe
- Save ComfyUI AUDIO type inputs to WAV files
- Create VIDEO type outputs from file paths
- Manage temporary files in ComfyUI's temp directory
"""

import os
import io
import math
import wave
import uuid
import tempfile
import subprocess

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

try:
    import av as _av
except ImportError:
    _av = None

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from comfy_api.latest import InputImpl
    _HAS_V3_API = True
except ImportError:
    _HAS_V3_API = False


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------

def check_ffmpeg():
    """Verify that ffmpeg-python and the ffmpeg binary are available."""
    if ffmpeg is None:
        raise ImportError(
            "ffmpeg-python is not installed. Install it with: pip install ffmpeg-python"
        )
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg binary is not installed or not found in PATH. "
            "Please install FFmpeg: https://ffmpeg.org/download.html"
        )
    except subprocess.CalledProcessError:
        pass  # some builds return non-zero for -version


# ---------------------------------------------------------------------------
# Temp file management
# ---------------------------------------------------------------------------

def get_temp_dir():
    """Return ComfyUI's temp directory (or the OS temp dir as fallback)."""
    if folder_paths is not None:
        return folder_paths.get_temp_directory()
    return tempfile.gettempdir()


def get_temp_path(suffix=".mp4"):
    """Return a unique file path inside the temp directory."""
    temp_dir = get_temp_dir()
    os.makedirs(temp_dir, exist_ok=True)
    return os.path.join(temp_dir, f"ffmpeg_video_{uuid.uuid4().hex}{suffix}")


# ---------------------------------------------------------------------------
# Video path extraction
# ---------------------------------------------------------------------------

def _video_trim_window(video_input):
    """
    Read the lazy trim window ``(start_time, duration)`` carried by a ComfyUI
    VIDEO type input, in seconds.

    ComfyUI video transform nodes such as "Crop Video (Temporal)" are *fully
    lazy*: they keep the original source file untouched and only record a
    trim offset (``as_trimmed``).  Returns ``(0.0, 0.0)`` when the input is a
    plain path or exposes no active trim window.
    """
    start_time = 0.0
    duration = 0.0
    if hasattr(video_input, "get_active_trim_window"):
        try:
            st, du = video_input.get_active_trim_window()
            start_time = float(st or 0.0)
            duration = float(du or 0.0)
        except Exception:
            start_time, duration = 0.0, 0.0
    return start_time, duration


def resolve_video(video_input):
    """
    Resolve a ComfyUI VIDEO type input to a concrete file on disk plus the
    lazy trim window that must be applied when reading that file.

    Returns ``(path, start_time, duration)``:

      - ``path``          — the concrete file holding the pixels.
      - ``start_time``    — seconds to seek/start from (0 when no lazy trim).
      - ``duration``      — seconds to keep (0 when no lazy trim).

    While ``get_video_path`` only extracts the path, nodes that *re-encode*
    the video (e.g. resize) must honour the trim window: on the raw
    stream-source fast path the resolved file is the *untrimmed* original, so
    the offset is meaningful.  Fallback paths that already rebuild a trimmed
    file report ``(0.0, 0.0)`` to avoid double-trimming.
    """
    # Already a file path (no lazy trim metadata available)
    if isinstance(video_input, str):
        if not os.path.exists(video_input):
            raise FileNotFoundError(f"Video file not found: {video_input}")
        return video_input, 0.0, 0.0

    # VideoInput abstract object — fast path via stream source (raw source file)
    if hasattr(video_input, "get_stream_source"):
        try:
            source = video_input.get_stream_source()
            if isinstance(source, str):
                st, du = _video_trim_window(video_input)
                return source, st, du
            if isinstance(source, (io.BytesIO, io.BufferedReader)):
                tmp = get_temp_path()
                with open(tmp, "wb") as f:
                    f.write(source.read())
                if os.path.getsize(tmp) > 0:
                    st, du = _video_trim_window(video_input)
                    return tmp, st, du
                os.remove(tmp)  # empty buffer — fall through
        except Exception:
            pass  # fall through to in-memory rebuild

    # In-memory video: rebuild the file from components with a compatible
    # audio codec (PCM works at any sample rate). The rebuild already applies
    # any trim window, so the resulting file is pre-trimmed.
    if hasattr(video_input, "get_components"):
        tmp = get_temp_path()
        save_in_memory_video(video_input, tmp)
        return tmp, 0.0, 0.0

    # Fallback: save_to (also applies the trim window while saving)
    if hasattr(video_input, "save_to"):
        tmp = get_temp_path()
        video_input.save_to(tmp)
        return tmp, 0.0, 0.0

    raise ValueError(
        f"Cannot extract file path from video input of type {type(video_input)}"
    )


def get_video_path(video_input):
    """
    Extract a file-system path from a ComfyUI VIDEO type input.

    Handles:
      - Plain string (file path)
      - VideoInput objects (via get_stream_source / save_to)
      - In-memory videos: falls back to rebuilding the file from the
        in-memory components when ``get_stream_source`` fails (e.g. for
        old mono-PCM audio whose sample rate the AAC encoder ComfyUI
        always uses cannot handle).
    """
    path, _start_time, _duration = resolve_video(video_input)
    return path


def _recover_fps_from_audio(n_frames, audio_info):
    """
    Estimate the real frame rate from the audio track duration.

    ``audio_info`` is ``(num_samples, sample_rate)`` or ``None``.
    Because the audio duration is reliable even when the container's
    reported frame rate is garbage, ``fps = frames / audio_duration``
    recovers the true value.  Returns ``None`` when not derivable.
    """
    if audio_info is None:
        return None
    n_samples, sample_rate = audio_info
    if n_samples <= 0 or sample_rate <= 0:
        return None
    duration = n_samples / sample_rate
    if duration <= 0:
        return None
    fps = n_frames / duration
    if 0 < fps < 100000:
        return fps
    return None


def _sanitize_sample_rate(sample_rate):
    """
    Normalize an audio sample rate to a positive ``int`` that fits PyAV's
    C ``int`` fields (AVRational / AudioFrame.sample_rate).

    Broken containers can report garbage (0, negative, or astronomically
    large) sample rates; those fall back to 44100.  Real rates such as
    the old video's 22300 Hz pass through unchanged.
    """
    try:
        sample_rate = int(sample_rate)
    except (TypeError, ValueError, OverflowError):
        return 44100
    if sample_rate <= 0 or sample_rate >= 2**31:
        return 44100
    return sample_rate


def _sanitize_frame_rate(frame_rate, n_frames=None, audio_info=None):
    """
    Normalize a frame-rate value to a positive ``Fraction`` whose
    numerator / denominator fit in a C ``int`` (PyAV AVRational).

    Some old / malformed files report a huge or non-sensical frame rate
    (e.g. ``Fraction(0, 0)`` or astronomically large ratios) which would
    otherwise crash PyAV with "Python int too large to convert to C long".
    When the reported value is unusable, the frame rate is recovered from
    the audio duration (``frames / audio_duration``), falling back to
    30 fps and clamping to 0.001–1000 fps.
    """
    from fractions import Fraction

    fr = None
    try:
        if isinstance(frame_rate, Fraction):
            fr = frame_rate
        elif frame_rate is not None:
            fr = Fraction(frame_rate)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        fr = None

    ok = (
        fr is not None
        and fr > 0
        and fr.numerator < 2**31
        and fr.denominator < 2**31
    )
    if ok:
        return fr

    fps = _recover_fps_from_audio(n_frames, audio_info)
    if fps is None:
        if fr is not None:
            try:
                fps = float(fr)
            except (TypeError, ValueError, OverflowError):
                fps = 30.0
        else:
            fps = 30.0
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    fps = max(0.001, min(fps, 1000.0))
    return Fraction(fps).limit_denominator(1000000)


def save_in_memory_video(video_input, output_path):
    """
    Rebuild an in-memory ComfyUI VIDEO input as a file on disk.

    ComfyUI's own ``save_to`` always encodes audio as AAC, whose encoder
    fails to open for very low sample rates (below ~7350 Hz) found in
    some old mono-PCM videos.  This fallback encodes audio as PCM
    (``pcm_s16le``), which is accepted at any sample rate, and video as
    H.264 / yuv420p.
    """
    if _av is None:
        raise RuntimeError(
            "PyAV is not installed. Install it with: pip install av"
        )

    components = video_input.get_components()
    images = components.images
    audio = components.audio
    frame_rate = components.frame_rate

    # --- video tensor -> uint8 numpy [N, H, W, 3] ----------------------
    if hasattr(images, "cpu"):
        images = images.cpu().numpy()
    elif not isinstance(images, np.ndarray):
        images = np.array(images)

    images = np.ascontiguousarray(images)
    n_frames, height, width, n_ch = images.shape
    if images.dtype.kind == "f":
        images = (np.clip(images, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        images = images.astype(np.uint8)

    # --- prepare audio data (layout / sample count) ---------------------
    audio_data = None  # (waveform_np, sample_rate, layout)
    if audio is not None:
        waveform = audio["waveform"]
        sample_rate = _sanitize_sample_rate(audio["sample_rate"])
        if hasattr(waveform, "cpu"):
            waveform = waveform.cpu().numpy()
        elif not isinstance(waveform, np.ndarray):
            waveform = np.array(waveform)
        if waveform.ndim == 3:
            waveform = waveform[0]

        n_audio_ch = waveform.shape[0]
        layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(
            n_audio_ch, "stereo"
        )
        if n_audio_ch not in (1, 2, 6):
            waveform = waveform[:2] if n_audio_ch > 2 else np.repeat(
                waveform, 2, axis=0
            )

        if waveform.shape[1] > 0:
            audio_data = (
                np.ascontiguousarray(waveform, dtype=np.float32),
                sample_rate,
                layout,
            )

    # --- normalize frame rate to a bounded positive Fraction ---------
    # PyAV stores the rate as AVRational (C int), so reject values whose
    # numerator / denominator overflow int32 (some old files report huge
    # ratios).  When unusable, recover the true fps from the audio
    # duration (e.g. the old video is really 15 fps, not the huge ratio).
    audio_info = None
    if audio_data is not None:
        audio_info = (audio_data[0].shape[1], audio_data[1])
    fr = _sanitize_frame_rate(frame_rate, n_frames=n_frames,
                              audio_info=audio_info)

    # Truncate audio to the video length (mirrors ComfyUI's save_to).
    if audio_data is not None:
        wave, sample_rate, layout = audio_data
        n_samples = int(round((sample_rate / float(fr)) * n_frames))
        if wave.shape[1] > n_samples:
            wave = wave[:, :n_samples]
        audio_data = (wave, sample_rate, layout)

    # Both streams must be added before any muxing, otherwise PyAV cannot
    # assign a time base to the audio stream.
    with _av.open(output_path, mode="w", format="mp4") as out:
        vstream = out.add_stream("h264", rate=fr)
        vstream.width = width
        vstream.height = height
        vstream.pix_fmt = "yuv420p"

        astream = None
        if audio_data is not None:
            wave, sample_rate, layout = audio_data
            astream = out.add_stream("pcm_s16le", rate=sample_rate,
                                     layout=layout)

        for i in range(n_frames):
            frame = _av.VideoFrame.from_ndarray(images[i], format="rgb24")
            frame = frame.reformat(width=width, height=height, format="yuv420p")
            for packet in vstream.encode(frame):
                out.mux(packet)
        for packet in vstream.encode(None):
            out.mux(packet)

        if astream is not None:
            aframe = _av.AudioFrame.from_ndarray(
                wave, format="fltp", layout=layout
            )
            aframe.sample_rate = sample_rate
            aframe.pts = 0
            for packet in astream.encode(aframe):
                out.mux(packet)
            for packet in astream.encode(None):
                out.mux(packet)


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------

def _parse_frame_rate(rate_str):
    """Parse a frame-rate string like '30/1' or '30000/1001' to float."""
    if not rate_str or rate_str == "0/0":
        return 0.0
    if "/" in rate_str:
        num, den = rate_str.split("/")
        num = float(num)
        den = float(den)
        if den == 0:
            return 0.0
        return num / den
    return float(rate_str)


def get_video_info(video_path):
    """
    Probe a video file and return a dict with:
      width, height, fps, duration, frame_count, codec, sar
    """
    probe = ffmpeg.probe(video_path)

    video_stream = None
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise ValueError(f"No video stream found in: {video_path}")

    fps = _parse_frame_rate(video_stream.get("r_frame_rate", "0/1"))
    duration = float(
        video_stream.get("duration")
        or probe.get("format", {}).get("duration", 0)
    )

    frame_count = int(video_stream.get("nb_frames", 0))
    if frame_count == 0 and fps > 0 and duration > 0:
        frame_count = int(round(duration * fps))

    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": fps,
        "duration": duration,
        "frame_count": frame_count,
        "codec": video_stream.get("codec_name", "unknown"),
        "sar": video_stream.get("sample_aspect_ratio") or "1:1",
    }


def get_audio_info(audio_path):
    """
    Probe an audio file and return a dict with:
      duration, sample_rate, channels
    """
    probe = ffmpeg.probe(audio_path)

    audio_stream = None
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "audio":
            audio_stream = stream
            break

    if audio_stream is None:
        raise ValueError(f"No audio stream found in: {audio_path}")

    duration = float(
        audio_stream.get("duration")
        or probe.get("format", {}).get("duration", 0)
    )

    return {
        "duration": duration,
        "sample_rate": int(audio_stream.get("sample_rate", 44100)),
        "channels": int(audio_stream.get("channels", 2)),
    }


def _bitrate_kbps(bit_rate):
    """Convert an ffprobe bit-rate string (bps) to integer kB/s."""
    if not bit_rate:
        return 0
    try:
        return int(float(bit_rate) / 1000)
    except (TypeError, ValueError):
        return 0


_INTEGER_SAMPLE_FORMAT_BITS = {
    "u8": 8,
    "u8p": 8,
    "s16": 16,
    "s16p": 16,
    "s24": 24,
    "s24p": 24,
    "s32": 32,
    "s32p": 32,
    "s64": 64,
    "s64p": 64,
}

_LOSSY_CODEC_BIT_DEPTHS = {
    "aac": 16,
    "aac_latm": 16,
    "mp3": 16,
    "mp2": 16,
    "opus": 16,
    "vorbis": 16,
    "ac3": 16,
    "eac3": 16,
    "wmav2": 16,
    "amr_nb": 16,
    "amr_wb": 16,
    "speex": 16,
    "cook": 16,
}


def _audio_bit_depth(stream):
    """
    Best-effort estimate of an audio stream's bit depth.

    1. Use the exact per-sample values ffprobe reports for PCM / FLAC /
       other lossless codecs (``bits_per_raw_sample`` / ``bits_per_sample``).
    2. Fall back to the integer ``sample_fmt`` when available.
    3. For lossy codecs (AAC, MP3, Opus, ...) no bit depth is stored in
       the file, so report their conventional source depth (16-bit);
       ffprobe's decoded ``sample_fmt`` (e.g. ``fltp`` = float32) is a
       decoder output and would be misleading.

    Returns ``0`` when nothing usable is found.
    """
    for key in ("bits_per_raw_sample", "bits_per_sample"):
        value = stream.get(key)
        if value in (None, "", "N/A"):
            continue
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value

    sample_fmt = stream.get("sample_fmt", "")
    if sample_fmt in _INTEGER_SAMPLE_FORMAT_BITS:
        return _INTEGER_SAMPLE_FORMAT_BITS[sample_fmt]

    codec = stream.get("codec_name", "")
    return _LOSSY_CODEC_BIT_DEPTHS.get(codec, 0)


def get_video_details(video_path):
    """
    Probe a video file and return detailed metadata about its video and
    audio streams.

    Returns
    -------
    dict
        width, height, fps, duration, frame_count, video_codec,
        video_bitrate_kbps, audio_channels, audio_sample_rate,
        audio_bit_depth, audio_codec, audio_bitrate_kbps.

        When the file has no audio stream, the ``audio_*`` fields are
        ``0`` / empty strings.  Bitrates are averaged over the whole
        file in kB/s (kilobits per second).
    """
    probe = ffmpeg.probe(video_path)

    video_stream = None
    audio_stream = None
    for stream in probe.get("streams", []):
        ctype = stream.get("codec_type")
        if ctype == "video" and video_stream is None:
            video_stream = stream
        elif ctype == "audio" and audio_stream is None:
            audio_stream = stream
        if video_stream is not None and audio_stream is not None:
            break

    if video_stream is None:
        raise ValueError(f"No video stream found in: {video_path}")

    fps = _parse_frame_rate(video_stream.get("r_frame_rate", "0/1"))
    duration = float(
        video_stream.get("duration")
        or probe.get("format", {}).get("duration", 0)
    )

    frame_count = int(video_stream.get("nb_frames", 0))
    if frame_count == 0 and fps > 0 and duration > 0:
        frame_count = int(round(duration * fps))

    # --- video bitrate --------------------------------------------------
    # Prefer the per-stream bit_rate; fall back to the container total
    # (minus the audio bitrate when one is present).
    format_info = probe.get("format", {})
    video_bitrate = video_stream.get("bit_rate")
    if not video_bitrate and format_info.get("bit_rate"):
        total_bitrate = format_info["bit_rate"]
        audio_bitrate = audio_stream.get("bit_rate") if audio_stream else None
        if audio_bitrate:
            try:
                video_bitrate = str(
                    max(0, int(float(total_bitrate)) - int(float(audio_bitrate)))
                )
            except (TypeError, ValueError):
                video_bitrate = total_bitrate
        else:
            video_bitrate = total_bitrate

    # --- audio fields ----------------------------------------------------
    has_audio = audio_stream is not None
    audio_channels = 0
    audio_sample_rate = 0
    audio_bit_depth = 0
    audio_codec = ""
    audio_bitrate_kbps = 0

    if has_audio:
        audio_channels = int(audio_stream.get("channels", 0))
        audio_sample_rate = int(audio_stream.get("sample_rate", 0))
        audio_bit_depth = _audio_bit_depth(audio_stream)
        audio_codec = audio_stream.get("codec_name", "unknown")
        audio_bitrate_kbps = _bitrate_kbps(audio_stream.get("bit_rate"))

    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": fps,
        "duration": duration,
        "frame_count": frame_count,
        "video_codec": video_stream.get("codec_name", "unknown"),
        "video_bitrate_kbps": _bitrate_kbps(video_bitrate),
        "audio_channels": audio_channels,
        "audio_sample_rate": audio_sample_rate,
        "audio_bit_depth": audio_bit_depth,
        "audio_codec": audio_codec,
        "audio_bitrate_kbps": audio_bitrate_kbps,
    }


# ---------------------------------------------------------------------------
# Audio save (AUDIO type -> WAV)
# ---------------------------------------------------------------------------

def save_audio_to_wav(audio_input, output_path):
    """
    Save a ComfyUI AUDIO type input to a WAV file.

    AUDIO type is a TypedDict:
        { "waveform": torch.Tensor [B, C, T], "sample_rate": int }
    """
    waveform = audio_input["waveform"]
    sample_rate = audio_input["sample_rate"]

    # Move to numpy
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu().numpy()
    elif not isinstance(waveform, np.ndarray):
        waveform = np.array(waveform)

    # [B, C, T] -> [C, T]
    if waveform.ndim == 3:
        waveform = waveform[0]

    # Determine channel count
    if waveform.ndim == 2:
        n_channels = waveform.shape[0]
    else:
        n_channels = 1
        waveform = waveform[np.newaxis, :]

    # Clamp and convert to int16
    waveform = np.clip(waveform, -1.0, 1.0)
    waveform = (waveform * 32767).astype(np.int16)

    # Interleave for WAV (T, C)
    if n_channels > 1:
        waveform = waveform.T

    with wave.open(output_path, "w") as wav:
        wav.setnchannels(n_channels)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(waveform.tobytes())


# ---------------------------------------------------------------------------
# Output creation
# ---------------------------------------------------------------------------

def create_video_output(file_path):
    """Wrap a file path into a ComfyUI VIDEO type output object."""
    if _HAS_V3_API:
        return InputImpl.VideoFromFile(file_path)
    return file_path


# ---------------------------------------------------------------------------
# Video audio stream detection
# ---------------------------------------------------------------------------

def get_video_audio_info(video_path):
    """
    Check whether a video file contains an audio stream.

    Returns a tuple ``(has_audio, sample_rate, channels)``.
    If no audio stream is found, returns ``(False, 44100, 2)``.
    """
    probe = ffmpeg.probe(video_path)
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "audio":
            return (
                True,
                int(stream.get("sample_rate", 44100)),
                int(stream.get("channels", 2)),
            )
    return (False, 44100, 2)


# ---------------------------------------------------------------------------
# Channel layout helpers
# ---------------------------------------------------------------------------

def _channel_layout_name(channels):
    """Map a channel count to an FFmpeg channel-layout name."""
    layouts = {
        1: "mono",
        2: "stereo",
        3: "2.1",
        4: "3.1",
        5: "4.1",
        6: "5.1",
        7: "6.1",
        8: "7.1",
    }
    return layouts.get(channels, "stereo")


# ---------------------------------------------------------------------------
# Frame extraction (video -> IMAGE tensor)
# ---------------------------------------------------------------------------

def extract_frames(video_path, start_time, duration, width, height,
                   expected_frames):
    """
    Extract a segment of frames from a video as a ComfyUI IMAGE tensor.

    Uses FFmpeg to decode the segment and pipe raw RGB24 pixels to stdout.
    The raw bytes are reshaped into ``[N, H, W, 3]`` uint8 and then
    normalised to float32 in ``[0, 1]``.

    Parameters
    ----------
    video_path : str
        Path to the source video file.
    start_time : float
        Start position in seconds (frame-accurate via ``-ss`` before input).
    duration : float
        Duration to extract in seconds.
    width, height : int
        Frame dimensions (from ``get_video_info``).
    expected_frames : int
        Expected number of frames (used to truncate excess output).

    Returns
    -------
    torch.Tensor or np.ndarray
        Shape ``[N, H, W, 3]``, dtype float32, values in ``[0, 1]``.
        Returns a torch tensor when PyTorch is available (normal ComfyUI
        runtime), otherwise a NumPy array (testing fallback).
    """
    cmd = [
        "ffmpeg",
        "-ss", f"{start_time:.6f}",
        "-i", video_path,
        "-t", f"{duration:.6f}",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-vsync", "0",
        "-",  # pipe to stdout
    ]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"FFmpeg frame extraction failed:\n{stderr}"
        )

    raw_data = proc.stdout
    frame_size = width * height * 3

    if len(raw_data) == 0:
        raise RuntimeError(
            "FFmpeg extracted 0 bytes. Check the video path and parameters."
        )

    actual_frames = len(raw_data) // frame_size
    if actual_frames == 0:
        raise RuntimeError(
            f"FFmpeg extracted {len(raw_data)} bytes, less than one frame "
            f"(expected {frame_size} bytes per frame)."
        )

    # Truncate excess frames (ffmpeg may output 1-2 extra due to rounding)
    if actual_frames > expected_frames:
        actual_frames = expected_frames

    # Reshape raw bytes → [N, H, W, 3] uint8 → float32 [0, 1]
    frames = np.frombuffer(raw_data, dtype=np.uint8,
                           count=actual_frames * frame_size)
    frames = frames.reshape(actual_frames, height, width, 3)
    frames = frames.astype(np.float32) / 255.0
    frames = np.ascontiguousarray(frames)

    if _HAS_TORCH:
        frames = torch.from_numpy(frames)

    return frames


# ---------------------------------------------------------------------------
# Audio extraction (video -> AUDIO dict)
# ---------------------------------------------------------------------------

def create_silent_audio(duration, sample_rate=44100, channels=2):
    """
    Create a silent audio segment as a ComfyUI AUDIO dict.

    Returns
    -------
    dict
        ``{"waveform": tensor [1, C, T], "sample_rate": int}``
        Waveform is zero-filled float32.
    """
    num_samples = int(round(duration * sample_rate))
    if num_samples < 1:
        num_samples = 1
    waveform = np.zeros((1, channels, num_samples), dtype=np.float32)

    if _HAS_TORCH:
        waveform = torch.from_numpy(waveform)

    return {"waveform": waveform, "sample_rate": sample_rate}


def extract_audio_segment(video_path, start_time, duration, has_audio,
                          sample_rate, channels):
    """
    Extract an audio segment from a video file.

    Pipes raw float32 PCM (f32le) from FFmpeg stdout and packages it
    as a ComfyUI AUDIO dict.  If the video has no audio (or the segment
    falls outside the audio range), silent audio is returned instead.

    The output is padded or truncated so that the waveform length matches
    the requested ``duration`` exactly.

    Parameters
    ----------
    video_path : str
        Path to the source video file.
    start_time : float
        Start position in seconds.
    duration : float
        Duration to extract in seconds.
    has_audio : bool
        Whether the video contains an audio stream.
    sample_rate : int
        Audio sample rate (from ``get_video_audio_info``).
    channels : int
        Number of audio channels (from ``get_video_audio_info``).

    Returns
    -------
    dict
        ``{"waveform": tensor [1, C, T], "sample_rate": int}``
    """
    expected_samples = int(round(duration * sample_rate))
    if expected_samples < 1:
        expected_samples = 1

    # --- no audio stream → silence ----------------------------------
    if not has_audio:
        return create_silent_audio(duration, sample_rate, channels)

    # --- extract via ffmpeg pipe ------------------------------------
    cmd = [
        "ffmpeg",
        "-ss", f"{start_time:.6f}",
        "-i", video_path,
        "-t", f"{duration:.6f}",
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-",  # pipe to stdout
    ]

    proc = subprocess.run(cmd, capture_output=True)

    # If ffmpeg fails or returns no data, fall back to silence
    if proc.returncode != 0 or len(proc.stdout) == 0:
        return create_silent_audio(duration, sample_rate, channels)

    raw_data = proc.stdout
    samples = np.frombuffer(raw_data, dtype=np.float32)

    if len(samples) == 0:
        return create_silent_audio(duration, sample_rate, channels)

    # Reshape interleaved samples → [num_samples, channels] → [C, T]
    actual_samples = len(samples) // channels
    if actual_samples == 0:
        return create_silent_audio(duration, sample_rate, channels)

    samples = samples[:actual_samples * channels]
    samples = samples.reshape(actual_samples, channels)
    samples = samples.T  # [channels, num_samples]
    samples = np.ascontiguousarray(samples, dtype=np.float32)

    # Pad with silence if shorter than expected
    if actual_samples < expected_samples:
        pad_len = expected_samples - actual_samples
        padding = np.zeros((channels, pad_len), dtype=np.float32)
        samples = np.concatenate([samples, padding], axis=1)

    # Truncate if longer than expected
    if samples.shape[1] > expected_samples:
        samples = samples[:, :expected_samples]

    # Add batch dimension → [1, C, T]
    waveform = samples[np.newaxis, :]
    waveform = np.ascontiguousarray(waveform)

    if _HAS_TORCH:
        waveform = torch.from_numpy(waveform)

    return {"waveform": waveform, "sample_rate": sample_rate}


# ---------------------------------------------------------------------------
# Resize helpers (used by the VideoResize node)
# ---------------------------------------------------------------------------
#
# These helpers mirror the semantics of KJNodes' ``ImageResizeKJv2`` +
# ``ImagePadKJ`` nodes, but build an FFmpeg filter graph instead of doing
# tensor math.  Because a filter graph is evaluated frame by frame, memory
# usage stays flat no matter how large the video is.
#
# keep_proportion modes and their FFmpeg realisation:
#   stretch        -> scale (aspect ratio ignored)
#   resize         -> scale to the aspect-preserving size that fits the target
#   total_pixels   -> scale so that w*h == width*height, aspect preserved
#   crop           -> crop to the target aspect ratio (crop_position), then scale
#   pad            -> scale (fit) + pad with a solid colour
#   pad_edge       -> scale (fit) + pad, border filled with the *mean colour*
#                     of each edge line (per frame, via a 1x1 area downscale)
#   pad_edge_pixel -> scale (fit) + pad, border filled by replicating the
#                     nearest edge pixels
#   pillarbox_blur -> scale (fit) + pad, border filled with a blurred, 20%
#                     desaturated and 65% dimmed "cover" version of the frame
#
# Everything is done with ordinary filters (crop / scale / pad / overlay /
# gblur / lutyuv), so it works with any reasonably recent FFmpeg build.

_SCALE_FLAGS = {
    "nearest-exact": "neighbor",
    "nearest": "neighbor",
    "bilinear": "bilinear",
    "area": "area",
    "bicubic": "bicubic",
    "lanczos": "lanczos",
}

_RESIZE_MODES = (
    "stretch",
    "resize",
    "crop",
    "total_pixels",
    "pad",
    "pad_edge",
    "pad_edge_pixel",
    "pillarbox_blur",
)

_COLOR_NAMES = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "silver": (192, 192, 192),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
}

# Neutral chroma value for 8-bit YUV (used by the pillarbox_blur dimming).
_YUV_NEUTRAL = 128


def parse_pad_color(value):
    """
    Parse a KJNodes-style colour string into an FFmpeg colour spec.

    Accepts ``"0, 0, 0"`` (0-255 ints), ``"0.0, 0.0, 0.0"`` (0-1 floats),
    a single value (grey), ``"#ff8800"`` / ``"0xff8800"`` / ``"ff8800"``
    and a few common colour names.
    """
    text = str(value if value is not None else "").strip()
    if not text:
        return "0x000000"

    lowered = text.lower()
    if lowered in _COLOR_NAMES:
        return "0x%02X%02X%02X" % _COLOR_NAMES[lowered]

    if lowered.startswith("#") or lowered.startswith("0x"):
        hexpart = lowered[2:] if lowered.startswith("0x") else lowered[1:]
        if len(hexpart) == 3:
            hexpart = "".join(ch * 2 for ch in hexpart)
        if len(hexpart) == 6:
            try:
                int(hexpart, 16)
            except ValueError:
                pass
            else:
                return "0x" + hexpart.upper()

    # Bare hex such as "f80" / "ff8800" — it must contain a hex letter so that
    # plain numbers like "128" are still read as a grey level.
    if len(text) in (3, 6) and all(ch in "0123456789abcdefABCDEF" for ch in text):
        if any(ch in "abcdefABCDEF" for ch in text):
            hexpart = text.upper() if len(text) == 6 else \
                "".join(ch * 2 for ch in text.upper())
            return "0x" + hexpart

    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    if len(parts) == 1:
        parts = parts * 3
    if len(parts) == 3:
        try:
            values = [float(p) for p in parts]
        except ValueError:
            raise ValueError(f"Cannot parse pad_color: {value!r}")
        if max(values) <= 1.0:
            values = [v * 255.0 for v in values]
        channels = [max(0, min(255, int(round(v)))) for v in values]
        return "0x%02X%02X%02X" % tuple(channels)

    raise ValueError(f"Cannot parse pad_color: {value!r}")


def _even_floor(value, divisor):
    """
    Round a dimension down to a multiple of ``divisor`` (at least ``divisor``).

    yuv420p frames (and libx264 in 4:2:0) require even dimensions, so the
    effective divisor is never below 2.
    """
    if divisor < 2:
        divisor = 2
    value = int(value)
    value -= value % divisor
    return max(divisor, value)


def _pad_offsets(canvas_w, canvas_h, inner_w, inner_h, position):
    """
    Mirror KJNodes' padding layout: returns ``(left, right, top, bottom)``.

    ``left``/``top`` are kept even so that chroma sub-sampled formats and the
    overlay filter stay aligned; ``right``/``bottom`` absorb the difference.
    """
    dx = max(0, int(canvas_w) - int(inner_w))
    dy = max(0, int(canvas_h) - int(inner_h))

    if position == "top":
        left, top = dx // 2, 0
    elif position == "bottom":
        left, top = dx // 2, dy
    elif position == "left":
        left, top = 0, dy // 2
    elif position == "right":
        left, top = dx, dy // 2
    else:  # center (and any unknown value)
        left, top = dx // 2, dy // 2

    left -= left % 2
    top -= top % 2
    return left, dx - left, top, dy - top


def build_resize_graph(stream, src_w, src_h, width, height,
                       upscale_method="lanczos", keep_proportion="resize",
                       crop_position="center", pad_color="0, 0, 0",
                       divisible_by=2):
    """
    Build an FFmpeg filter chain that resizes ``stream``.

    Parameters mirror KJNodes' ``ImageResizeKJv2``:
      width/height    target size (0 = keep the source dimension; for the
                      aspect-preserving modes they act as a bounding box)
      upscale_method  interpolation: nearest-exact / bilinear / area /
                      bicubic / lanczos
      keep_proportion see the module comment above
      crop_position   center / top / bottom / left / right — used both for
                      cropping and for aligning the image inside the padded area
      pad_color       colour for the "pad" mode
      divisible_by    sizes are rounded down to a multiple of this value

    Returns
    -------
    tuple
        ``(stream, out_w, out_h, changed)`` where ``changed`` is False when no
        geometry filter was needed (the caller can then stream-copy instead of
        re-encoding).
    """
    if ffmpeg is None:
        raise ImportError(
            "ffmpeg-python is not installed. Install it with: pip install ffmpeg-python"
        )

    src_w = int(src_w)
    src_h = int(src_h)
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid source resolution: {src_w}x{src_h}")

    mode = keep_proportion if keep_proportion in _RESIZE_MODES else "resize"
    flag = _SCALE_FLAGS.get(upscale_method, "lanczos")
    div = int(divisible_by or 0)
    if div < 2:
        div = 2

    changed = False

    def scale_to(src_stream, from_w, from_h, to_w, to_h, flags=None):
        """Scale ``src_stream`` unless it already has the requested size."""
        nonlocal changed
        if to_w == from_w and to_h == from_h:
            return src_stream
        changed = True
        return ffmpeg.filter(src_stream, "scale", to_w, to_h,
                             flags=flags or flag)

    # ---- stretch: ignore the aspect ratio -------------------------------
    if mode == "stretch":
        out_w = _even_floor(width if width > 0 else src_w, div)
        out_h = _even_floor(height if height > 0 else src_h, div)
        return scale_to(stream, src_w, src_h, out_w, out_h), out_w, out_h, changed

    # ---- crop: cut to the target aspect ratio, then scale ---------------
    if mode == "crop":
        out_w = _even_floor(width if width > 0 else src_w, div)
        out_h = _even_floor(height if height > 0 else src_h, div)

        src_ar = src_w / float(src_h)
        dst_ar = out_w / float(out_h)
        if src_ar > dst_ar:
            crop_w, crop_h = int(round(src_h * dst_ar)), src_h
        else:
            crop_w, crop_h = src_w, int(round(src_w / dst_ar))
        crop_w = max(2, min(crop_w, src_w))
        crop_h = max(2, min(crop_h, src_h))
        crop_w -= crop_w % 2
        crop_h -= crop_h % 2

        dx = src_w - crop_w
        dy = src_h - crop_h
        if crop_position == "top":
            cx, cy = dx // 2, 0
        elif crop_position == "bottom":
            cx, cy = dx // 2, dy
        elif crop_position == "left":
            cx, cy = 0, dy // 2
        elif crop_position == "right":
            cx, cy = dx, dy // 2
        else:
            cx, cy = dx // 2, dy // 2
        cx -= cx % 2
        cy -= cy % 2

        cropped = stream
        if crop_w != src_w or crop_h != src_h:
            changed = True
            cropped = ffmpeg.filter(stream, "crop", crop_w, crop_h, cx, cy)

        return (scale_to(cropped, crop_w, crop_h, out_w, out_h),
                out_w, out_h, changed)

    # ---- aspect-preserving inner size -----------------------------------
    if mode == "total_pixels":
        total_pixels = max(1, int(width or 0) * int(height or 0))
        src_ar = src_w / float(src_h)
        raw_w = int(math.sqrt(total_pixels * src_ar))
        raw_h = int(math.sqrt(total_pixels / src_ar))
    elif width <= 0 and height <= 0:
        raw_w, raw_h = src_w, src_h
    elif width <= 0:
        ratio = height / float(src_h)
        raw_w, raw_h = int(round(src_w * ratio)), int(height)
    elif height <= 0:
        ratio = width / float(src_w)
        raw_w, raw_h = int(width), int(round(src_h * ratio))
    else:
        ratio = min(width / float(src_w), height / float(src_h))
        raw_w, raw_h = int(round(src_w * ratio)), int(round(src_h * ratio))

    inner_w = _even_floor(raw_w, div)
    inner_h = _even_floor(raw_h, div)

    # ---- resize: no padding, the fit size is the output size -------------
    if mode == "resize":
        return (scale_to(stream, src_w, src_h, inner_w, inner_h),
                inner_w, inner_h, changed)

    # ---- pad / pad_edge / pad_edge_pixel / pillarbox_blur ---------------
    canvas_w = max(inner_w, int(width) if width > 0 else inner_w)
    canvas_h = max(inner_h, int(height) if height > 0 else inner_h)

    pad_l, pad_r, pad_t, pad_b = _pad_offsets(
        canvas_w, canvas_h, raw_w, raw_h, crop_position
    )

    # Mirror KJNodes: grow the bottom/right padding so that the padded frame
    # is a multiple of divisible_by.
    p_w = inner_w + pad_l + pad_r
    p_h = inner_h + pad_t + pad_b
    if p_w % div:
        pad_r += div - (p_w % div)
    if p_h % div:
        pad_b += div - (p_h % div)
    out_w = inner_w + pad_l + pad_r
    out_h = inner_h + pad_t + pad_b

    inner = scale_to(stream, src_w, src_h, inner_w, inner_h)

    if pad_l == 0 and pad_r == 0 and pad_t == 0 and pad_b == 0:
        return inner, inner_w, inner_h, changed

    changed = True

    # ---- solid colour padding -------------------------------------------
    if mode == "pad":
        return (ffmpeg.filter(inner, "pad", out_w, out_h, pad_l, pad_t,
                              color=parse_pad_color(pad_color)),
                out_w, out_h, changed)

    # ---- pillarbox blur -------------------------------------------------
    if mode == "pillarbox_blur":
        split = ffmpeg.filter_multi_output(inner, "split")
        fg = ffmpeg.filter(split.stream(0), "format", "yuv420p")

        # Background: cover the padded area, centre crop, blur, then
        # desaturate 20% and dim to 35% (0.8*col + 0.2*luma, then *0.35).
        bg = ffmpeg.filter(split.stream(1), "scale", out_w, out_h,
                           flags="bilinear",
                           force_original_aspect_ratio="increase")
        bg = ffmpeg.filter(bg, "crop", out_w, out_h)
        bg = ffmpeg.filter(
            bg, "gblur",
            sigma=round(max(1.0, 0.006 * float(min(out_w, out_h))), 3),
        )
        bg = ffmpeg.filter(bg, "format", "yuv420p")
        bg = ffmpeg.filter(
            bg, "lutyuv",
            y="val*0.35",
            u="(val-%d)*0.28+%d" % (_YUV_NEUTRAL, _YUV_NEUTRAL),
            v="(val-%d)*0.28+%d" % (_YUV_NEUTRAL, _YUV_NEUTRAL),
        )

        return (ffmpeg.filter([bg, fg], "overlay", x=pad_l, y=pad_t),
                out_w, out_h, changed)

    # ---- pad_edge / pad_edge_pixel --------------------------------------
    # The border is built from strips taken from the scaled frame: top row,
    # bottom row, left column (full padded height) and right column (full
    # padded height); the real content is pasted back on top last.
    #   pad_edge_pixel -> the strip is stretched (edge pixels are replicated)
    #   pad_edge       -> the strip is first reduced to 1x1 with an area
    #                     filter, so the border gets the mean edge colour
    use_mean = mode == "pad_edge"
    strips = []
    if pad_t > 0:
        strips.append(((inner_w, 1, 0, 0), (inner_w, pad_t), (pad_l, 0)))
    if pad_b > 0:
        strips.append(((inner_w, 1, 0, inner_h - 1),
                       (inner_w, pad_b), (pad_l, pad_t + inner_h)))
    if pad_l > 0:
        strips.append(((1, inner_h, 0, 0), (pad_l, out_h), (0, 0)))
    if pad_r > 0:
        strips.append(((1, inner_h, inner_w - 1, 0),
                       (pad_r, out_h), (inner_w + pad_l, 0)))

    split = ffmpeg.filter_multi_output(inner, "split")
    out = ffmpeg.filter(split.stream(0), "pad", out_w, out_h, pad_l, pad_t,
                        color="black")

    for index, (crop_args, size, position) in enumerate(strips, start=1):
        block = ffmpeg.filter(split.stream(index), "crop", *crop_args)
        if use_mean:
            block = ffmpeg.filter(block, "scale", 1, 1, flags="area")
        block = ffmpeg.filter(block, "scale", size[0], size[1], flags="neighbor")
        out = ffmpeg.filter([out, block], "overlay", x=position[0], y=position[1])

    content = split.stream(len(strips) + 1)
    out = ffmpeg.filter([out, content], "overlay", x=pad_l, y=pad_t)

    return out, out_w, out_h, changed

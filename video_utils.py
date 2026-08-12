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

def get_video_path(video_input):
    """
    Extract a file-system path from a ComfyUI VIDEO type input.

    Handles:
      - Plain string (file path)
      - VideoInput objects (via get_stream_source / save_to)
    """
    # Already a file path
    if isinstance(video_input, str):
        if not os.path.exists(video_input):
            raise FileNotFoundError(f"Video file not found: {video_input}")
        return video_input

    # VideoInput abstract object
    if hasattr(video_input, "get_stream_source"):
        source = video_input.get_stream_source()
        if isinstance(source, str):
            return source
        if isinstance(source, (io.BytesIO, io.BufferedReader)):
            tmp = get_temp_path()
            with open(tmp, "wb") as f:
                f.write(source.read())
            return tmp

    # Fallback: save_to
    if hasattr(video_input, "save_to"):
        tmp = get_temp_path()
        video_input.save_to(tmp)
        return tmp

    raise ValueError(
        f"Cannot extract file path from video input of type {type(video_input)}"
    )


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
      width, height, fps, duration, frame_count, codec
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

"""
ComfyUI custom nodes for FFmpeg-based video processing.

Node 1 — VideoConcat
    Concatenate two videos with optional transition (linear crossfade,
    cut-source, or cut-extend).  Resolution must match; frame-rate is
    auto-converted.  Audio is concatenated with matching transition
    (acrossfade / atrim+concat).  If only one video has audio, the other
    is treated as silence.  If neither has audio, the output has no audio.
    All processing is done via FFmpeg stream filters, so memory usage
    stays flat regardless of resolution or frame count.

Node 2 — VideoAddAudio
    Add or replace the audio track of a video.  Supports start / end
    alignment when video and audio durations differ.

Node 3 — VideoExtractSegment
    Extract a segment of frames (as IMAGE tensor) and corresponding
    audio from a video.  Frame-accurate positioning via FFmpeg.
    Returns silent audio if the video has no audio track.

Node 4 — VideoInfo
    Inspect a video and return its technical metadata (dimensions,
    frame rate, duration, frame count, codecs, bitrates, audio specs).
"""

import os

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

from .video_utils import (
    check_ffmpeg,
    get_video_path,
    get_video_info,
    get_audio_info,
    get_video_details,
    get_video_audio_info,
    extract_frames,
    extract_audio_segment,
    get_temp_path,
    save_audio_to_wav,
    create_video_output,
    _channel_layout_name,
)


# ===================================================================
# Node 1 — Video Concatenation
# ===================================================================

class VideoConcat:
    """Concatenate two videos with transition effects using FFmpeg."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video": (
                    "VIDEO",
                    {"tooltip": "The first (source) video to concatenate."},
                ),
                "extend_video": (
                    "VIDEO",
                    {"tooltip": "The second (extend) video to append."},
                ),
                "overlap": (
                    "INT",
                    {
                        "default": 15,
                        "min": 0,
                        "max": 10000,
                        "step": 1,
                        "tooltip": "Number of overlap frames for the transition.",
                    },
                ),
                "overlap_type": (
                    [
                        "linear_crossfade",
                        "cut_source",
                        "cut_extend",
                    ],
                    {
                        "tooltip": (
                            "linear_crossfade: smooth fade between the two videos. "
                            "cut_source: drop the last N frames of source, then concat. "
                            "cut_extend: drop the first N frames of extend, then concat."
                        ),
                    },
                ),
                "audio_sample_rate": (
                    [
                        "source_video",
                        "extend_video",
                    ],
                    {
                        "tooltip": (
                            "When the two videos have different audio sample rates, "
                            "resample the other video's audio to match the selected one. "
                            "source_video: use source video's sample rate. "
                            "extend_video: use extend video's sample rate."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "concat"
    CATEGORY = "FFmpeg Video"
    DESCRIPTION = (
        "Concatenate two videos with optional transition. "
        "Audio is concatenated with matching transition. "
        "Uses FFmpeg stream filters to keep memory usage flat."
    )

    # ------------------------------------------------------------------

    def concat(self, source_video, extend_video, overlap, overlap_type,
               audio_sample_rate):
        check_ffmpeg()

        # --- resolve file paths ---------------------------------------
        source_path = get_video_path(source_video)
        extend_path = get_video_path(extend_video)

        # --- probe video metadata -------------------------------------
        src = get_video_info(source_path)
        ext = get_video_info(extend_path)

        # --- resolution check -----------------------------------------
        if src["width"] != ext["width"] or src["height"] != ext["height"]:
            raise ValueError(
                f"Resolution mismatch — source is "
                f"{src['width']}x{src['height']}, extend is "
                f"{ext['width']}x{ext['height']}. "
                f"Both videos must have the same resolution."
            )

        # --- frame-rate check -----------------------------------------
        need_fps_convert = abs(src["fps"] - ext["fps"]) > 0.01

        # --- overlap duration (seconds) -------------------------------
        if src["fps"] <= 0:
            raise ValueError(f"Invalid source frame rate: {src['fps']}")

        overlap_duration = overlap / src["fps"]

        # --- validate overlap -----------------------------------------
        if overlap > 0:
            if overlap_duration >= src["duration"]:
                raise ValueError(
                    f"Overlap ({overlap} frames = {overlap_duration:.2f}s) "
                    f"exceeds source duration ({src['duration']:.2f}s)."
                )
            if overlap_type == "cut_extend" and overlap_duration >= ext["duration"]:
                raise ValueError(
                    f"Overlap ({overlap} frames = {overlap_duration:.2f}s) "
                    f"exceeds extend duration ({ext['duration']:.2f}s)."
                )

        # --- probe audio streams --------------------------------------
        src_has_audio, src_sr, src_ch = get_video_audio_info(source_path)
        ext_has_audio, ext_sr, ext_ch = get_video_audio_info(extend_path)
        has_any_audio = src_has_audio or ext_has_audio

        # --- determine target sample rate -----------------------------
        if audio_sample_rate == "source_video":
            target_sr = src_sr if src_has_audio else ext_sr
        else:  # "extend_video"
            target_sr = ext_sr if ext_has_audio else src_sr

        # --- determine target channel count ---------------------------
        if src_has_audio and ext_has_audio:
            target_ch = max(src_ch, ext_ch)
        elif src_has_audio:
            target_ch = src_ch
        elif ext_has_audio:
            target_ch = ext_ch
        else:
            target_ch = 2  # irrelevant — no audio in output
        target_layout = _channel_layout_name(target_ch)

        # --- build inputs ---------------------------------------------
        source_input = ffmpeg.input(source_path)
        extend_input = ffmpeg.input(extend_path)

        # --- video filter graph ---------------------------------------
        # Convert extend frame rate if necessary
        if need_fps_convert:
            extend_v = ffmpeg.filter(
                extend_input.video, "fps", fps=src["fps"]
            )
        else:
            extend_v = extend_input.video

        if overlap == 0:
            # ---- simple concat (no transition) -----------------------
            s = ffmpeg.filter(source_input.video, "setpts", "PTS-STARTPTS")
            e = ffmpeg.filter(extend_v, "setpts", "PTS-STARTPTS")
            s = ffmpeg.filter(s, "format", pix_fmts="yuv420p")
            e = ffmpeg.filter(e, "format", pix_fmts="yuv420p")
            video = ffmpeg.concat(s, e, v=1, a=0)

        elif overlap_type == "cut_source":
            # ---- trim last N frames of source, then concat ------------
            trim_dur = src["duration"] - overlap_duration
            s = ffmpeg.filter(source_input.video, "trim", duration=trim_dur)
            s = ffmpeg.filter(s, "setpts", "PTS-STARTPTS")
            s = ffmpeg.filter(s, "format", pix_fmts="yuv420p")

            e = ffmpeg.filter(extend_v, "setpts", "PTS-STARTPTS")
            e = ffmpeg.filter(e, "format", pix_fmts="yuv420p")

            video = ffmpeg.concat(s, e, v=1, a=0)

        elif overlap_type == "cut_extend":
            # ---- trim first N frames of extend, then concat -----------
            s = ffmpeg.filter(source_input.video, "setpts", "PTS-STARTPTS")
            s = ffmpeg.filter(s, "format", pix_fmts="yuv420p")

            e = ffmpeg.filter(extend_v, "trim", start=overlap_duration)
            e = ffmpeg.filter(e, "setpts", "PTS-STARTPTS")
            e = ffmpeg.filter(e, "format", pix_fmts="yuv420p")

            video = ffmpeg.concat(s, e, v=1, a=0)

        else:
            # ---- linear crossfade via xfade ---------------------------
            offset = src["duration"] - overlap_duration
            video = ffmpeg.filter(
                [source_input.video, extend_v],
                "xfade",
                transition="fade",
                duration=overlap_duration,
                offset=offset,
            )

        # --- audio filter graph ---------------------------------------
        audio = None
        if has_any_audio:
            # Source audio stream (or silence if no audio)
            if src_has_audio:
                src_audio = ffmpeg.filter(
                    source_input.audio, "aresample",
                    osr=target_sr, ochl=target_layout,
                )
            else:
                src_audio = ffmpeg.input(
                    f"anullsrc=channel_layout={target_layout}"
                    f":sample_rate={target_sr}",
                    f="lavfi",
                    t=src["duration"],
                ).audio

            # Extend audio stream (or silence if no audio)
            if ext_has_audio:
                ext_audio = ffmpeg.filter(
                    extend_input.audio, "aresample",
                    osr=target_sr, ochl=target_layout,
                )
            else:
                ext_audio = ffmpeg.input(
                    f"anullsrc=channel_layout={target_layout}"
                    f":sample_rate={target_sr}",
                    f="lavfi",
                    t=ext["duration"],
                ).audio

            # Apply the same transition as video
            if overlap == 0:
                # ---- simple audio concat --------------------------------
                src_a = ffmpeg.filter(src_audio, "asetpts", "PTS-STARTPTS")
                ext_a = ffmpeg.filter(ext_audio, "asetpts", "PTS-STARTPTS")
                audio = ffmpeg.concat(src_a, ext_a, v=0, a=1)

            elif overlap_type == "linear_crossfade":
                # ---- audio crossfade via acrossfade ---------------------
                audio = ffmpeg.filter(
                    [src_audio, ext_audio],
                    "acrossfade",
                    duration=overlap_duration,
                )

            elif overlap_type == "cut_source":
                # ---- trim source audio, then concat ---------------------
                trim_dur = src["duration"] - overlap_duration
                src_a = ffmpeg.filter(src_audio, "atrim", duration=trim_dur)
                src_a = ffmpeg.filter(src_a, "asetpts", "PTS-STARTPTS")
                ext_a = ffmpeg.filter(ext_audio, "asetpts", "PTS-STARTPTS")
                audio = ffmpeg.concat(src_a, ext_a, v=0, a=1)

            elif overlap_type == "cut_extend":
                # ---- trim extend audio, then concat ---------------------
                src_a = ffmpeg.filter(src_audio, "asetpts", "PTS-STARTPTS")
                ext_a = ffmpeg.filter(
                    ext_audio, "atrim", start=overlap_duration
                )
                ext_a = ffmpeg.filter(ext_a, "asetpts", "PTS-STARTPTS")
                audio = ffmpeg.concat(src_a, ext_a, v=0, a=1)

        # --- encode ---------------------------------------------------
        output_path = get_temp_path(".mp4")

        if audio is not None:
            out = ffmpeg.output(
                video, audio,
                output_path,
                vcodec="libx264",
                preset="medium",
                crf=23,
                pix_fmt="yuv420p",
                acodec="aac",
            )
        else:
            out = ffmpeg.output(
                video,
                output_path,
                vcodec="libx264",
                preset="medium",
                crf=23,
                pix_fmt="yuv420p",
            )

        try:
            ffmpeg.run(out, overwrite_output=True, capture_stderr=True)
        except ffmpeg.Error as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
            raise RuntimeError(f"FFmpeg encoding failed:\n{stderr}")

        return (create_video_output(output_path),)


# ===================================================================
# Node 2 — Add Audio to Video
# ===================================================================

class VideoAddAudio:
    """Add or replace the audio track of a video using FFmpeg."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (
                    "VIDEO",
                    {"tooltip": "The target video."},
                ),
                "align": (
                    ["start", "end"],
                    {
                        "tooltip": (
                            "start: audio begins at the video start. "
                            "end: audio ends at the video end."
                        ),
                    },
                ),
            },
            "optional": {
                "audio": (
                    "AUDIO",
                    {"tooltip": "Audio input from ComfyUI audio nodes."},
                ),
                "audio_file": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to an audio file (used when no AUDIO input is connected).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "add_audio"
    CATEGORY = "FFmpeg Video"
    DESCRIPTION = (
        "Add or replace the audio track of a video. "
        "Supports start / end alignment for mismatched durations."
    )

    # ------------------------------------------------------------------

    def add_audio(self, video, align, audio=None, audio_file=""):
        check_ffmpeg()

        # --- resolve video path ---------------------------------------
        video_path = get_video_path(video)
        vinfo = get_video_info(video_path)
        video_duration = vinfo["duration"]

        # --- resolve audio path ---------------------------------------
        audio_path = None
        is_temp_audio = False

        if audio is not None:
            audio_path = get_temp_path(".wav")
            save_audio_to_wav(audio, audio_path)
            is_temp_audio = True
        elif audio_file and audio_file.strip():
            audio_path = audio_file.strip()
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
        else:
            raise ValueError(
                "No audio input provided. "
                "Connect an AUDIO input or set audio_file path."
            )

        # --- probe audio ----------------------------------------------
        ainfo = get_audio_info(audio_path)
        audio_duration = ainfo["duration"]

        # --- determine output path early ------------------------------
        output_path = get_temp_path(".mp4")

        # --- build filter graph ---------------------------------------
        video_input = ffmpeg.input(video_path)
        audio_input = ffmpeg.input(audio_path)

        if align == "start":
            # Audio starts at video start.
            # If audio is longer, use -shortest to trim to video length.
            # If audio is shorter, video continues (audio simply ends early).
            if audio_duration > video_duration:
                out = ffmpeg.output(
                    video_input.video, audio_input.audio,
                    output_path,
                    vcodec="copy",
                    acodec="aac",
                    **{"shortest": None},
                )
            else:
                out = ffmpeg.output(
                    video_input.video, audio_input.audio,
                    output_path,
                    vcodec="copy",
                    acodec="aac",
                )

        else:  # end alignment
            if audio_duration > video_duration:
                # Trim the beginning so the audio tail aligns with video end
                trim_start = audio_duration - video_duration
                a = ffmpeg.filter(audio_input.audio, "atrim", start=trim_start)
                a = ffmpeg.filter(a, "asetpts", "PTS-STARTPTS")
            else:
                # Delay audio so it ends with the video
                delay_ms = int(round((video_duration - audio_duration) * 1000))
                if delay_ms > 0:
                    a = ffmpeg.filter(
                        audio_input.audio, "adelay", delays=delay_ms, all=1
                    )
                else:
                    a = audio_input.audio

            out = ffmpeg.output(
                video_input.video, a,
                output_path,
                vcodec="copy",
                acodec="aac",
            )

        # --- run ffmpeg -----------------------------------------------
        try:
            ffmpeg.run(out, overwrite_output=True, capture_stderr=True)
        except ffmpeg.Error as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
            raise RuntimeError(f"FFmpeg audio merge failed:\n{stderr}")

        # --- cleanup temp audio ---------------------------------------
        if is_temp_audio and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass

        return (create_video_output(output_path),)


# ===================================================================
# Node 3 — Extract Frame Segment & Audio
# ===================================================================

class VideoExtractSegment:
    """Extract a segment of frames and audio from a video using FFmpeg."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (
                    "VIDEO",
                    {"tooltip": "The source video to extract from."},
                ),
                "start_frame": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "step": 1,
                        "tooltip": "Frame number to start extraction from (0-indexed).",
                    },
                ),
                "length": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "step": 1,
                        "tooltip": "Number of frames to extract. "
                                   "Clamped to the remaining frames in the video.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "extract_segment"
    CATEGORY = "FFmpeg Video"
    DESCRIPTION = (
        "Extract a segment of frames (as IMAGE tensor) and corresponding "
        "audio from a video. Frame-accurate extraction via FFmpeg pipe. "
        "If the segment exceeds the video length, only available frames "
        "are returned. If the video has no audio, silent audio is returned."
    )

    # ------------------------------------------------------------------

    def extract_segment(self, video, start_frame, length):
        check_ffmpeg()

        # --- resolve video path ---------------------------------------
        video_path = get_video_path(video)
        vinfo = get_video_info(video_path)

        fps = vinfo["fps"]
        if fps <= 0:
            raise ValueError(f"Invalid frame rate: {fps}")

        total_frames = vinfo["frame_count"]
        width = vinfo["width"]
        height = vinfo["height"]

        # --- validate start_frame -------------------------------------
        if start_frame >= total_frames:
            raise ValueError(
                f"start_frame ({start_frame}) exceeds video frame count "
                f"({total_frames})."
            )

        # --- clamp length to remaining frames -------------------------
        remaining = total_frames - start_frame
        extract_length = min(length, remaining)

        # --- convert frame positions to time --------------------------
        start_time = start_frame / fps
        extract_duration = extract_length / fps

        # --- detect audio stream --------------------------------------
        has_audio, audio_sr, audio_ch = get_video_audio_info(video_path)

        # --- extract frames as IMAGE tensor ---------------------------
        images = extract_frames(
            video_path,
            start_time,
            extract_duration,
            width,
            height,
            extract_length,
        )

        # --- extract audio as AUDIO dict ------------------------------
        audio = extract_audio_segment(
            video_path,
            start_time,
            extract_duration,
            has_audio,
            audio_sr,
            audio_ch,
        )

        return (images, audio)


# ===================================================================
# Node 4 — Video Information
# ===================================================================

class VideoInfo:
    """Inspect a video and return its technical metadata."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (
                    "VIDEO",
                    {"tooltip": "The video to inspect."},
                ),
            }
        }

    RETURN_TYPES = (
        "INT", "INT", "FLOAT", "FLOAT", "INT",
        "STRING", "INT",
        "INT", "INT", "INT", "STRING", "INT",
    )
    RETURN_NAMES = (
        "width", "height", "fps", "duration", "frame_count",
        "video_codec", "video_bitrate_kbps",
        "audio_channels", "audio_sample_rate", "audio_bit_depth",
        "audio_codec", "audio_bitrate_kbps",
    )
    FUNCTION = "inspect"
    CATEGORY = "FFmpeg Video"
    DESCRIPTION = (
        "Inspect a video file and return its technical metadata: "
        "dimensions, frame rate, duration, frame count, video codec and "
        "average bitrate, plus audio channel count, sample rate, bit "
        "depth, codec and average bitrate.  Audio fields are 0 / empty "
        "when the video has no audio track."
    )

    # ------------------------------------------------------------------

    def inspect(self, video):
        check_ffmpeg()

        video_path = get_video_path(video)
        info = get_video_details(video_path)

        return (
            info["width"],
            info["height"],
            info["fps"],
            info["duration"],
            info["frame_count"],
            info["video_codec"],
            info["video_bitrate_kbps"],
            info["audio_channels"],
            info["audio_sample_rate"],
            info["audio_bit_depth"],
            info["audio_codec"],
            info["audio_bitrate_kbps"],
        )

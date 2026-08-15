"""
ComfyUI FFmpeg Video — Custom Nodes

Nodes for memory-efficient video processing via FFmpeg:

  VideoConcat         — concatenate two videos with crossfade / cut transitions
  VideoAddAudio       — add or replace audio on a video with start/end alignment
  VideoExtractSegment — extract a frame segment (IMAGE) and audio from a video
  VideoInfo           — inspect a video and return its technical metadata

Install:  copy this folder into ComfyUI/custom_nodes/  then restart.
Depends:  pip install ffmpeg-python   (FFmpeg binary must also be on PATH)
"""

from .nodes import (
    VideoConcat,
    VideoAddAudio,
    VideoExtractSegment,
    VideoInfo,
)

NODE_CLASS_MAPPINGS = {
    "VideoConcat": VideoConcat,
    "VideoAddAudio": VideoAddAudio,
    "VideoExtractSegment": VideoExtractSegment,
    "VideoInfo": VideoInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoConcat": "Video Concat (FFmpeg)",
    "VideoAddAudio": "Video Add Audio (FFmpeg)",
    "VideoExtractSegment": "Video Extract Segment (FFmpeg)",
    "VideoInfo": "Video Info (FFmpeg)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

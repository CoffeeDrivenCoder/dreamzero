#!/usr/bin/env python3
"""Merge AgiBot segment videos and resample to a target frame count."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import av
import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--video-dir',
        default='/root/autodl-tmp/real_world_eval_gen_20260319_0/GEAR-Dreams-DreamZero-AgiBot',
        help='Directory containing numbered mp4 segment files.',
    )
    parser.add_argument(
        '--source-video',
        default='/root/autodl-tmp/dreamzero/debug_image/head_color.mp4',
        help='Reference source video used to infer target frame count/fps.',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output merged mp4 path. Defaults to <video-dir>/merged_full_synced.mp4',
    )
    parser.add_argument(
        '--target-frames',
        type=int,
        default=None,
        help='Target total frame count. Defaults to the source video frame count.',
    )
    parser.add_argument(
        '--target-fps',
        type=float,
        default=None,
        help='Output fps. Defaults to the source video fps.',
    )
    return parser.parse_args()


def get_video_info(path: Path) -> tuple[int, float, int, int]:
    container = av.open(str(path))
    stream = container.streams.video[0]
    frames = stream.frames
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    width = stream.width
    height = stream.height
    container.close()
    return frames, fps, width, height


def gather_segments(video_dir: Path) -> list[Path]:
    return sorted(
        p for p in video_dir.glob('*.mp4')
        if p.name[:6].isdigit()
    )


def build_selected_indices(total_input_frames: int, target_frames: int) -> list[int]:
    if target_frames <= 0:
        raise ValueError('target_frames must be positive')
    if total_input_frames <= 0:
        raise ValueError('total_input_frames must be positive')
    if target_frames == 1:
        return [0]
    if total_input_frames == 1:
        return [0] * target_frames

    selected = []
    for i in range(target_frames):
        src_idx = round(i * (total_input_frames - 1) / (target_frames - 1))
        selected.append(src_idx)
    return selected


def main() -> None:
    args = parse_args()
    video_dir = Path(args.video_dir)
    segments = gather_segments(video_dir)
    if not segments:
        raise SystemExit(f'No numbered segment mp4 files found in {video_dir}')

    source_path = Path(args.source_video)
    source_frames, source_fps, _, _ = get_video_info(source_path)
    target_frames = args.target_frames or source_frames
    target_fps = args.target_fps or source_fps
    output_path = Path(args.output) if args.output else video_dir / 'merged_full_synced_2418f_30fps_h264.mp4'

    first_frames, _, width, height = get_video_info(segments[0])
    if first_frames <= 0:
        raise SystemExit(f'Failed to inspect first segment: {segments[0]}')

    total_input_frames = 0
    for segment in segments:
        frames, fps, seg_w, seg_h = get_video_info(segment)
        if seg_w != width or seg_h != height:
            raise SystemExit(f'Segment size mismatch: {segment} has {(seg_w, seg_h)} vs {(width, height)}')
        total_input_frames += frames

    selected_indices = build_selected_indices(total_input_frames, target_frames)
    selected_set = set(selected_indices)

    out_container = av.open(str(output_path), mode='w')
    out_stream = out_container.add_stream('libx264', rate=Fraction(str(target_fps)))
    out_stream.width = width
    out_stream.height = height
    out_stream.pix_fmt = 'yuv420p'
    out_stream.options = {'crf': '18', 'preset': 'medium'}

    written = 0
    current_global_idx = 0
    target_ptr = 0

    try:
        for segment in segments:
            cap = cv2.VideoCapture(str(segment))
            if not cap.isOpened():
                raise SystemExit(f'Failed to open segment {segment}')
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if current_global_idx in selected_set:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
                    video_frame = video_frame.reformat(width=width, height=height, format='yuv420p')
                    for packet in out_stream.encode(video_frame):
                        out_container.mux(packet)
                    written += 1
                    while target_ptr < len(selected_indices) and selected_indices[target_ptr] == current_global_idx:
                        target_ptr += 1
                current_global_idx += 1
            cap.release()

        for packet in out_stream.encode():
            out_container.mux(packet)
    finally:
        out_container.close()

    print({
        'output': str(output_path),
        'segments': len(segments),
        'input_frames': total_input_frames,
        'target_frames': target_frames,
        'written_frames': written,
        'target_fps': target_fps,
        'approx_duration_sec': written / target_fps if target_fps else None,
    })


if __name__ == '__main__':
    main()

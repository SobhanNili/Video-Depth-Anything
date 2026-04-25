import argparse
import concurrent.futures
from pathlib import Path

import torch
from tqdm import tqdm

from utils.dc_utils import read_video_frames, save_video
from video_depth_anything.video_depth import VideoDepthAnything


def find_videos(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4"
    )


def process_one_video(
    video: Path,
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    model: VideoDepthAnything,
    device: str,
) -> tuple[Path, bool, str]:
    rel_path = video.relative_to(input_dir)
    final_output = output_dir / rel_path
    final_output.parent.mkdir(parents=True, exist_ok=True)

    if final_output.exists() and not args.overwrite:
        return video, True, ""

    try:
        frames, target_fps = read_video_frames(
            str(video),
            args.max_len,
            args.target_fps,
            args.max_res,
        )
    except Exception as exc:
        return video, False, str(exc)

    return process_one_video_from_frames(video, input_dir, output_dir, args, model, device, frames, target_fps)


def process_one_video_from_frames(
    video: Path,
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    model: VideoDepthAnything,
    device: str,
    frames,
    target_fps,
) -> tuple[Path, bool, str]:
    rel_path = video.relative_to(input_dir)
    final_output = output_dir / rel_path
    final_output.parent.mkdir(parents=True, exist_ok=True)

    if final_output.exists() and not args.overwrite:
        return video, True, ""

    try:
        depths, fps = model.infer_video_depth(
            frames,
            target_fps,
            input_size=args.input_size,
            device=device,
            fp32=args.fp32,
        )
        save_video(depths, str(final_output), fps=fps, is_depths=True, grayscale=args.grayscale)
        return video, True, ""
    except Exception as exc:
        return video, False, str(exc)


def load_model(args: argparse.Namespace, device: str) -> VideoDepthAnything:
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    checkpoint_name = "metric_video_depth_anything" if args.metric else "video_depth_anything"
    checkpoint_path = Path(__file__).resolve().parent / "checkpoints" / f"{checkpoint_name}_{args.encoder}.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = VideoDepthAnything(**model_configs[args.encoder], metric=args.metric)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"), strict=True)
    model = model.to(device).eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract depth for all .mp4 files recursively and mirror folder hierarchy in output."
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Root directory to search for .mp4 files.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Root directory to save depth videos.")
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--metric", action="store_true", help="Pass --metric to run.py.")
    parser.add_argument("--fp32", action="store_true", help="Infer with torch.float32 (default: float16).")
    parser.add_argument("--grayscale", action="store_true", help="Save grayscale depth videos.")
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--max_res", type=int, default=1280)
    parser.add_argument("--max_len", type=int, default=-1)
    parser.add_argument("--target_fps", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Compatibility flag. Kept for CLI stability; inference runs with one shared model for speed.",
    )
    parser.add_argument(
        "--decode_workers",
        type=int,
        default=2,
        help="Number of CPU threads used to decode upcoming videos while GPU infers the current one.",
    )
    parser.add_argument(
        "--prefetch_videos",
        type=int,
        default=2,
        help="How many videos to prefetch in memory ahead of inference.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    videos = find_videos(input_dir)
    if not videos:
        print(f"No .mp4 files found under: {input_dir}")
        return 0
    if args.decode_workers < 1:
        raise ValueError("--decode_workers must be >= 1")
    if args.prefetch_videos < 1:
        raise ValueError("--prefetch_videos must be >= 1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = load_model(args, device)

    failures: list[tuple[Path, str]] = []
    decode_workers = min(args.decode_workers, len(videos))
    prefetch = min(args.prefetch_videos, len(videos))

    with concurrent.futures.ThreadPoolExecutor(max_workers=decode_workers) as executor:
        decode_futures: dict[int, concurrent.futures.Future] = {}

        def submit_decode(index: int) -> None:
            video = videos[index]
            rel_path = video.relative_to(input_dir)
            final_output = output_dir / rel_path
            if final_output.exists() and not args.overwrite:
                decode_futures[index] = None
                return
            decode_futures[index] = executor.submit(
                read_video_frames,
                str(video),
                args.max_len,
                args.target_fps,
                args.max_res,
            )

        for idx in range(prefetch):
            submit_decode(idx)

        for idx, video in enumerate(tqdm(videos, desc="Processing videos", unit="video")):
            next_idx = idx + prefetch
            if next_idx < len(videos):
                submit_decode(next_idx)

            decode_future = decode_futures.pop(idx)
            if decode_future is None:
                continue

            try:
                frames, target_fps = decode_future.result()
            except Exception as exc:
                failures.append((video, str(exc)))
                continue

            _, ok, err = process_one_video_from_frames(
                video,
                input_dir,
                output_dir,
                args,
                model,
                device,
                frames,
                target_fps,
            )
            if not ok:
                failures.append((video, err))

    succeeded = len(videos) - len(failures)
    print(f"Completed: {succeeded}/{len(videos)} videos")

    if failures:
        print("Failures:")
        for video, err in failures:
            tail = "\n".join(err.splitlines()[-8:])
            print(f"- {video}\n{tail}\n")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
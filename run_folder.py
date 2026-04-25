import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm


def find_videos(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4"
    )


def build_run_cmd(args: argparse.Namespace, input_video: Path, output_dir: Path, run_script: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(run_script),
        "--input_video",
        str(input_video),
        "--output_dir",
        str(output_dir),
        "--encoder",
        args.encoder,
        "--input_size",
        str(args.input_size),
        "--max_res",
        str(args.max_res),
        "--max_len",
        str(args.max_len),
        "--target_fps",
        str(args.target_fps),
    ]

    if args.metric:
        cmd.append("--metric")
    if args.fp32:
        cmd.append("--fp32")
    if args.grayscale:
        cmd.append("--grayscale")

    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run run.py on all .mp4 files recursively and mirror folder hierarchy in output."
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Root directory to search for .mp4 files.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Root directory to save depth videos.")
    parser.add_argument(
        "--run_script",
        type=Path,
        default=Path(__file__).resolve().parent / "run.py",
        help="Path to run.py.",
    )
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--metric", action="store_true", help="Pass --metric to run.py.")
    parser.add_argument("--fp32", action="store_true", help="Pass --fp32 to run.py.")
    parser.add_argument("--grayscale", action="store_true", help="Pass --grayscale to run.py.")
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--max_res", type=int, default=1280)
    parser.add_argument("--max_len", type=int, default=-1)
    parser.add_argument("--target_fps", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    run_script = args.run_script.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not run_script.is_file():
        raise FileNotFoundError(f"run.py not found: {run_script}")

    videos = find_videos(input_dir)
    if not videos:
        print(f"No .mp4 files found under: {input_dir}")
        return 0

    failures: list[tuple[Path, str]] = []

    for video in tqdm(videos, desc="Processing videos", unit="video"):
        rel_path = video.relative_to(input_dir)
        final_output = output_dir / rel_path
        final_output.parent.mkdir(parents=True, exist_ok=True)

        if final_output.exists() and not args.overwrite:
            continue

        with tempfile.TemporaryDirectory(prefix="vda_tmp_") as tmp_dir:
            tmp_output = Path(tmp_dir)
            cmd = build_run_cmd(args, video, tmp_output, run_script)

            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                err_msg = (result.stderr or result.stdout or "run.py failed with no output").strip()
                failures.append((video, err_msg))
                continue

            vis_file = tmp_output / f"{video.stem}_vis.mp4"
            if not vis_file.exists():
                failures.append((video, f"Expected output not found: {vis_file}"))
                continue

            if final_output.exists() and args.overwrite:
                final_output.unlink()
            shutil.move(str(vis_file), str(final_output))

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
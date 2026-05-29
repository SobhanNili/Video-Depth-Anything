# Extracting Depth From Videos

## Setup
- `cd third_party/vda`
- Create and activate a uv virtual environment.
- Install dependencies: `uv pip install -r requirements.txt`
- Install extra package used by the environment: `uv pip install setuptools`
- Put the checkpoint from [huggingface repo](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/blob/main/metric_video_depth_anything_vits.pth) in `third_party/vda/checkpoints/` as `metric_video_depth_anything_vits.pth`.

## Single Video
`python run.py --input_video INPUT_VIDEO_PATH --output_dir OUTPUT_DIRECTORY --encoder vits --metric`

## Folder (Recursive)
- Command:
	`python run_folder.py --input_dir INPUT_ROOT --output_dir OUTPUT_ROOT --encoder vits --metric --decode_workers 4 --prefetch_videos 3`
- Behavior:
	Recursively finds `.mp4` under `INPUT_ROOT`, and writes depth videos to `OUTPUT_ROOT` with the same relative folder hierarchy and same filenames.
- Useful options:
	`--overwrite` to replace existing outputs.

## Example (Izar)
- `python run_folder.py --input_dir /work/cs-503/phys_reason/clevrer_video/ --output_dir /work/cs-503/phys_reason/clevrer_depth_video/ --encoder vits --metric --decode_workers 4 --prefetch_videos 3`
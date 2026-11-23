# Samples

Input clips to try the counter on, each with the counting lines that were used for the example results in [../results/README.md](../results/README.md).

| Clip | Length | Lines file | Source |
|---|---|---|---|
| [`shinjuku_90s.mp4`](shinjuku_90s.mp4) | 90 s | [`shinjuku_lines.json`](shinjuku_lines.json) (12 lines) | Tokyo Shinjuku crossing live webcam: https://www.youtube.com/watch?v=6dp-bvQ7RWo |
| [`shinjuku_3min.mp4`](shinjuku_3min.mp4) | 3 min | [`shinjuku_lines.json`](shinjuku_lines.json) (12 lines) | same webcam, a different moment |
| [`abbey_road_3min.mp4`](abbey_road_3min.mp4) | 3 min | [`abbey_road_lines.json`](abbey_road_lines.json) (4 lines) | Abbey Road crossing live webcam: https://www.youtube.com/watch?v=zMCea32gpmg |

## About the footage

- The clips were captured from public live webcam streams (street cameras filming a public crossing). People appear in public spaces and are shown only to illustrate the method; if you appear in a clip and want it removed, please open an issue.
- Each clip is H.264, 1920x1080, 30 fps, without audio, re-encoded to keep the file small.
- The lines files use pixel coordinates of the 1920x1080 frame, so they only fit these clips at that resolution. The click order of a line defines which way is IN.

## Usage

```bash
line-counter --input samples/abbey_road_3min.mp4 --lines samples/abbey_road_lines.json --no-display \
    --imgsz 1920 --conf 0.15 --tracker configs/botsort_reid_static.yaml --frame-step 3 \
    --output outputs/abbey_road_3min_panel.mp4
```

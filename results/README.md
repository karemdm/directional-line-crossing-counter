# Results

Annotated videos produced by this project.

- `shinjuku_line_count.mp4`: Busy intersection with several counting lines; per-line IN/OUT counts for people and vehicles in the HUD.
- Three live-webcam runs with the side panel, described below. Each has the annotated video (`*_counted.mp4`), the tracks (`*_tracks.jsonl`) and the final counts (`*_counts.json`).

The three runs share the same settings: `--imgsz 1920 --conf 0.15 --tracker configs/botsort_reid_static.yaml --frame-step 3 --layout side` with the default model. The videos here are re-encoded to H.264 at 1600 px wide without audio, so the panel text stays readable but the files stay small. The counts in the tables are the final counts of the original run, copied from its log.

Counts depend on detection and tracking quality: people on the far side of the crossing are only a few pixels tall, so some are missed and some IDs switch. Treat the numbers as an illustration of the method, not as ground truth.

## Shinjuku, 90 s

- Source: Tokyo Shinjuku crossing live webcam: https://www.youtube.com/watch?v=6dp-bvQ7RWo
- Input: [`../samples/shinjuku_90s.mp4`](../samples/shinjuku_90s.mp4), lines: [`../samples/shinjuku_lines.json`](../samples/shinjuku_lines.json)
- Output: [`shinjuku_90s_counted.mp4`](shinjuku_90s_counted.mp4), tracks: [`shinjuku_90s_tracks.jsonl`](shinjuku_90s_tracks.jsonl), counts: [`shinjuku_90s_counts.json`](shinjuku_90s_counts.json)

```bash
line-counter --input samples/shinjuku_90s.mp4 --lines samples/shinjuku_lines.json --no-display \
    --imgsz 1920 --conf 0.15 --tracker configs/botsort_reid_static.yaml --frame-step 3 --layout side \
    --output outputs/shinjuku_90s_counted.mp4 \
    --save-tracks results/shinjuku_90s_tracks.jsonl --save-counts results/shinjuku_90s_counts.json
```

Re-render without the detector (same counts, any `--layout`):

```bash
line-counter --input samples/shinjuku_90s.mp4 --lines samples/shinjuku_lines.json --no-display \
    --replay-tracks results/shinjuku_90s_tracks.jsonl --output outputs/shinjuku_90s_replay.mp4
```

| Line | Person IN / OUT | Other classes (IN / OUT, only non-zero) |
|---|---|---|
| Line_1 | 1 / 12 | - |
| Line_2 | 7 / 22 | - |
| Line_3 | 49 / 47 | - |
| Line_4 | 32 / 41 | - |
| Line_5 | 0 / 0 | bus 1 / 0 |
| Line_6 | 0 / 1 | - |
| Line_7 | 3 / 5 | - |
| Line_8 | 8 / 9 | - |
| Line_9 | 5 / 5 | car 25 / 11, motorcycle 0 / 1, bus 1 / 0, truck 1 / 0 |
| Line_10 | 1 / 2 | car 0 / 5, truck 0 / 1 |
| Line_11 | 2 / 0 | car 12 / 24, motorcycle 1 / 0, bus 1 / 0, truck 1 / 0 |
| Line_12 | 4 / 1 | car 10 / 11, bus 0 / 1, truck 0 / 2 |

## Abbey Road, 3 min

- Source: Abbey Road crossing live webcam: https://www.youtube.com/watch?v=zMCea32gpmg
- Input: [`../samples/abbey_road_3min.mp4`](../samples/abbey_road_3min.mp4), lines: [`../samples/abbey_road_lines.json`](../samples/abbey_road_lines.json)
- Output: [`abbey_road_3min_counted.mp4`](abbey_road_3min_counted.mp4), tracks: [`abbey_road_3min_tracks.jsonl`](abbey_road_3min_tracks.jsonl), counts: [`abbey_road_3min_counts.json`](abbey_road_3min_counts.json)

```bash
line-counter --input samples/abbey_road_3min.mp4 --lines samples/abbey_road_lines.json --no-display \
    --imgsz 1920 --conf 0.15 --tracker configs/botsort_reid_static.yaml --frame-step 3 --layout side \
    --output outputs/abbey_road_3min_counted.mp4 \
    --save-tracks results/abbey_road_3min_tracks.jsonl --save-counts results/abbey_road_3min_counts.json
```

Re-render without the detector (same counts, any `--layout`):

```bash
line-counter --input samples/abbey_road_3min.mp4 --lines samples/abbey_road_lines.json --no-display \
    --replay-tracks results/abbey_road_3min_tracks.jsonl --output outputs/abbey_road_3min_replay.mp4
```

| Line | Person IN / OUT | Other classes (IN / OUT, only non-zero) |
|---|---|---|
| Line_1 | 9 / 14 | - |
| Line_2 | 15 / 13 | - |
| Line_3 | 6 / 6 | car 12 / 14, motorcycle 1 / 0, bus 1 / 0, truck 3 / 0 |
| Line_4 | 6 / 5 | - |

## Shinjuku, 3 min

- Source: Tokyo Shinjuku crossing live webcam: https://www.youtube.com/watch?v=6dp-bvQ7RWo (a different moment than the 90 s run)
- Input: [`../samples/shinjuku_3min.mp4`](../samples/shinjuku_3min.mp4), lines: [`../samples/shinjuku_lines.json`](../samples/shinjuku_lines.json)
- Output: [`shinjuku_3min_counted.mp4`](shinjuku_3min_counted.mp4), tracks: [`shinjuku_3min_tracks.jsonl`](shinjuku_3min_tracks.jsonl), counts: [`shinjuku_3min_counts.json`](shinjuku_3min_counts.json)

```bash
line-counter --input samples/shinjuku_3min.mp4 --lines samples/shinjuku_lines.json --no-display \
    --imgsz 1920 --conf 0.15 --tracker configs/botsort_reid_static.yaml --frame-step 3 --layout side \
    --output outputs/shinjuku_3min_counted.mp4 \
    --save-tracks results/shinjuku_3min_tracks.jsonl --save-counts results/shinjuku_3min_counts.json
```

Re-render without the detector (same counts, any `--layout`):

```bash
line-counter --input samples/shinjuku_3min.mp4 --lines samples/shinjuku_lines.json --no-display \
    --replay-tracks results/shinjuku_3min_tracks.jsonl --output outputs/shinjuku_3min_replay.mp4
```

| Line | Person IN / OUT | Other classes (IN / OUT, only non-zero) |
|---|---|---|
| Line_1 | 18 / 17 | - |
| Line_2 | 35 / 34 | - |
| Line_3 | 28 / 58 | - |
| Line_4 | 37 / 79 | - |
| Line_5 | 1 / 0 | - |
| Line_6 | 5 / 3 | - |
| Line_7 | 13 / 15 | - |
| Line_8 | 9 / 11 | - |
| Line_9 | 13 / 5 | car 57 / 41, motorcycle 1 / 1, bus 1 / 1, truck 1 / 0 |
| Line_10 | 2 / 5 | car 2 / 5, bus 1 / 0 |
| Line_11 | 14 / 11 | car 43 / 60, motorcycle 0 / 2, bus 2 / 1, truck 1 / 2 |
| Line_12 | 7 / 3 | car 11 / 13, motorcycle 0 / 1, bus 0 / 2, truck 0 / 1 |

## Source footage

The three live-webcam runs use clips captured from public live streams (see [../samples/README.md](../samples/README.md)). `shinjuku_line_count.mp4` is a publicly available street recording that the author considers free to use.
<!-- TODO(author): add the link and license of the source video of shinjuku_line_count.mp4 here before publishing. -->

People appear in public spaces; the outputs are shown only to illustrate the method.
If you appear in a video and want it removed, please open an issue.

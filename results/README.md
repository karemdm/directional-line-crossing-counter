# Results

Annotated videos produced by this project.

- `shinjuku_line_count.mp4`: Busy intersection with several counting lines; per-line IN/OUT counts for people and vehicles in the HUD.
- Three live-webcam runs with the side panel, described below. Each has the annotated video (`*_counted.mp4`), the tracks (`*_tracks.jsonl`) and the final counts (`*_counts.json`).

The three runs share the same settings: `--imgsz 1920 --conf 0.15 --tracker configs/botsort_reid_static.yaml --frame-step 3 --layout side` with the default model. The videos here are re-encoded to H.264 at 1600 px wide without audio, so the panel text stays readable but the files stay small. The counts in the tables were produced by replaying the saved tracks with the current default `--min-cross-distance 0.05` (the detection runs themselves used the original rule, which is the same as `--min-cross-distance 0`).

## Counts changed in this version

Earlier versions counted a crossing as soon as one step touched the line. The default is now crossing hysteresis (`--min-cross-distance 0.05`): a person standing on a line, whose anchor jitters across it, is no longer counted. The tracks are the same; only the counting rule differs, so the totals are 6 to 7 percent lower (Abbey Road 105 -> 98, Shinjuku 90 s 366 -> 344, Shinjuku 3 min 673 -> 625). In a manual audit of Abbey Road this value kept all 79 verified crossings and removed 6 of 19 false ones; the other removed crossings in these runs were not judged by hand. See [the evaluation](../docs/how-it-works.md#evaluation-of-the-two-filters). `--min-cross-distance 0` reproduces the old counts. Totals per line, all classes and both directions:

**Shinjuku, 90 s** (all lines: 366 before, 344 after)

| Line | Before | After |
|---|---|---|
| Line_1 | 13 | 9 |
| Line_2 | 29 | 28 |
| Line_3 | 96 | 95 |
| Line_4 | 73 | 71 |
| Line_5 | 1 | 0 |
| Line_6 | 1 | 1 |
| Line_7 | 8 | 7 |
| Line_8 | 17 | 17 |
| Line_9 | 49 | 45 |
| Line_10 | 9 | 8 |
| Line_11 | 41 | 37 |
| Line_12 | 29 | 26 |

**Abbey Road, 3 min** (all lines: 105 before, 98 after)

| Line | Before | After |
|---|---|---|
| Line_1 | 23 | 21 |
| Line_2 | 28 | 26 |
| Line_3 | 43 | 40 |
| Line_4 | 11 | 11 |

**Shinjuku, 3 min** (all lines: 673 before, 625 after)

| Line | Before | After |
|---|---|---|
| Line_1 | 35 | 32 |
| Line_2 | 69 | 69 |
| Line_3 | 86 | 85 |
| Line_4 | 116 | 105 |
| Line_5 | 1 | 1 |
| Line_6 | 8 | 8 |
| Line_7 | 28 | 16 |
| Line_8 | 20 | 20 |
| Line_9 | 121 | 118 |
| Line_10 | 15 | 12 |
| Line_11 | 136 | 125 |
| Line_12 | 38 | 34 |


Accuracy (visual checks, no external ground truth): on Abbey Road about 80 to 87 percent of the 98 crossings counted here (default `--min-cross-distance 0.05`) were real, against about 74 to 82 percent of the 105 counted without the filter, and about 94 to 97.5 percent of the real crossings were counted in both cases (uncertain cases counted as wrong or as right give the ranges); the Shinjuku runs were only spot-checked. Stationary people at a line are the main source of false counts, dense groups are undercounted, and a rider can count twice (person and motorcycle) unless `--rider-overlap` is set. See [the evaluation](../docs/how-it-works.md#evaluation-of-the-two-filters).

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

Re-render without the detector (same counts as the table, any `--layout`; add `--min-cross-distance 0` for the old rule):

```bash
line-counter --input samples/shinjuku_90s.mp4 --lines samples/shinjuku_lines.json --no-display \
    --replay-tracks results/shinjuku_90s_tracks.jsonl --output outputs/shinjuku_90s_replay.mp4
```

| Line | Person IN / OUT | Other classes (IN / OUT, only non-zero) |
|---|---|---|
| Line_1 | 0 / 9 | - |
| Line_2 | 7 / 21 | - |
| Line_3 | 48 / 47 | - |
| Line_4 | 31 / 40 | - |
| Line_5 | 0 / 0 | - |
| Line_6 | 0 / 1 | - |
| Line_7 | 3 / 4 | - |
| Line_8 | 8 / 9 | - |
| Line_9 | 4 / 2 | car 25 / 11, motorcycle 0 / 1, bus 1 / 0, truck 1 / 0 |
| Line_10 | 1 / 1 | car 0 / 5, truck 0 / 1 |
| Line_11 | 0 / 0 | car 13 / 23, motorcycle 1 / 0 |
| Line_12 | 2 / 1 | car 9 / 12, bus 0 / 1, truck 0 / 1 |

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

Re-render without the detector (same counts as the table, any `--layout`; add `--min-cross-distance 0` for the old rule):

```bash
line-counter --input samples/abbey_road_3min.mp4 --lines samples/abbey_road_lines.json --no-display \
    --replay-tracks results/abbey_road_3min_tracks.jsonl --output outputs/abbey_road_3min_replay.mp4
```

| Line | Person IN / OUT | Other classes (IN / OUT, only non-zero) |
|---|---|---|
| Line_1 | 8 / 13 | - |
| Line_2 | 14 / 12 | - |
| Line_3 | 6 / 3 | car 12 / 14, motorcycle 1 / 0, bus 1 / 0, truck 3 / 0 |
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

Re-render without the detector (same counts as the table, any `--layout`; add `--min-cross-distance 0` for the old rule):

```bash
line-counter --input samples/shinjuku_3min.mp4 --lines samples/shinjuku_lines.json --no-display \
    --replay-tracks results/shinjuku_3min_tracks.jsonl --output outputs/shinjuku_3min_replay.mp4
```

| Line | Person IN / OUT | Other classes (IN / OUT, only non-zero) |
|---|---|---|
| Line_1 | 17 / 15 | - |
| Line_2 | 35 / 34 | - |
| Line_3 | 27 / 58 | - |
| Line_4 | 33 / 72 | - |
| Line_5 | 1 / 0 | - |
| Line_6 | 5 / 3 | - |
| Line_7 | 6 / 10 | - |
| Line_8 | 9 / 11 | - |
| Line_9 | 11 / 5 | car 57 / 40, motorcycle 1 / 1, bus 0 / 1, truck 2 / 0 |
| Line_10 | 1 / 3 | car 2 / 5, bus 1 / 0 |
| Line_11 | 8 / 7 | car 43 / 58, motorcycle 0 / 2, bus 2 / 1, truck 1 / 3 |
| Line_12 | 5 / 2 | car 11 / 13, motorcycle 0 / 1, bus 0 / 1, truck 0 / 1 |

## Source footage

The three live-webcam runs use clips captured from public live streams (see [../samples/README.md](../samples/README.md)). `shinjuku_line_count.mp4` is a publicly available street recording that the author considers free to use.
<!-- TODO(author): add the link and license of the source video of shinjuku_line_count.mp4 here before publishing. -->

People appear in public spaces; the outputs are shown only to illustrate the method.
If you appear in a video and want it removed, please open an issue.

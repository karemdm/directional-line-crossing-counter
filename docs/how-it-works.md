# How it works

This document explains the counting geometry, the processing pipeline and how
to tune the parameters.

## Pipeline

```mermaid
flowchart TD
    A["Video frame"] --> B["YOLO detection<br/>(conf, imgsz, classes)"]
    B --> C["BoT-SORT tracking<br/>(persistent IDs)"]
    C --> D{"Face blur<br/>enabled?"}
    D -- yes --> E["Blur faces inside<br/>person boxes"]
    D -- no --> F
    E --> F["Anchor = bottom-center<br/>segment prev → curr"]
    F --> G{"Strict segment<br/>intersection with a line?"}
    G -- no --> J
    G -- yes --> H["Side test (cross product)<br/>→ IN / OUT"]
    H --> I["Count once per ID per line"]
    I --> J["Draw boxes, lines, HUD"]
    J --> K["Write frame to &lt;name&gt;_proc.mp4<br/>+ live preview"]
```

1. **Line definition.** The first frame is shown in a resizable OpenCV window.
   Each pair of left clicks creates a line (`Line_1`, `Line_2`, ...). Because
   the window can be resized or maximised, click coordinates are scaled back to
   the original video resolution:
   `x_orig = x_win * (W_orig / W_win)`, `y_orig = y_win * (H_orig / H_win)`.
   ENTER or ESC finishes. Lines can also be loaded from a JSON file
   (`--lines`) and exported (`--save-lines`).
2. **Detection and tracking.** Each frame goes through Ultralytics
   `model.track(..., persist=True, tracker="botsort.yaml")`, restricted to the
   COCO classes person (0), car (2), motorcycle (3), bus (5) and truck (7).
3. **Anchor point.** For every tracked box `(x1, y1, x2, y2)` the anchor is the
   bottom-center point `((x1 + x2) / 2, y2)`, roughly where feet or wheels
   touch the ground. The previous anchor of each track ID is stored, so every
   frame gives a short trajectory segment `C = prev -> D = curr`.
4. **Counting.** Each segment is tested against each counting line (below).
5. **Output.** Annotated frames are written to disk immediately, so stopping
   early (Q in the preview window, or Ctrl+C) keeps every frame processed so
   far in a readable file.

## Directional line crossing (IN / OUT)

The logic uses 2D analytic geometry on the anchor points.

### 1. Strict segment intersection

Testing against an infinite line would count objects that pass beside the end
of the drawn line. Instead, the trajectory segment `CD` (previous to current
position) must physically intersect the drawn segment `AB`.

This uses the counter-clockwise (CCW) orientation test of three points:

$$\text{CCW}(A, B, C) = (C_y - A_y)(B_x - A_x) > (B_y - A_y)(C_x - A_x)$$

Two segments intersect if and only if the endpoints of the trajectory lie on
opposite sides of the drawn line **and** the endpoints of the drawn line lie
on opposite sides of the trajectory:

$$\text{Intersects} = \big(\text{CCW}(A, C, D) \neq \text{CCW}(B, C, D)\big) \land \big(\text{CCW}(A, B, C) \neq \text{CCW}(A, B, D)\big)$$

Implemented in `line_counter.geometry.ccw` and `segments_intersect`.

### 2. Side classification

Once a crossing is confirmed, the direction comes from the sign of the 2D
cross product (determinant) of the directed line `AB` and the point `P`:

$$\text{Side}(P) = (B_x - A_x)(P_y - A_y) - (B_y - A_y)(P_x - A_x)$$

In image coordinates (y grows downward):

- `Side(P) < 0`: `P` is on the left of `A -> B` as seen on screen.
- `Side(P) > 0`: `P` is on the right of `A -> B` as seen on screen.
- `Side(P) = 0`: `P` is on the line.

Implemented in `line_counter.geometry.side_of_line`.

### 3. Direction rules

- **IN**: the current anchor is on the positive side (`curr_side > 0`), i.e.
  the anchor came from the negative side (or from exactly on the line).
- **OUT**: the current anchor is not on the positive side (`curr_side <= 0`),
  i.e. the anchor left the positive side.

For a step that strictly crosses the line this is the same as "negative to
positive is IN, positive to negative is OUT".

So the meaning of IN/OUT depends on the order in which the two points of a
line were clicked. Example: for a horizontal line drawn left to right, an
object moving down the screen is IN and an object moving up is OUT. Draw the
line in the opposite order to swap the labels.

### 4. Deduplication

Each line keeps a set of track IDs already counted (`counted_ids`). A track is
counted at most once per line, whatever its class or direction later. The same
ID can still be counted once on each of several lines. Counts are stored per
line, per class name, per direction.

### 5. Crossing hysteresis (optional, `--min-cross-distance`)

By default a crossing is counted as soon as one step touches the line. A person
standing on a line has a bottom-centre anchor that jitters by a few pixels, so
it can step across the line by chance and be counted. With
`--min-cross-distance F` the counter keeps, per track and per line, the last
*confirmed* side: the side where the anchor was at least `F x box height` away
from the line, measured perpendicular to it. A crossing is counted when the
anchor is confirmed on the other side and the path touched the line segment in
between. While the anchor stays inside the band it decides nothing, so jitter
never counts, and a slow walker is counted on the first frame where the distance
is reached, however long that takes. A track first seen inside the band has no
confirmed side yet and cannot count until it has been confirmed on one side and
then on the other. A step after a `--max-track-gap` absence updates the side
but cannot count. With `F = 0` the original rule above is used unchanged.

### 6. Riders (optional, `--rider-overlap`)

A person on a scooter or bicycle is often detected twice: as a person and as a
motorcycle (or bicycle). With `--rider-overlap R` a person crossing is not counted when
the intersection of the person box with a bicycle or motorcycle box is at least
`R` of the person box in the frame of the crossing; the track is still marked as
counted. COCO class 1 (bicycle) is not in the default `--classes`; class 3
(motorcycle) is. In replay the vehicle class must be present in the tracks file.

### Edge cases

- The CCW test is strict, so a point exactly on the line is grouped with the
  negative side. With integer pixel anchors this is uncommon but possible:
  - negative -> on the line: not a crossing yet; the following step
    on the line -> positive is counted as IN;
  - positive -> on the line: counted as OUT immediately; the following step
    on the line -> negative is not a crossing.

  Either way the object is counted once, in the direction of its overall
  movement. A trajectory collinear with the line is never counted.
- Exactly at an endpoint of the line the strict test stays asymmetric: a step
  from the positive side that ends exactly on the endpoint is counted as OUT,
  while a step that leaves the endpoint toward the positive side is not
  counted. A trajectory that only grazes an endpoint is ambiguous anyway.
- If the tracker loses an object and gives it a new ID, the object can be
  counted again.
- The class used for a count is the class in the frame where the crossing
  happens; class flicker between frames can move a count to another class.

## Evaluation of the two filters

**Method.** Every crossing counted on the Abbey Road run (`results/abbey_road_3min_tracks.jsonl`,
`samples/abbey_road_lines.json`, no filters) was judged by hand from frame
sheets (a few frames around each crossing with the track box and trail): 105
counted, 79 true positives (TP), 19 false positives (FP) and 7 uncertain (U).
The same was done for Line_3 of the first 30 s of the Shinjuku 90 s run: 34
counted, 27 TP, 4 FP, 3 U. The filters were replayed from the saved tracks (no
detector ran), and each event was matched by line and track ID. This is one
reviewer, one pass, two videos, so the figures are an indication and not a
validation; a crossing that was never counted (a miss) is not measured. The main
causes of FP were people standing on the kerb line (about 13 of 19), ID swaps,
and a scooter rider counted as person and as motorcycle.

**`--min-cross-distance` on Abbey Road** (all four lines):

| Value | Counted | TP kept (of 79) | FP removed (of 19) | U kept (of 7) |
|---|---|---|---|---|
| 0 (off) | 105 | 79 | 0 | 7 |
| 0.02 | 101 | 79 | 3 | 6 |
| 0.03 | 99 | 79 | 5 | 6 |
| 0.05 | 98 | 79 | 6 | 6 |
| 0.075 | 95 | 78 | 8 | 6 |
| 0.1 | 91 | 76 | 9 | 5 |
| 0.15 | 87 | 74 | 11 | 5 |
| 0.2 | 87 | 74 | 11 | 5 |
| 0.3 | 79 | 70 | 14 | 4 |

No value counted a crossing that the unfiltered run did not count. Most FP that
remain are people who walk up to the line and stop on it (the anchor drifts
slowly through the band, which looks like a slow crossing), ID swaps and the
scooter duplicate; distance alone cannot tell those from real crossings.

**On Shinjuku** (Line_3, first 30 s, 34 counted): values 0.02 to 0.2 kept all 34,
so no FP was removed and nothing was lost. At 0.3 three counts disappeared (tracks
543, 684 and 2281); in the frame sheets all three walk clearly through the line,
so they look like TP lost (a re-check of those three events, not the original
per-event verdicts). Over the whole videos, without verdicts, the filter removes
about 6 to 7 percent of the counted crossings at 0.05 and 14 to 20 percent at 0.2
(Shinjuku 90 s: 366 -> 344 at 0.05, 334 at 0.1, 314 at 0.2).

**Decision.** No value removed most of the FP while keeping nearly all TP. 0.05 is
the largest value that kept every TP and it removes 6 of 19 FP; beyond that real
crossings are lost. The default therefore stays `0` (off) and the counts in
`results/` are unchanged. `0.05` is a reasonable starting point for scenes where
people stand on the line.

**`--rider-overlap` on Abbey Road** (default classes, so motorcycles are the only
two-wheelers tracked; bicycles are not):

| Value | Counted | Counts removed |
|---|---|---|
| 0 (off) | 105 | none |
| 0.2 and 0.3 | 103 | person #943 (a scooter rider also counted as motorcycle #950) and person #1194 (an ID swap, not a rider) |
| 0.5 and 0.7 | 104 | person #943 only |
| 0.9 | 105 | none |

Only the scooter case where both boxes exist in the crossing frame is touched:
rider and scooter were two counts of one road user, and 0.5 leaves the
motorcycle count only. The other scooter cases (a rider counted as person with
no motorcycle box in that frame, and a label flipping between motorcycle and
person) are not affected. On both Shinjuku runs 0.3 and 0.5 changed nothing. One
event is too little to choose a default from, so it stays `0` (off); 0.5 is the
suggested value when riders are counted twice.

**Uncertain cases** on Abbey Road (seven): the person #119 walking from the road
onto the pavement on Line_1; on Line_2 the person #465 who stands at the kerb and
steps down and back, and #817, #1100, #1087 and #1144 (people at the kerb who
start moving slowly or walk along it); on Line_3 the person #547 who walks to the
end of the line and stops on it. The three uncertain cases of Shinjuku are not listed
individually.

## Optional face blurring

With `--blur-faces`, every detected person box (with or without a track ID)
is cropped and passed to a separate YOLO face detector (confidence 0.3); the
crops of one frame are sent to the face model in batches of up to 16. Each detected face region is replaced by a
Gaussian blur (kernel 51x51, sigma 30). Blurring happens before boxes and text
are drawn. This is best-effort: faces the detector misses are not blurred, and
faces outside detected person boxes are never searched.

## Parameter guide and tuning tips

| Parameter | Default | Notes |
|---|---|---|
| `--conf` | `0.01` | Very permissive; keeps weak detections so the tracker rarely loses objects, but adds false positives. Values around 0.1-0.25 are a common starting point. |
| `--imgsz` | `2048` | Large inference size helps small/distant objects; slower and needs more GPU memory. Try 1280 or 640 on CPU or for close-up scenes. |
| `--model` | `weights/yolov8m.pt` | Smaller models (`yolov8n.pt`, `yolov8s.pt`) are faster, larger ones (`yolov8l.pt`, `yolov8x.pt`) may detect more. |
| `--tracker` | `botsort.yaml` | `bytetrack.yaml` is a lighter alternative. Custom YAML files can tune track buffer and matching thresholds. |
| `--classes` | `0 2 3 5 7` | Any COCO class IDs supported by the model. |
| `--min-cross-distance` | `0` | Fraction of the box height; 0.05 is a starting point for people who stand on a line (see the evaluation above). |
| `--rider-overlap` | `0` | Fraction of the person box; 0.5 is a starting point when riders are counted twice. Needs class 1 or 3 in `--classes`. |

Tips:

- Place lines where objects are fully visible and move across the line, not
  along it. Avoid lines near the frame border, where boxes are clipped.
- Because the anchor is the bottom-center of the box, draw lines on the ground
  plane (floor, road) rather than at head height.
- Make lines long enough to cover the whole passage; trajectories that pass
  beside a segment end are intentionally not counted.
- Low frame rates produce long trajectory steps; the segment test still works,
  but the tracker is more likely to switch IDs.

# SO-HOTA-TP

The Track 2 ranking metric, proposed for the MVA2027 SMOTP4SB Challenge. It
adapts SO-HOTA, the Track 1 metric introduced for MVA2025, to trajectory
prediction. Its full definition will be published in the MVA2027 challenge
report; please cite that report when you use it.

`evaluation/so_hota_tp.py` is the implementation, and the leaderboard runs the
same file.

## Why not minADE / minFDE

ADE and FDE need a ground-truth position at every predicted frame. In the
training data 41% of the birds observed in a window are not visible for all 150
frames that follow, and over the missing frames ADE and FDE are undefined.
Scoring only the visible frames would make emitting all 150 positions free,
however wrong the guess about the bird's survival. SO-HOTA-TP scores the absent
frames too.

## Definition

```
SO-HOTA-TP(α) = sqrt( SO-DetA-TP(α) × SO-AssA-TP(α) )
SO-HOTA-TP    = mean of SO-HOTA-TP(α) over α ∈ {0.05, 0.10, ..., 0.95}
```

- **Similarity** is Dot Distance, `DotD = exp(-D / S)`, where `D` is the
  distance between centres and `S = sqrt(mean ground-truth box area)`, computed
  once over the whole split. A predicted position is a true positive at
  threshold α when `DotD ≥ α`.
- **SO-DetA-TP** = TP / (TP + FN + FP), over predicted frames.
- **SO-AssA-TP** measures whether the same ground-truth bird stays with the same
  predicted track across the horizon, as AssA does in HOTA.
- **SO-LocA-TP**, **SO-DetRe-TP**, **SO-DetPr-TP**, **SO-AssRe-TP** and
  **SO-AssPr-TP** are reported alongside, named as in SO-HOTA with a `-TP`
  suffix.

### Per frame

| bird | prediction | counts as |
|---|---|---|
| visible | present, DotD ≥ α | TP |
| visible | present, DotD < α | FN and FP |
| visible | absent | FN |
| absent | present | FP |
| absent | absent | nothing |

### Best-of-K

For each track-window, the mode with the highest mean Dot Distance over the
frames where the bird is visible is selected, and only that mode is scored.
Endpoint (FDE-style) selection is not used, since the endpoint is undefined when
the bird disappears before it.

### Which predicted track is which bird

Submissions use their own track ids. For each window, ids are matched to
ground-truth ids by Hungarian assignment over the window's 60 observed frames,
using the submission's tracking file, with the mean centre distance as cost.
Pairs further apart than `-S·ln(0.05)` are not matched, nor are tracks sharing
fewer than 5 observed frames. The predicted frames never take part in matching.

### Across windows and sequences

Windows are cut every 150 frames from frame 1 of each sequence
([track2.md](track2.md#windows)). TP, FN and FP are summed over all windows and
sequences before the ratios are taken, and association scores are averaged
weighted by TP, as TrackEval's `combine_sequences` does,
so a sequence with many windows weighs more than one with few. A sequence with
no prediction file is scored as all misses.

## Supplementary metrics

minADE₆, minFDE₆ and Miss Rate are computed over the visible frames and
reported for comparison with the trajectory-prediction literature. They do not
affect the ranking.

## Running it

One sequence:

```bash
python -m evaluation.so_hota_tp \
    --pred prediction/0002.txt --tracks tracking/0002.txt --gt gt/0002.txt \
    --output-json result.json
```

`--gt` is a MOT file (`frame, id, x, y, w, h, ...`). `S` is computed from it
unless `--S` is given; pass the split-wide value to reproduce leaderboard
numbers on a single sequence.

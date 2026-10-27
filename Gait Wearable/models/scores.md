# Gait Wearable: results

14 recordings, 533 windows of 2 s (every 1 s), scored leave-one-recording-out.

| Activity | Recordings | Windows |
|---|---|---|
| standing | 4 | 72 |
| walking | 4 | 208 |
| running | 3 | 155 |
| biking | 3 | 98 |

| Model | Windows right |
|---|---|
| model 1 (peak) | 98.7% |
| model 2 (decision tree) | 97.7% |
| model 2 (k-nearest neighbors) (kept) | 98.3% |
| model 2 (nearest centroid) | 98.1% |
| model 2 (k-means) | 98.3% |

Model 1 thresholds (peak ax): biking < 0.88 g <= standing < 1.16 g <= walking < 2.34 g <= running

Decision tree trained on all windows (its questions explain what separates the activities):

```
|--- amag_std <= 0.45
|   |--- ax_max <= 1.16
|   |   |--- ay_max <= 0.33
|   |   |   |--- class: standing
|   |   |--- ay_max >  0.33
|   |   |   |--- class: biking
|   |--- ax_max >  1.16
|   |   |--- ax_std <= 0.30
|   |   |   |--- az_mean <= -0.04
|   |   |   |   |--- class: walking
|   |   |   |--- az_mean >  -0.04
|   |   |   |   |--- class: walking
|   |   |--- ax_std >  0.30
|   |   |   |--- class: walking
|--- amag_std >  0.45
|   |--- class: running
```

Figures: `peak_hist.png` (model 1), `confusion.png` (all models), `examples.png` (raw signals).

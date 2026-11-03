# Step counter scores

Leave-one-recording-out: each recording is counted by a net trained on the others.

| recording | true steps | neural net | peak detector |
|---|---|---|---|
| walking_20261031-163149.csv | 72 | 85 (+18.1%) | 70 (-2.8%) |
| walking_20261031-164123.csv | 50 | 57 (+14.0%) | 49 (-2.0%) |
| walking_20261031-164227.csv | 56 | 62 (+10.7%) | 49 (-12.5%) |
| walking_20261031-164324.csv | 35 | 28 (-20.0%) | 24 (-31.4%) |
| walking_20261031-165133.csv | 60 | 71 (+18.3%) | 59 (-1.7%) |
| walking_20261031-165235.csv | 18 | 23 (+27.8%) | 15 (-16.7%) |
| walking_20261031-165302.csv | 24 | 29 (+20.8%) | 22 (-8.3%) |
| walking_20261031-170404.csv | 62 | 68 (+9.7%) | 62 (+0.0%) |
| walking_20261031-171325.csv | 60 | 52 (-13.3%) | 45 (-25.0%) |
| walking_20261031-171430.csv | 60 | 67 (+11.7%) | 51 (-15.0%) |
| walking_20261031-172840.csv | 50 | 55 (+10.0%) | 46 (-8.0%) |
| walking_20261031-173118.csv | 32 | 35 (+9.4%) | 30 (-6.2%) |
| walking_20261102-214308.csv | 100 | 110 (+10.0%) | 99 (-1.0%) |
| walking_20261102-215447.csv | 90 | 98 (+8.9%) | 90 (+0.0%) |
| walking_20261102-215808.csv | 100 | 115 (+15.0%) | 102 (+2.0%) |

Mean absolute count error: neural net **14.5%**, peak detector 8.8%.

Recordings with no steps (every step counted is false):

| recording | seconds | neural net | peak detector |
|---|---|---|---|
| standing_20261102-220917.csv (standing) | 72 | 5 | 36 |
| standing_20261102-221117.csv (standing) | 52 | 6 | 7 |

Step-curve threshold sweep (same held-out predictions; 0.35 is the untuned default):

| threshold | mean abs. count error | mean signed error | false steps on non-step recordings |
|---|---|---|---|
| 0.35 | 14.5% | +10.1% | 11 |
| 0.4 | 13.2% | +8.7% | 10 |
| 0.45 | 11.4% | +6.5% | 9 |
| 0.5 | 10.2% | +5.2% | 9 |
| 0.55 | 9.2% | +3.7% | 8 |
| 0.6 | 8.3% | +2.9% | 8 |
| 0.7 | 7.5% | +0.9% | 6 |

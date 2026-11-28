# Town02 re-evaluation after the ground-truth patch

Code: `deployment` at `cc5cff0da` (includes patch `47c77c22d`). Checkpoint: `final/latest.pt`, unchanged. Camera perception (`external`), held-out Town02, one scenario per process, on the Vast server.

**Runs finished: 32. Successful: 22 (69%).** Scenarios 1 to 6 are the original suite; 7 to 16 are new.

## Result per scenario

| # | Weather | Route m | Encounters | Before patch | Run 1 | Run 2 | Passes |
|---|---|---|---|---|---|---|---|
| 1 | ClearNoon | 425 | ACAC | Pass | Fail 60% (offroad) | Fail 59% (collision) | 0 of 2 |
| 2 | WetSunset | 503 | VACAVC | Pass | Pass | Pass | 2 of 2 |
| 3 | SoftRainNoon | 482 | ACA | Pass | Pass | Pass | 2 of 2 |
| 4 | CloudyNoon | 463 | VACAVC | Pass | Pass | Pass | 2 of 2 |
| 5 | ClearNight | 485 | CAC | Pass | Pass | Fail 44% (pedestrian_collision) | 1 of 2 |
| 6 | HardRainNight | 556 | VACAVC | Pass | Pass | Pass | 2 of 2 |
| 7 | ClearSunset | 416 | VCACV | new | Fail 66% (inactivity_timeout) | Fail 89% (collision) | 0 of 2 |
| 8 | CloudyNoon | 531 | CCAVC | new | Fail 86% (collision) | Fail 84% (collision) | 0 of 2 |
| 9 | WetNoon | 425 | AVCAC | new | Pass | Pass | 2 of 2 |
| 10 | MidRainyNoon | 515 | VACVA | new | Pass | Pass | 2 of 2 |
| 11 | CloudySunset | 565 | CACAC | new | Pass | Pass | 2 of 2 |
| 12 | SoftRainSunset | 647 | VCACV | new | Pass | Fail 46% (red_light_violation) | 1 of 2 |
| 13 | WetCloudyNoon | 533 | ACVAC | new | Fail 38% (inactivity_timeout) | Pass | 1 of 2 |
| 14 | ClearNight | 476 | VACA | new | Fail 64% (collision) | Pass | 1 of 2 |
| 15 | WetNight | 424 | CVAC | new | Pass | Pass | 2 of 2 |
| 16 | MidRainyNight | 404 | ACVCA | new | Pass | Pass | 2 of 2 |

Encounters: V = slow lead vehicle, A = adult crossing, C = child crossing, in route order.

## Metrics per run

| # | Run | Result | Route done | Distance m | Sim time s | Collisions | Pedestrian incidents | Red-light violations | Lane departures | Speeding s | Min separation m | Supervisor override | Lane from camera | Mean lane error m | p95 lane error m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | Fail: offroad | 59.6% | 254 | 35 | 0 | 0 | 0 | 1 | 0.3 | 0.34 | 47% | 84% | 0.28 | 0.65 |
| 1 | 2 | Fail: collision | 58.8% | 250 | 35 | 1 | 0 | 0 | 0 | 0.4 | 0.71 | 47% | 85% | 0.22 | 0.68 |
| 2 | 1 | Pass | 99.5% | 501 | 184 | 0 | 0 | 0 | 0 | 0.4 | 2.01 | 86% | 43% | 0.21 | 0.40 |
| 2 | 2 | Pass | 99.5% | 501 | 182 | 0 | 0 | 0 | 0 | 0.6 | 1.99 | 87% | 42% | 0.20 | 0.40 |
| 3 | 1 | Pass | 99.4% | 481 | 103 | 0 | 0 | 0 | 0 | 1.3 | 1.41 | 68% | 66% | 0.23 | 0.74 |
| 3 | 2 | Pass | 99.4% | 481 | 106 | 0 | 0 | 0 | 0 | 1.1 | 1.70 | 68% | 66% | 0.26 | 0.77 |
| 4 | 1 | Pass | 99.4% | 460 | 133 | 0 | 0 | 0 | 0 | 1.3 | 3.96 | 73% | 66% | 0.13 | 0.41 |
| 4 | 2 | Pass | 99.4% | 460 | 133 | 0 | 0 | 0 | 0 | 1.3 | 3.96 | 77% | 66% | 0.11 | 0.42 |
| 5 | 1 | Pass | 99.4% | 482 | 110 | 0 | 0 | 0 | 0 | 1.9 | 1.69 | 68% | 65% | 0.23 | 0.42 |
| 5 | 2 | Fail: pedestrian_collision | 44.0% | 213 | 28 | 1 | 1 | 0 | 0 | 0.5 | 0.25 | 61% | 82% | 0.21 | 0.57 |
| 6 | 1 | Pass | 99.5% | 554 | 233 | 0 | 0 | 0 | 0 | 0.9 | 4.26 | 83% | 43% | 0.17 | 0.41 |
| 6 | 2 | Pass | 99.5% | 553 | 228 | 0 | 0 | 0 | 0 | 1.0 | 4.59 | 85% | 43% | 0.16 | 0.40 |
| 7 | 1 | Fail: inactivity_timeout | 65.9% | 274 | 132 | 0 | 0 | 0 | 0 | 1.2 | 2.98 | 84% | 40% | 0.15 | 0.48 |
| 7 | 2 | Fail: collision | 89.2% | 373 | 169 | 1 | 0 | 0 | 0 | 1.0 | 0.00 | 81% | 37% | 0.14 | 0.41 |
| 8 | 1 | Fail: collision | 86.2% | 458 | 107 | 1 | 0 | 0 | 0 | 0.8 | 6.06 | 73% | 62% | 0.16 | 0.56 |
| 8 | 2 | Fail: collision | 84.3% | 448 | 107 | 1 | 0 | 0 | 0 | 1.0 | 6.53 | 72% | 62% | 0.14 | 0.40 |
| 9 | 1 | Pass | 99.3% | 421 | 141 | 0 | 0 | 0 | 0 | 0.0 | 1.81 | 80% | 61% | 0.13 | 0.41 |
| 9 | 2 | Pass | 99.3% | 421 | 143 | 0 | 0 | 0 | 0 | 0.0 | 1.43 | 80% | 64% | 0.18 | 0.70 |
| 10 | 1 | Pass | 99.4% | 510 | 151 | 0 | 0 | 0 | 0 | 0.0 | 2.19 | 77% | 59% | 0.29 | 0.76 |
| 10 | 2 | Pass | 99.5% | 512 | 154 | 0 | 0 | 0 | 0 | 0.0 | 1.92 | 82% | 55% | 0.35 | 0.73 |
| 11 | 1 | Pass | 99.5% | 562 | 113 | 0 | 0 | 0 | 0 | 0.7 | 0.04 | 51% | 69% | 0.17 | 0.64 |
| 11 | 2 | Pass | 99.6% | 562 | 113 | 0 | 0 | 0 | 0 | 0.7 | 0.13 | 54% | 68% | 0.15 | 0.35 |
| 12 | 1 | Pass | 99.6% | 645 | 198 | 0 | 0 | 0 | 0 | 0.0 | 7.12 | 68% | 57% | 0.28 | 0.64 |
| 12 | 2 | Fail: red_light_violation | 46.0% | 297 | 71 | 0 | 0 | 1 | 0 | 0.0 | 2.28 | 52% | 81% | 0.44 | 0.83 |
| 13 | 1 | Fail: inactivity_timeout | 38.4% | 204 | 135 | 0 | 0 | 0 | 0 | 0.0 | 16.97 | 84% | 34% | 0.14 | 0.31 |
| 13 | 2 | Pass | 99.5% | 529 | 178 | 0 | 0 | 0 | 0 | 0.6 | 2.51 | 72% | 56% | 0.19 | 0.59 |
| 14 | 1 | Fail: collision | 63.9% | 307 | 130 | 1 | 0 | 0 | 0 | 0.0 | 3.02 | 62% | 60% | 0.21 | 0.49 |
| 14 | 2 | Pass | 99.4% | 476 | 157 | 0 | 0 | 0 | 0 | 0.0 | 2.43 | 63% | 65% | 0.23 | 0.58 |
| 15 | 1 | Pass | 99.3% | 422 | 190 | 0 | 0 | 0 | 0 | 0.0 | 3.86 | 75% | 55% | 0.28 | 0.61 |
| 15 | 2 | Pass | 99.3% | 422 | 189 | 0 | 0 | 0 | 0 | 0.0 | 3.85 | 68% | 61% | 0.28 | 0.62 |
| 16 | 1 | Pass | 99.3% | 402 | 124 | 0 | 0 | 0 | 0 | 0.5 | 3.95 | 74% | 56% | 0.25 | 0.62 |
| 16 | 2 | Pass | 99.4% | 402 | 124 | 0 | 0 | 0 | 0 | 0.0 | 3.75 | 74% | 58% | 0.28 | 0.64 |

## Totals over finished runs

| Measure | Value |
|---|---|
| Runs | 32 |
| Successful | 22 (69%) |
| Original scenarios 1 to 6 | 9 of 12 runs passed |
| New scenarios 7 to 16 | 13 of 20 runs passed |
| Mean route completion | 88.2% |
| Collisions | 6 |
| Pedestrian incidents | 1 |
| Red-light violations | 1 |
| Lane departures | 1 |
| Speeding, total | 17.2 s |
| Supervisor override, mean | 71% of steps |
| Lane signal from the camera road model, mean | 60% of steps |
| Failure endings | offroad: 1, collision: 5, pedestrian_collision: 1, inactivity_timeout: 2, red_light_violation: 1 |

## Before the patch (team's earlier report, scenarios 1 to 6)

| # | Result | Route done | Collisions | Lane departures | Supervisor override |
|---|---|---|---|---|---|
| 1 | Pass | 99.3% | 0 | 0 | 56% |
| 2 | Pass | 99.5% | 0 | 0 | 86% |
| 3 | Pass | 99.5% | 0 | 0 | 64% |
| 4 | Pass | 99.4% | 0 | 0 | 72% |
| 5 | Pass | 99.4% | 0 | 0 | 63% |
| 6 | Pass | 99.5% | 0 | 0 | 83% |

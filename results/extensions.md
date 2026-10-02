### Step-size limit (codimension 1)

60 sliding cases re-run with the ODE time horizon fixed (600 time units).

| step size | median \|occupancy − α\| | mean | correlation with α | churn (flips per update, median) |
|---|---|---|---|---|
| 0.008 | 0.065 | 0.113 | 0.83 | 0.0343 |
| 0.004 | 0.063 | 0.088 | 0.91 | 0.0275 |
| 0.002 | 0.041 | 0.071 | 0.93 | 0.0241 |
| 0.001 | 0.029 | 0.057 | 0.95 | 0.0234 |
| 0.0005 | 0.014 | 0.040 | 0.97 | 0.0232 |

### Moving target (Polyak averaging at rate β, step size 0.002)

| β | cases | still churning (> 0.005 flips per update) | churn (median) | late windows with mixed greedy action (median) |
|---|---|---|---|---|
| 0 | 30 | 100% | 0.0357 | 1.00 |
| 1e-05 | 30 | 13% | 0.0000 | 0.00 |
| 0.0001 | 30 | 7% | 0.0000 | 0.00 |

### Codimension 2

45 instances with two states chattering at once and a non-empty Filippov feasible set (scan of 1,500 instances at each of ω = 1.0 and 1.3). Occupancies are over the four sign patterns; distances are L1.

| quantity | median |
|---|---|
| width of the feasible set (largest range of any pattern's weight) | 0.144 |
| distance of the Euler occupancy to the feasible set, dt = 0.01 | 0.0054 |
| same, dt = 0.0025 | 0.0023 |
| change in Euler occupancy between dt = 0.01 and 0.0025 | 0.0076 |
| distance of the stochastic occupancy to the feasible set, a = 0.002 | 0.146 |
| same, a = 0.0005 | 0.083 |
| stochastic (a = 0.002) vs Euler (dt = 0.0025) occupancy | 0.344 |
| stochastic (a = 0.0005) vs Euler (dt = 0.0025) occupancy | 0.230 |
| product of codimension-1 weights vs Euler occupancy (12 cases with both weights defined) | 0.510 |


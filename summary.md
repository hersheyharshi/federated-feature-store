# Results summary

Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]. Settings are fixed in `Config`. The neural-network settings were compared on two spare split seeds of the same patients (not an independent test).

## Data checks

Values outside the schema range (count):

|  | age | sex | chest_pain | resting_bp | cholesterol | fasting_sugar | rest_ecg | max_heart_rate | exercise_angina | st_depression | st_slope | major_vessels | thal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cleveland | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Hungary | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Switzerland | 0 | 0 | 0 | 0 | 123 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| VA Long Beach | 0 | 0 | 0 | 1 | 49 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Share of each input missing after cleaning:

|  | age | sex | chest_pain | resting_bp | cholesterol | fasting_sugar | rest_ecg | max_heart_rate | exercise_angina | st_depression | st_slope | major_vessels | thal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cleveland | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.01 | 0.01 |
| Hungary | 0.0 | 0.0 | 0.0 | 0.0 | 0.08 | 0.03 | 0.0 | 0.0 | 0.0 | 0.0 | 0.65 | 0.99 | 0.9 |
| Switzerland | 0.0 | 0.0 | 0.0 | 0.02 | 1.0 | 0.61 | 0.01 | 0.01 | 0.01 | 0.05 | 0.14 | 0.96 | 0.42 |
| VA Long Beach | 0.0 | 0.0 | 0.0 | 0.28 | 0.28 | 0.04 | 0.0 | 0.26 | 0.26 | 0.28 | 0.51 | 0.99 | 0.83 |

## Main model: logistic regression

### Test accuracy (mean ± std over seeds)

|  | Cleveland | Hungary | Switzerland | VA Long Beach | Average |
|---|---|---|---|---|---|
| Local, at home | 82.5% ± 2.3% | 84.9% ± 3.1% | 93.2% ± 2.3% | 75.0% ± 5.7% | 83.9% ± 2.0% |
| Local, away (avg of other 3) | 76.9% ± 1.4% | 67.2% ± 4.4% | 52.6% ± 0.6% | 79.1% ± 3.2% | 69.0% ± 1.5% |
| Pooled (ceiling) | 83.6% ± 3.2% | 84.4% ± 3.1% | 78.6% ± 8.7% | 76.8% ± 5.2% | 80.9% ± 2.9% |
| Pooled + local calibration | 83.5% ± 3.2% | 83.8% ± 3.3% | 92.7% ± 1.8% | 76.8% ± 6.0% | 84.2% ± 2.0% |
| Federated, own scaling | 84.1% ± 2.7% | 82.2% ± 3.4% | 57.8% ± 7.5% | 66.2% ± 8.1% | 72.6% ± 3.0% |
| Federated, shared feature store | 83.5% ± 2.6% | 84.9% ± 2.8% | 77.3% ± 7.0% | 78.2% ± 6.3% | 81.0% ± 2.9% |
| Federated + store + local calibration | 83.5% ± 2.7% | 83.7% ± 3.4% | 93.0% ± 2.3% | 77.7% ± 5.8% | 84.5% ± 2.0% |
| Lazy guess | 53.8% ± 0.0% | 64.0% ± 0.0% | 94.6% ± 0.0% | 75.0% ± 0.0% | 71.9% ± 0.0% |

### AUC (not fooled by the lazy-guess trap)

|  | Cleveland | Hungary | Switzerland | VA Long Beach | Average |
|---|---|---|---|---|---|
| Local, at home | 0.910 ± 0.023 | 0.910 ± 0.033 | 0.577 ± 0.153 | 0.705 ± 0.123 | 0.776 ± 0.056 |
| Pooled (ceiling) | 0.914 ± 0.023 | 0.906 ± 0.031 | 0.579 ± 0.208 | 0.742 ± 0.110 | 0.785 ± 0.056 |
| Federated, own scaling | 0.918 ± 0.023 | 0.910 ± 0.026 | 0.583 ± 0.198 | 0.723 ± 0.117 | 0.784 ± 0.056 |
| Federated, shared feature store | 0.916 ± 0.023 | 0.907 ± 0.031 | 0.599 ± 0.201 | 0.753 ± 0.110 | 0.793 ± 0.055 |

### Local grid (rows trained at, columns tested at)

|  | Cleveland | Hungary | Switzerland | VA Long Beach |
|---|---|---|---|---|
| Cleveland | 82.5% ± 2.3% | 82.6% ± 3.1% | 74.9% ± 4.2% | 73.3% ± 4.6% |
| Hungary | 79.8% ± 2.6% | 84.9% ± 3.1% | 50.5% ± 9.4% | 71.3% ± 6.7% |
| Switzerland | 46.9% ± 1.6% | 36.5% ± 1.9% | 93.2% ± 2.3% | 74.5% ± 1.6% |
| VA Long Beach | 75.6% ± 5.0% | 81.8% ± 4.3% | 80.0% ± 7.2% | 75.0% ± 5.7% |

### Paired comparisons (difference per seed; 95% t-interval over seeds)

Seeds re-split the same patients, so intervals reflect split and training randomness, not new patients.

| comparison | mean difference (points / AUC) | 95% CI | positive in seeds |
|---|---|---|---|
| Shared store vs own scaling | 8.4 | [7.4, 9.4] | 10/10 |
| Federated (shared store) vs pooled ceiling | 0.1 | [-0.7, 0.9] | 6/10 |
| Federated (shared store) vs local at home | -2.9 | [-4.1, -1.8] | 0/10 |
| Federated + calibration vs local at home | 0.5 | [0.0, 1.0] | 9/10 |
| Federated + calibration vs federated, Switzerland | 15.7 | [10.9, 20.5] | 10/10 |
| Federated + calibration vs local at home, Switzerland | -0.3 | [-2.6, 2.0] | 1/10 |
| Federated + calibration vs local at home, VA Long Beach | 2.7 | [1.2, 4.2] | 9/10 |
| Federated (shared store) vs local at home, AUC | 0.018 | [-0.013, 0.049] | 8/10 |
| Federated (shared store) vs pooled, AUC | 0.008 | [0.003, 0.014] | 9/10 |

### Privacy versus accuracy (federated, shared store, delta=1e-5)

| epsilon | accuracy | with local calibration |
|---|---|---|
| 0.5 | 78.1% ± 1.9% | 82.9% ± 1.7% |
| 1.0 | 79.8% ± 2.5% | 84.1% ± 1.9% |
| 2.0 | 80.6% ± 2.5% | 84.7% ± 1.9% |
| 4.0 | 80.5% ± 2.7% | 84.9% ± 1.9% |
| 8.0 | 80.8% ± 2.6% | 84.6% ± 2.1% |
| inf | 81.0% ± 2.9% | 84.5% ± 2.0% |

### Cost of privacy, paired by seed

| epsilon | change vs no privacy (points) | 95% CI |
|---|---|---|
| 0.5 | -2.9 | [-4.9, -0.8] |
| 1.0 | -1.2 | [-2.3, -0.1] |
| 2.0 | -0.4 | [-1.2, 0.5] |
| 4.0 | -0.4 | [-1.4, 0.5] |
| 8.0 | -0.2 | [-0.9, 0.5] |

### Making the shared feature-store summaries private too (training epsilon fixed at 4)

summary epsilon = inf means exact (non-private) summaries.

| summary epsilon | total epsilon (training + summaries) | accuracy |
|---|---|---|
| 1.0 | 5.0 | 58.3% ± 11.3% |
| 3.0 | 7.0 | 60.7% ± 6.0% |
| 10.0 | 14.0 | 57.6% ± 7.1% |
| 30.0 | 34.0 | 62.1% ± 14.4% |
| 100.0 | 104.0 | 80.5% ± 2.6% |
| inf | 4.0 | 80.5% ± 2.7% |

### Forgetting and recovery (p_recover = 0.9)

| strategy | acc_old_before | acc_new_before | acc_old_after | acc_new_after | forgetting | both_stages_after |
|---|---|---|---|---|---|---|
| No protection | 79.8% ± 4.6% | 51.0% ± 3.9% | 47.2% ± 5.7% | 76.6% ± 4.6% | 32.6% ± 7.0% | 61.9% ± 3.4% |
| Replay | 79.8% ± 4.6% | 51.0% ± 3.9% | 70.1% ± 6.0% | 61.0% ± 3.8% | 9.7% ± 3.7% | 65.5% ± 3.7% |
| EWC (lambda=1) | 79.8% ± 4.6% | 51.0% ± 3.9% | 57.2% ± 9.9% | 72.4% ± 6.3% | 22.5% ± 9.1% | 64.8% ± 4.9% |
| EWC (lambda=10) | 79.8% ± 4.6% | 51.0% ± 3.9% | 78.9% ± 4.2% | 52.9% ± 4.0% | 0.8% ± 1.2% | 65.9% ± 3.4% |
| EWC (lambda=100) | 79.8% ± 4.6% | 51.0% ± 3.9% | 79.7% ± 4.7% | 51.1% ± 4.1% | 0.1% ± 0.6% | 65.4% ± 3.6% |

### Paired against no protection (95% CI)

| strategy vs no protection | forgetting avoided (points) | new-stage accuracy change (points) | mean-of-both-stages change (points) |
|---|---|---|---|
| Replay | 22.9 [18.9, 26.9] | -15.6 [-19.5, -11.7] | 3.6 [1.3, 6.0] |
| EWC (lambda=1) | 10.0 [5.6, 14.4] | -4.2 [-7.7, -0.7] | 2.9 [0.8, 5.0] |
| EWC (lambda=10) | 31.7 [26.7, 36.7] | -23.7 [-28.1, -19.4] | 4.0 [0.6, 7.3] |
| EWC (lambda=100) | 32.5 [27.5, 37.4] | -25.5 [-30.4, -20.6] | 3.5 [-0.0, 7.0] |

### Drift alarm (DDM)
Detected after the drift in 8/10 seeds (median delay 27 patients, range 8 to 113); false alarms before the drift: 2/10.

### Drift bound check
Held in 10/10 seeds. Delta range 0.28 to 0.41.

### Gentler and stronger drift

| drift strength (p_recover) | strategy | old model on new stage | forgetting | new-stage accuracy after | alarm detected | median delay | false alarms |
|---|---|---|---|---|---|---|---|
| 0.3 | No protection | 68.9% | 3.8% ± 2.1% | 68.6% | 3/10 | 62 | 2 |
| 0.3 | Replay | 68.9% | 1.0% ± 1.5% | 68.9% | 3/10 | 62 | 2 |
| 0.3 | EWC (lambda=10) | 68.9% | 0.2% ± 0.9% | 68.8% | 3/10 | 62 | 2 |
| 0.6 | No protection | 60.7% | 18.0% ± 7.8% | 66.1% | 7/10 | 41 | 2 |
| 0.6 | Replay | 60.7% | 3.8% ± 2.2% | 64.4% | 7/10 | 41 | 2 |
| 0.6 | EWC (lambda=10) | 60.7% | 0.6% ± 0.8% | 62.0% | 7/10 | 41 | 2 |
| 0.9 | No protection | 51.0% | 32.6% ± 7.0% | 76.6% | 8/10 | 27 | 2 |
| 0.9 | Replay | 51.0% | 9.7% ± 3.7% | 61.0% | 8/10 | 27 | 2 |
| 0.9 | EWC (lambda=10) | 51.0% | 0.8% ± 1.2% | 52.9% | 8/10 | 27 | 2 |

### Membership-guessing attack (AUC 0.5 = attacker guesses; averaged over the 4 hospitals)

| model | attack AUC | train accuracy | test accuracy |
|---|---|---|---|
| DP eps=1 | 0.503 ± 0.042 | 75.3% | 74.1% |
| no privacy | 0.509 ± 0.046 | 86.6% | 82.8% |

## Robustness: small neural network (mlp16)

### Test accuracy (mean ± std over seeds)

|  | Cleveland | Hungary | Switzerland | VA Long Beach | Average |
|---|---|---|---|---|---|
| Local, at home | 81.2% ± 3.2% | 83.4% ± 3.2% | 94.1% ± 1.1% | 75.2% ± 7.0% | 83.5% ± 2.0% |
| Local, away (avg of other 3) | 75.5% ± 2.4% | 63.1% ± 6.1% | 52.4% ± 0.0% | 73.9% ± 4.8% | 66.2% ± 1.7% |
| Pooled (ceiling) | 81.2% ± 3.7% | 83.4% ± 2.9% | 80.3% ± 5.8% | 76.2% ± 5.3% | 80.3% ± 2.8% |
| Pooled + local calibration | 81.8% ± 3.4% | 83.3% ± 3.2% | 92.2% ± 2.7% | 75.8% ± 5.0% | 83.3% ± 2.4% |
| Federated, own scaling | 81.0% ± 4.2% | 82.4% ± 3.5% | 58.6% ± 8.4% | 65.5% ± 9.0% | 71.9% ± 4.1% |
| Federated, shared feature store | 81.6% ± 2.8% | 82.8% ± 4.1% | 77.3% ± 8.2% | 78.0% ± 5.2% | 79.9% ± 2.9% |
| Federated + store + local calibration | 81.4% ± 3.5% | 82.5% ± 2.6% | 93.2% ± 1.4% | 78.5% ± 3.8% | 83.9% ± 1.6% |
| Lazy guess | 53.8% ± 0.0% | 64.0% ± 0.0% | 94.6% ± 0.0% | 75.0% ± 0.0% | 71.9% ± 0.0% |

### AUC (not fooled by the lazy-guess trap)

|  | Cleveland | Hungary | Switzerland | VA Long Beach | Average |
|---|---|---|---|---|---|
| Local, at home | 0.894 ± 0.028 | 0.899 ± 0.035 | 0.479 ± 0.170 | 0.689 ± 0.103 | 0.740 ± 0.056 |
| Pooled (ceiling) | 0.911 ± 0.026 | 0.902 ± 0.035 | 0.569 ± 0.232 | 0.699 ± 0.092 | 0.770 ± 0.066 |
| Federated, own scaling | 0.911 ± 0.028 | 0.905 ± 0.028 | 0.549 ± 0.239 | 0.705 ± 0.117 | 0.767 ± 0.064 |
| Federated, shared feature store | 0.912 ± 0.027 | 0.905 ± 0.037 | 0.583 ± 0.209 | 0.733 ± 0.104 | 0.783 ± 0.060 |

### Local grid (rows trained at, columns tested at)

|  | Cleveland | Hungary | Switzerland | VA Long Beach |
|---|---|---|---|---|
| Cleveland | 81.2% ± 3.2% | 79.9% ± 5.1% | 73.2% ± 7.6% | 73.3% ± 3.9% |
| Hungary | 68.0% ± 10.7% | 83.4% ± 3.2% | 53.0% ± 11.6% | 68.3% ± 8.2% |
| Switzerland | 46.2% ± 0.0% | 36.0% ± 0.0% | 94.1% ± 1.1% | 75.0% ± 0.0% |
| VA Long Beach | 66.9% ± 7.4% | 78.7% ± 4.6% | 76.2% ± 9.4% | 75.2% ± 7.0% |

### Paired comparisons (difference per seed; 95% t-interval over seeds)

Seeds re-split the same patients, so intervals reflect split and training randomness, not new patients.

| comparison | mean difference (points / AUC) | 95% CI | positive in seeds |
|---|---|---|---|
| Shared store vs own scaling | 8.1 | [6.3, 9.8] | 10/10 |
| Federated (shared store) vs pooled ceiling | -0.3 | [-2.1, 1.5] | 6/10 |
| Federated (shared store) vs local at home | -3.5 | [-5.7, -1.3] | 2/10 |
| Federated + calibration vs local at home | 0.5 | [-1.1, 2.0] | 6/10 |
| Federated + calibration vs federated, Switzerland | 15.9 | [9.9, 22.0] | 9/10 |
| Federated + calibration vs local at home, Switzerland | -0.8 | [-2.1, 0.5] | 1/10 |
| Federated + calibration vs local at home, VA Long Beach | 3.3 | [-1.3, 8.0] | 6/10 |
| Federated (shared store) vs local at home, AUC | 0.043 | [-0.003, 0.089] | 8/10 |
| Federated (shared store) vs pooled, AUC | 0.013 | [-0.007, 0.033] | 4/10 |

### Privacy versus accuracy (federated, shared store, delta=1e-5)

| epsilon | accuracy | with local calibration |
|---|---|---|
| 0.5 | 74.4% ± 3.5% | 79.8% ± 2.8% |
| 1.0 | 77.0% ± 3.5% | 81.6% ± 2.3% |
| 2.0 | 78.7% ± 2.9% | 82.6% ± 2.0% |
| 4.0 | 79.5% ± 3.2% | 82.8% ± 2.3% |
| 8.0 | 79.7% ± 3.7% | 83.1% ± 2.3% |
| inf | 79.9% ± 2.9% | 83.9% ± 1.6% |

### Cost of privacy, paired by seed

| epsilon | change vs no privacy (points) | 95% CI |
|---|---|---|
| 0.5 | -5.5 | [-8.4, -2.6] |
| 1.0 | -3.0 | [-5.3, -0.6] |
| 2.0 | -1.3 | [-2.2, -0.4] |
| 4.0 | -0.4 | [-1.5, 0.6] |
| 8.0 | -0.2 | [-1.6, 1.1] |

### Forgetting and recovery (p_recover = 0.9)

| strategy | acc_old_before | acc_new_before | acc_old_after | acc_new_after | forgetting | both_stages_after |
|---|---|---|---|---|---|---|
| No protection | 80.1% ± 4.0% | 51.5% ± 3.6% | 46.5% ± 6.8% | 75.2% ± 5.7% | 33.6% ± 7.6% | 60.9% ± 5.2% |
| Replay | 80.1% ± 4.0% | 51.5% ± 3.6% | 68.5% ± 8.2% | 62.0% ± 3.4% | 11.6% ± 6.6% | 65.3% ± 4.1% |
| EWC (lambda=1) | 80.1% ± 4.0% | 51.5% ± 3.6% | 49.3% ± 7.8% | 73.8% ± 5.8% | 30.8% ± 7.8% | 61.5% ± 4.6% |
| EWC (lambda=10) | 80.1% ± 4.0% | 51.5% ± 3.6% | 61.2% ± 9.7% | 66.8% ± 6.1% | 18.9% ± 7.7% | 64.0% ± 3.9% |
| EWC (lambda=100) | 80.1% ± 4.0% | 51.5% ± 3.6% | 74.1% ± 7.7% | 52.8% ± 8.4% | 6.0% ± 8.8% | 63.4% ± 3.3% |

### Paired against no protection (95% CI)

| strategy vs no protection | forgetting avoided (points) | new-stage accuracy change (points) | mean-of-both-stages change (points) |
|---|---|---|---|
| Replay | 22.0 [16.8, 27.1] | -13.2 [-16.6, -9.9] | 4.4 [1.7, 7.1] |
| EWC (lambda=1) | 2.7 [0.8, 4.6] | -1.5 [-3.7, 0.7] | 0.6 [-0.8, 2.1] |
| EWC (lambda=10) | 14.7 [9.7, 19.7] | -8.5 [-13.6, -3.3] | 3.1 [0.0, 6.2] |
| EWC (lambda=100) | 27.5 [20.2, 34.8] | -22.5 [-31.3, -13.6] | 2.5 [-2.3, 7.4] |

### Drift alarm (DDM)
Detected after the drift in 8/10 seeds (median delay 31 patients, range 9 to 117); false alarms before the drift: 2/10.

### Drift bound check
Held in 10/10 seeds. Delta range 0.28 to 0.41.

### Membership-guessing attack (AUC 0.5 = attacker guesses; averaged over the 4 hospitals)

| model | attack AUC | train accuracy | test accuracy |
|---|---|---|---|
| DP eps=1 | 0.498 ± 0.032 | 73.0% | 73.3% |
| no privacy | 0.574 ± 0.063 | 97.4% | 79.9% |

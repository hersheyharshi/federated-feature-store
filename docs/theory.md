# What we can prove and what we only measured

## 1. The privacy promise (PROVEN, under assumptions)

**Statement.** Fix a hospital H with n training patients. In federated training, H runs DP-SGD
for T = rounds x steps-per-round steps with batch sampling rate q = B/n, per-patient gradient
clipping at C, and Gaussian noise with standard deviation sigma x C. Then the sequence of local
models H releases to the server is (epsilon, delta)-differentially private with respect to adding
or removing one patient of H, where epsilon is computed by the Renyi-DP accountant for the
Poisson-subsampled Gaussian mechanism (`privacy.epsilon`). Averaging at the server and every later
step are post-processing and cost nothing extra. We calibrate sigma per hospital so that epsilon
equals the target, with delta = 1e-5.

**Assumptions (they are part of the promise).**
1. Batches are Poisson-sampled (each patient included independently with probability q). The code does this in DP mode.
2. Every per-patient gradient, including the bias term, is clipped to norm C before summing.
3. A patient appears in exactly one hospital. A person treated at two hospitals is two records.
4. The adversary sees every model released in every round, but not a hospital's internal state or raw data.
5. epsilon and delta are for the training of the model only. Settings are fixed in `Config` ahead of the reported runs. The neural network's settings were compared on two spare split seeds (100, 101) of the same 920 patients, with no privacy accounting. That is hyper-parameter selection on the same patients, so it is not covered by the guarantee and the reported network results are not an independent test.

**What it does NOT cover.**
- The shared feature-store summaries (count, sum, sum of squares). By default these are NOT private.
  `FeatureStore.from_stats(..., epsilon=e)` adds Laplace noise (sensitivities from the schema ranges,
  budget split over 3 x 13 queries) and costs a separate epsilon that adds to the training epsilon
  by basic composition. The reported epsilon sweep uses the default (no noise on the summaries), so
  its epsilon is the training epsilon only. We measured the noisy variant (report Section 3.3): with
  range-based sensitivities it is unusable (accuracy falls below the lazy guess unless the summary
  epsilon is about 100), so we do not claim a private feature store.
- The replay buffer and EWC. They reuse stage-A patients and so are not covered by the accountant;
  `local_train` refuses to combine them with DP. The forgetting study therefore runs without DP.
- Any attack outside the model: re-identification from the data itself, side channels, a malicious server.
- A bound on what an attacker actually achieves. epsilon is a worst-case guarantee. The membership test
  in the results is a measurement of one simple attacker.

**Say it plainly.** "Each patient's presence or absence changes the probability of any outcome of the
released models by at most a factor of e^epsilon (plus delta), assuming the list above."

## 2. The drift bound (PROVEN for our simulated drift, under one assumption)

**Statement.** Let P_A and P_B be the patient distributions before and after the change, with the
same input distribution P(x) and different label rules eta_A(x) = P_A(y=1|x), eta_B(x) = P_B(y=1|x).
For any fixed classifier h, with error err_S(h) = P_S(h(x) != y):

    err_B(h) <= err_A(h) + Delta,   Delta = E_x | eta_A(x) - eta_B(x) |.

**Proof.** Pointwise, P_S(h(x) != y | x) is eta_S(x) if h(x)=0 and 1 - eta_S(x) if h(x)=1. So its
difference between A and B is at most |eta_A(x) - eta_B(x)|. Take the expectation over x. QED.

**Assumption.** P(x) is unchanged. In our simulation this holds by construction (stage A and stage B
are a random split of the same patients, and only the labels of flagged patients change).

**In the experiment.** Delta is the fraction of test labels the simulated treatment flipped. We score the
frozen stage-A model on the SAME stage-B test patients under the old labels and the new labels, so only the
label rule differs (`error_increase_same_patients` in `drift_bound_check.csv`). For this construction the
bound is close to an identity, so the check is a sanity test of the simulation, not independent evidence.
The column `error_increase_across_samples` compares different patients and includes sampling noise of
about 4 points; do not use it to test the bound.

**When it fails.** If the inputs also shift (new patient mix), the bound needs an extra term for
the distance between P_A(x) and P_B(x) (Ben-David et al. 2010). In real hospitals Delta is unknown, so
this bound explains the simulation. It does not certify any real deployment.

## 3. Everything else is MEASURED

| Claim | Status | Where |
|---|---|---|
| Epsilon of the DP training | Proven (section 1); accountant matches the TF-Privacy reference value | `privacy.py`, `tests/` |
| Accuracy at each epsilon | Measured | `epsilon_sweep.csv`, `privacy_cost_paired.csv` |
| Local calibration closes the Switzerland gap | Measured; calibration is local post-processing, nothing is released | `paired_comparisons.csv` |
| Private feature-store summaries cost accuracy | Measured (negative result) | `store_privacy.csv` |
| Forgetting and detection depend on drift strength | Measured on simulated drift | `drift_severity.csv` |
| Results hold for a small neural network | Measured (one lightly tuned network, tuned on the same patients) | `results/mlp16/` |
| Federated lands between local and pooled | Measured | `baseline_table.csv` |
| Shared feature store changes federated accuracy | Measured | `baseline_table.csv` |
| Error increase under drift is at most Delta | Proven for the simulation; sanity-checked, held in 10 of 10 seeds | `drift_bound_check.csv` |
| Drift alarm delay | Measured on simulated drift | `drift_detection.csv` |
| Amount forgotten, replay vs EWC | Measured on simulated drift | `drift_forgetting.csv` |
| Membership attack success | Measured, one simple attacker | `membership_raw.csv` |
| Anything about real hospitals or clinical use | Not claimed | |

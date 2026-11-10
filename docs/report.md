# Federated Continual Feature Stores with Differential Privacy: A Simulation Study on Multi-Site Heart Disease Data

Harshita Pandita, Pranjal Tomar. November 2026. Code: this repository. All numbers: `results/summary.md`
(10 seeds; mean ± standard deviation, or paired mean difference with a 95% interval). Status: simulation study on public data, not clinically validated.

## Abstract
Hospitals cannot pool patient records, yet models trained at one hospital generalise poorly to another.
Using the four UCI Heart Disease sites as simulated hospitals, we train one shared model by federated
averaging over a versioned shared feature store, with differential privacy and protection against
catastrophic forgetting, for a logistic regression and a small neural network. Local models lose about 15
points of accuracy away from home (83.9% to 69.0%). A shared feature store lifts federated accuracy by 8.4
points [7.4, 9.4] over letting each hospital scale its inputs alone, in 10 of 10 seeds, and matches a pooled
ceiling (+0.1 [-0.7, 0.9]). A federated model alone is 2.9 points below local models at home, entirely at
Switzerland, where 93% of patients are sick; one number per hospital that matches its own base rate (local
calibration, never shared) closes that gap (+0.5 [0.0, 1.0] versus local-at-home on average). Privacy costs
little but is measurable at strong settings: 1.2 points at epsilon 1 and 2.9 at epsilon 0.5 for the logistic
model, more for the neural network. Making the feature-store summaries private too is not practical with
our simple noise mechanism: accuracy falls below the lazy guess unless the summary budget is about 100.
Under simulated concept drift an unprotected update forgets 33 points; replay cuts this to 10 and EWC to
near 0, but all protections trade new-stage accuracy for old, and we cannot rank them on the average of both
stages. The privacy guarantee is proven under stated assumptions; everything about drift and forgetting is
measured on drift we simulated.

## 1. Introduction
Hospitals hold records that could help detect heart disease earlier, but privacy law forbids pooling them.
Three problems follow: records cannot be shared; shared lessons can still leak facts about individuals; and
medicine changes, so models go stale. Research questions:
1. How much accuracy is lost as privacy tightens in a federated setting?
2. How much does a model forget when data changes, and how well do replay and EWC help?
3. How quickly can drift be detected, and does the answer depend on how strong the drift is?
4. What can we guarantee, and how well do experiments support it?

## 2. Method
**Data.** UCI Heart Disease: Cleveland (303 patients, 46% with disease), Hungary (294, 36%), Switzerland
(123, 93%), VA Long Beach (200, 74%). Each site is split 70/30 into practice and test patients. Seeds change
the split and the training randomness; the 920 patients are the same, so intervals reflect split and training
randomness, not sampling of new patients. Public data split by site behave like separate hospitals; this is a
simulation, not a deployment.

**Data quality.** Missingness is severe and uneven (after cleaning): `major_vessels` is missing for 99% of
Hungary, 96% of Switzerland and 99% of VA patients; `thal` for 90%, 42%, 83%; Switzerland has no cholesterol
at all (all zeros, read as "not recorded") and 61% missing fasting sugar. Missing values are filled with the
shared mean, so those inputs carry signal almost only at Cleveland. The Switzerland test set is about 37
patients of whom about 2 are healthy, so any Switzerland-specific metric, AUC especially, is very noisy.

**Feature store.** A versioned schema of 13 inputs (name, meaning, unit, valid range; `feature_store_v1.json`).
Impossible values become missing. Each hospital shares only per-input count, sum and sum of squares; the
server combines them into shared means and spreads.

**Models.** Main: logistic regression. Robustness: a one-hidden-layer network (16 ReLU units), with exact
per-patient gradients (unit-tested against numerical gradients). Both train by mini-batch gradient descent
inside federated averaging (20 rounds, 2 local epochs, hospitals weighted by patient count). The network's
settings were checked on two spare seeds (100, 101) outside the reported ones; three settings gave similar
results and the defaults were kept. It is not heavily tuned.

**Local calibration.** After training, each hospital fits one offset to the model's output on its own practice
patients so predicted and true base rates match. The offset never leaves the hospital and test patients are not
used. This is a local adjustment of a released model, not a new release.

**Privacy.** DP-SGD at every hospital (Poisson sampling, per-patient clipping at 1.0, Gaussian noise),
Renyi-DP accountant, noise calibrated per hospital so total epsilon over all rounds hits the target,
delta = 1e-5, epsilon in {0.5, 1, 2, 4, 8}. Our accountant reproduces the published TF-Privacy reference value
(epsilon 3.01 with the classic conversion; unit test) and uses the tighter Balle et al. conversion. Separately,
the feature-store summaries can be made private with Laplace noise whose scale comes from the schema ranges.

**Drift.** Each hospital's patients are split at random into stage A and stage B. In stage B a simulated new
treatment makes sick patients with a high value of the strongest continuous risk marker known from stage A
(`st_depression` in 8 seeds, `max_heart_rate` in 2) recover with probability p. Main setting p = 0.9; we also
run 0.6 and 0.3. Inputs are unchanged; only the label rule changes. A DDM detector watches the error stream of
the frozen stage-A model.

**Forgetting protection.** After learning stage A together, training continues on stage B with no protection,
replay (each hospital rehearses its own stage-A patients, which never leave), or EWC (federated Fisher estimate,
strengths 1, 10, 100). These run without DP (Section 5).

**Metrics.** Test accuracy next to the lazy guess (always the more common answer), and AUC, because Switzerland
is 93% sick. Forgetting = old-stage accuracy before minus after. Comparisons are paired by seed.

## 3. Results
### 3.1 Baselines (logistic regression, test accuracy)
| Setup | Average | Cleveland | Hungary | Switzerland | VA Long Beach |
|---|---|---|---|---|---|
| Local, at home | 83.9% ± 2.0 | 82.5 | 84.9 | 93.2 | 75.0 |
| Local, away | 69.0% ± 1.5 | 76.9 | 67.2 | 52.6 | 79.1 |
| Pooled (ceiling) | 80.9% ± 2.9 | 83.6 | 84.4 | 78.6 | 76.8 |
| Federated, own scaling | 72.6% ± 3.0 | 84.1 | 82.2 | 57.8 | 66.2 |
| Federated, shared feature store | 81.0% ± 2.9 | 83.5 | 84.9 | 77.3 | 78.2 |
| Federated + store + local calibration | 84.5% ± 2.0 | 83.5 | 83.7 | 93.0 | 77.7 |
| Lazy guess | 71.9% | 53.8 | 64.0 | 94.6 | 75.0 |

| Paired comparison (per seed) | Logistic | Neural network |
|---|---|---|
| Shared store minus own scaling | +8.4 [7.4, 9.4], 10/10 | +8.1 [6.3, 9.8], 10/10 |
| Federated minus pooled | +0.1 [-0.7, 0.9] | -0.3 [-2.1, 1.5] |
| Federated minus local at home | -2.9 [-4.1, -1.8], 0/10 | -3.5 [-5.7, -1.3], 2/10 |
| Federated + calibration minus local at home | +0.5 [0.0, 1.0], 9/10 | +0.5 [-1.1, 2.0], 6/10 |
| Calibration gain at Switzerland | +15.7 [10.9, 20.5], 10/10 | +15.9 [9.9, 22.0], 9/10 |
| Federated + calibration minus local, VA Long Beach | +2.7 [1.2, 4.2], 9/10 | +3.3 [-1.3, 8.0], 6/10 |

1. **Local helpers do not travel.** The Switzerland-trained helper is right 46.9% and 36.5% of the time at
   Cleveland and Hungary, below chance: it calls nearly everyone sick.
2. **The shared feature store matters** (both models, 10 of 10 and 10 of 10 seeds). Without it the federated
   model averages roughly the lazy guess (72.6% vs 71.9%).
3. **Federated matches pooled** within noise for both models.
4. **A federated model alone loses to local at Switzerland** (77.3% vs 93.2%): a local helper learns that site's
   93% base rate, a single global model cannot. Local calibration fixes this (+15.7 points, and -0.3 [-2.6, 2.0]
   versus local at home), and then federated plus calibration equals local on average and beats it at VA Long
   Beach for the logistic model. Caveat: calibrated accuracy at Switzerland (93.0%) is still below the 94.6% lazy
   guess, and its AUC is about 0.6 with a spread of 0.2 on about 37 test patients, so calibration works by
   matching the base rate, not by learning to rank those patients.
5. **AUC.** Federated minus local at home is +0.018 [-0.013, 0.049] (logistic) and +0.043 [-0.003, 0.089]
   (network); federated minus pooled is +0.008 [0.003, 0.014] and +0.013 [-0.007, 0.033]. Differences are small.

### 3.2 Privacy versus accuracy (`results/logistic/figures/privacy_curve.png`)
| epsilon | 0.5 | 1 | 2 | 4 | 8 | none |
|---|---|---|---|---|---|---|
| Logistic, accuracy | 78.1% | 79.8% | 80.6% | 80.5% | 80.8% | 81.0% |
| Logistic, with calibration | 82.9% | 84.1% | 84.7% | 84.9% | 84.6% | 84.5% |
| Logistic, change vs none (paired) | -2.9 [-4.9, -0.8] | -1.2 [-2.3, -0.1] | -0.4 [-1.2, 0.5] | -0.4 [-1.4, 0.5] | -0.2 [-0.9, 0.5] | |
| Network, accuracy | 74.4% | 77.0% | 78.7% | 79.5% | 79.7% | 79.9% |
| Network, change vs none (paired) | -5.5 [-8.4, -2.6] | -3.0 [-5.3, -0.6] | -1.3 [-2.2, -0.4] | -0.4 [-1.5, 0.6] | -0.2 [-1.6, 1.1] | |

Privacy costs little for the logistic model: not distinguishable from zero at epsilon 2 and above, 1.2 points at
epsilon 1, 2.9 at epsilon 0.5. The network pays more (3.0 and 5.5 points), plausibly because it has far more
parameters to protect with the same per-step noise. With local calibration, even epsilon 0.5 is only about 1.6
points below the best (82.9% vs 84.5%). All settings stay above the lazy guess. Epsilon covers training only.

### 3.3 Private feature-store summaries: a negative result
Making the shared summaries private with Laplace noise (training epsilon fixed at 4):

| Summary epsilon | 1 | 3 | 10 | 30 | 100 | exact (not private) |
|---|---|---|---|---|---|---|
| Accuracy | 58.3% ± 11.3 | 60.7% ± 6.0 | 57.6% ± 7.1 | 62.1% ± 14.4 | 80.5% ± 2.6 | 80.5% ± 2.7 |

Below a summary epsilon of 100 the model falls below the lazy guess (71.9%). The noise scale follows the schema
ranges (for example squares of cholesterol up to 700), which dwarfs the true sums. A summary epsilon of 100 gives
almost no protection. So with our mechanism a private feature store is not usable; tighter data-dependent
bounds or clipping would be needed. The reported training epsilons therefore exclude the summaries.

### 3.4 Drift, forgetting and protection (`results/logistic/figures/drift_forgetting.png`)
Logistic model, p = 0.9. After the change the stage-A model falls from 79.8% to 51.0% on the new stage (chance).

| Strategy | Old stage after | New stage after | Forgetting | Mean of both | Mean of both vs none (paired) |
|---|---|---|---|---|---|
| No protection | 47.2% ± 5.7 | 76.6% ± 4.6 | 32.6 ± 7.0 | 61.9% | |
| Replay | 70.1% ± 6.0 | 61.0% ± 3.8 | 9.7 ± 3.7 | 65.5% | +3.6 [1.3, 6.0] |
| EWC, lambda 1 | 57.2% ± 9.9 | 72.4% ± 6.3 | 22.5 ± 9.1 | 64.8% | +2.9 [0.8, 5.0] |
| EWC, lambda 10 | 78.9% ± 4.2 | 52.9% ± 4.0 | 0.8 ± 1.2 | 65.9% | +4.0 [0.6, 7.3] |
| EWC, lambda 100 | 79.7% ± 4.7 | 51.1% ± 4.1 | 0.1 ± 0.6 | 65.4% | +3.5 [-0.0, 7.0] |

Protection trades new-stage learning for old-stage memory. EWC at lambda 10 and 100 barely forgets because it
barely moves; its new-stage accuracy stays near 50%, where it started. Replay keeps most of the old stage and
learns something new (61.0%). Every protection beats no protection on the mean of both stages, but their intervals
overlap and we cannot rank them. For the neural network, replay is the clear best (forgetting 11.6 versus 33.6
unprotected, mean of both +4.4 [1.7, 7.1]), and EWC is weaker at the same strengths (lambda 10: forgetting 18.9).

**Drift strength.** Gentler drift forgets less and is harder to detect:

| p (drift strength) | Old model on new stage | Forgetting, no protection | Replay | EWC 10 | Alarm detected | Median delay |
|---|---|---|---|---|---|---|
| 0.3 | 68.9% | 3.8 | 1.0 | 0.2 | 3/10 | 62 patients |
| 0.6 | 60.7% | 18.0 | 3.8 | 0.6 | 7/10 | 41 |
| 0.9 | 51.0% | 32.6 | 9.7 | 0.8 | 8/10 | 27 |

The alarm missed most gentle drift and raised a false alarm before the drift in 2 of 10 seeds (the same two
seeds in every setting, since the pre-drift stream is identical). At p = 0.9, delays ranged 8 to 113 patients.
Our main setting is therefore the harshest; real drift is usually gentler, where detection is least reliable.

### 3.5 Membership-guessing test
A loss-threshold attacker, averaged over the four hospitals: logistic regression, attack AUC 0.509 ± 0.046 with no
privacy and 0.503 ± 0.042 with epsilon 1, uninformative because the model barely overfits (train 86.6%, test
82.8%). The network overfits more (train 97.4%, test 79.9%) and the attacker succeeds a little: 0.574 ± 0.063,
dropping to 0.498 ± 0.032 with epsilon 1. This is consistent with privacy noise removing the overfitting signal,
but it measures one simple attacker; the privacy claim rests on the accounting, not on this test.

## 4. Guarantees (see `docs/theory.md`)
- **Proven, under assumptions:** each hospital's released models are (epsilon, delta)-DP for adding or removing
  one of its patients (Poisson sampling, per-patient clipping, accountant composition, one hospital per patient,
  adversary sees all rounds). This covers the training reported in Section 3.2.
- **Proven for our simulation:** error on the drifted stage is at most error under the old rule plus the fraction
  of labels changed, if the inputs do not shift. Checked on the same patients: held in 10 of 10 seeds. Because
  this is close to an identity for our construction, it is a sanity check of the simulation, not evidence about
  real drift.
- **Measured only:** every accuracy, calibration, forgetting, recovery and detection number above.

## 5. Limitations
- Public heart data split by site, not real hospital streams. The drift, its size and its timing are chosen by us.
  Our main setting is the harshest; the gentler settings are where detection struggles.
- Small models, one tabular dataset. The network was lightly tuned only.
- Heavy missingness (Section 2). `major_vessels` and `thal` are effectively Cleveland-only. Switzerland results are
  very noisy (about 37 test patients, about 2 healthy).
- The privacy guarantee covers training only. Feature-store summaries are not private by default, and our noisy
  variant destroys accuracy unless almost no privacy is left (Section 3.3).
- Replay and EWC are evaluated without DP because they reuse old patients outside the accountant; private and
  continual together is not tested. Local calibration uses each hospital's own data but releases nothing.
- Ten seeds re-split 920 patients: intervals do not capture uncertainty from sampling new patients.
- Many comparisons were examined; we report all of them, but a few intervals excluding zero by a small margin
  (for example epsilon 1, [-2.3, -0.1]) would not survive a strict multiplicity correction.
- The membership test is weak. Drift detection is slow and unreliable for gentle drift.
- Not validated clinically. Not ready for clinical use.

## References
- McMahan et al. (2017). Communication-efficient learning of deep networks from decentralized data.
- Abadi et al. (2016). Deep learning with differential privacy.
- Mironov, Talwar, Zhang (2019). Renyi differential privacy of the sampled Gaussian mechanism.
- Balle et al. (2020). Hypothesis testing interpretations and Renyi differential privacy.
- Kirkpatrick et al. (2017). Overcoming catastrophic forgetting in neural networks.
- Gama et al. (2004). Learning with drift detection.
- Ben-David et al. (2010). A theory of learning from different domains.
- Detrano et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. UCI Heart Disease dataset.

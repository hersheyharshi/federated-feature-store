# Two-minute pitch

**Problem (20 s).** Hospitals could train better heart-disease models together, but they cannot share patient records. Even shared lessons can leak hints about individuals, and medicine changes, so models go stale.

**What we built (40 s).** Four simulated hospitals, made from the public UCI heart data, train one shared model. Only lessons travel, never records. A shared feature store makes everyone use the same definitions. Measured noise gives each patient a privacy guarantee. An alarm notices when the world changes, and replay or EWC stop the model forgetting.

**What we found (45 s).** Local models lose 15 points away from home. The shared feature store lifts federated accuracy by eight points and matches the pooled ceiling. Privacy is cheap at moderate settings and costs about three points at epsilon 0.5. After simulated drift an unprotected update forgets 33 points; replay cuts that to ten. It is not all wins: the federated model alone is worse than local at Switzerland, where 93% of patients are sick, until each hospital adds one local offset; making the shared summaries private broke the model; and our drift alarm missed gentle drift.

**Honest limits (15 s).** Public data, simulated drift, two small models. The privacy guarantee is proven under stated assumptions and covers training only; the drift results are measured, not proven.

**Hard questions to rehearse.** What does epsilon 1 mean? Why does your drift bound need the inputs not to change? Why does federated lose on Switzerland? Why did private feature-store summaries fail? Why does local calibration fix Switzerland without learning to rank those patients? What did the neural network change?

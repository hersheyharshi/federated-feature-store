# Federated Continual Feature Stores with Differential Privacy and Concept-Drift Guarantees

A research project on how hospitals and other organisations can improve AI models together without sharing patient data, while keeping those models up to date as the world changes.

**Status:** Work in progress (started 17 October 2026)

## The problem in plain words

Hospitals want to build better AI models together, but privacy laws stop them from pooling patient records. At the same time, medicine keeps changing through new diseases and new treatments, so a model that was accurate last year can become outdated. This project designs a shared feature store that lets many institutions keep their models current over time, with a mathematical privacy guarantee for individual patients and with protection against the model forgetting what it already knew.

## The four ideas this project combines

| Idea | Plain-English meaning |
|---|---|
| Federated learning | Each hospital trains on its own records and shares only what the model learned, never the records themselves. |
| Differential privacy | Carefully measured noise is added so that no single patient's presence or absence can be detected from the output. |
| Continual learning | The model keeps learning from new data over time without erasing earlier knowledge (avoiding catastrophic forgetting). |
| Concept drift | The relationship between patient information and outcomes changes over time, so the system must notice and adapt. |

The **feature store** is the shared, versioned list of model inputs (for example age or blood pressure) and how they are prepared, so every institution uses the same definitions without exchanging raw records.

## Research questions

1. How much accuracy is lost as privacy is tightened (smaller epsilon) in a federated setting?
2. How much does a model forget when the data changes over time, and how well do continual-learning methods reduce this?
3. How quickly can the system detect drift and recover?
4. What privacy and forgetting guarantees can be stated, and how well do experiments support them?

## Planned approach

- Simulate several hospitals using public multi-site data (UCI Heart Disease, with MIMIC-IV or eICU if access is approved).
- Compare four setups: local training only, federated learning, federated learning with differential privacy, and federated learning with differential privacy and forgetting prevention.
- Build a lightweight shared feature store with versioned feature definitions and privacy-protected shared statistics.
- Inject and detect concept drift, and measure accuracy, forgetting and recovery time.
- State the privacy guarantee using standard differential privacy accounting, and test the drift and forgetting bounds against experimental results.

## Repository structure

```
federated-feature-store/
  README.md          Project overview (this file)
  requirements.txt   Python packages used
  data/              Dataset notes (raw data is not stored in this repository)
  notebooks/         Colab and Jupyter notebooks, numbered in order
  src/               Reusable project code
  results/           Result tables and figures
  docs/              Plan, glossary and report drafts
```

## Getting started

1. Open a notebook from the `notebooks/` folder in Google Colab.
2. Run the install cell at the top of the notebook.
3. Follow the notebooks in numerical order.

To run locally instead:

```
pip install -r requirements.txt
```

## Tools

Python, PyTorch, Flower (federated learning), Opacus (differential privacy), River (drift detection), Avalanche (continual learning), Pandas, Matplotlib.

## Data and privacy note

No patient-level data is stored in this repository. Datasets are downloaded from their original sources under their own licences and access terms. Use of MIMIC-IV and eICU requires PhysioNet credentialing and must follow their data use agreements.

## Authors

- Harshita Pandita
- [Add your teammate's name here]

## Licence

To be decided before the project is made public.

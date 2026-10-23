# RiskIntrusion_AIBased

Risk-Based Intrusion Detection Using Explainable Data Analytics for AI-Driven Cybersecurity Decision Intelligence.

Research in progress for DBA 802. Target: Expert Systems with Applications. Short Communication eligibility is **unverified**; do not claim journal compliance or submission readiness.

## Run (Python 3.12)

```bash
python -m venv .venv
source .venv/Scripts/activate  # Git Bash on Windows
# Linux/RunPod: source .venv/bin/activate
python -m pip install -r requirements.txt
python experiment.py download
python experiment.py audit
python -m unittest -v
python experiment.py run
```

Five tracked project files: this README, experiment.py, test_experiment.py, requirements.txt, .gitignore. Data and generated results stay outside version control. All commands resolve paths relative to the script. No GPU needed. Runtime is hardware-dependent.

## Dataset provenance

Official description and academic-use conditions: https://research.unsw.edu.au/projects/unsw-nb15-dataset

Downloaded CSVs use a pinned third-party mirror (Nir-J/ML-Projects, commit 44bbd02b9e70f74dd915218d5eb702f125341ffc) and SHA-256 checks. They match a second mirror after parsing, with its partitions reversed. This is cross-mirror consistency, **not verified equivalence to university-hosted original bytes**. The download command never silently substitutes another source. Training: 175,341 rows; testing: 82,332 rows; 42 predictors. Academic use is granted by the dataset authors; commercial use requires their agreement. Dataset redistribution is excluded. Retain the five citations requested on the official dataset page in the eventual manuscript.

## Protocol

Remove id, attack_cat, and label from predictors. Remove conflicting identical-feature groups from training and deduplicate training before a stratified 60/20/20 fit/calibration/validation split. Fit preprocessing only on the fit subset. Calibrate with the independent calibration subset. Select the decision model by validation Brier score only; evaluate each prespecified baseline on test data. No test-label model selection or tuning. The test is not a verified temporal split.

Report both full test results and a subset excluding feature vectors present in the retained training pool. Test duplicates remain; they affect uncertainty and must be addressed before inference. Full-test scores can be optimistic because of overlap. Label conflicts in test remain visible in the audit.

## Decision experiment assumptions

Assign independent hypothetical asset impacts of 1, 5, or 20 using fixed probabilities and a fixed seed. These are not observed asset values or losses. Compare probability, expected-loss, impact-only, and random ranking at fixed review budgets. Loss scales are 1, 5, and 20. Reviewing a flow costs one unit and is assumed to perfectly prevent attack loss. Every policy receives exactly the same budget. Expected-loss ranking follows elementary expected utility; novelty is **not established**. Cost savings under a matching simulated utility do not establish real business impact.

## Outputs and current limits

`results/<run>/seed_<seed>/data_audit.json`: hashes, class counts, duplicates, conflicts, overlap.
`table1_detection.csv`: AP, ROC-AUC, Brier, log loss, F1.
`table2_decisions.csv`: scenario costs, captured loss, precision.
`predictions.csv` and `run_metadata.json`: reproducibility evidence.
Exactly two programmatic figures: precision-recall and scenario decision cost.

These are preliminary experiment outputs, not final manuscript tables. Statistical intervals, repeated-seed/context checks, explainability evaluation, another dataset, novelty review, verified references, and journal format audit are pending. No SHAP or local explanations have been implemented yet. No financial or field deployment claims are justified.

## Software quality audit

The pipeline rejects checksum mismatches on every load, checks partition sizes and label consistency, clones preprocessing per estimator, rejects convergence warnings, and checks feature-disjoint fitting/calibration/validation partitions. Ranking validates finite inputs and budgets; ties use seeded label-independent random ordering rather than dataset order. Metadata includes source-code and partition hashes. Runtime includes fitting, calibration, and prediction and excludes metric calculation.

Run `python -m unittest -v` for budget boundaries, corrupt-file rejection, reproducible ties, label-excluded fingerprints, and disjointness checks. These checks and a successful full-data run establish a tested research prototype, not certification by a journal. Hardware/thread differences can affect numerical results and runtime. Dependency pins cover direct dependencies; a clean-environment installation and a complete transitive lock have not been validated.

Pending scientific requirements: uncertainty accounting for duplicate flows; multiple seeds and contexts; calibrated-versus-uncalibrated ablation; explanation fidelity/stability; independent benchmark validation; novelty and citation review; verified current journal instructions. Multiplying all impacts by a constant leaves rankings unchanged and is only a cost-scaling check, not an independent robustness experiment. Reusing test outputs for iterative method design would require a new untouched evaluation set before final claims.

Style checks (optional development tool): `python -m pip install ruff==0.14.2`, then `python -m ruff check --select E,F,I experiment.py test_experiment.py` and `python -m ruff format --check experiment.py test_experiment.py`.

## Local execution (Git Bash)

Use Python 3.12. Clone your repository, then copy the five delivered project files into its root. No dataset files belong in Git.

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m unittest -v
python experiment.py download
python experiment.py run --jobs 2 --seeds 802 --output-dir results/local_v1
```

## RunPod / Linux execution

Use Python 3.12 and CPU resources. These scikit-learn models do not use CUDA, so a GPU does not accelerate this implementation. Put the checkout and outputs on a mounted persistent volume; confirm its path in your pod. The example assumes /workspace is persistent. Start with --jobs 2, then increase within your allocated CPU/RAM limits. One-hot encoded dense matrices can increase memory use. More trees or iterations are training-effort settings, not a guarantee of better performance.

```bash
cd /workspace
# Clone only if the project is not already present:
git clone https://github.com/muralibhootham85-stack/RiskIntrusion_AIBased.git
cd RiskIntrusion_AIBased
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
python experiment.py download
mkdir -p logs
nohup python -u experiment.py run --jobs 4 --trees 600 --iterations 500 --seeds 802 803 804 805 806 --output-dir results/extended_v1 > logs/extended_v1.log 2>&1 &
echo $!
tail -f logs/extended_v1.log
```

Stop viewing logs with Ctrl+C; nohup training continues. A pod restart or shutdown can terminate training. For completed-seed continuation, repeat the same run command and add --resume. It verifies configuration, input bytes and output checksums before skipping completed seeds. It does **not** resume partially trained models: if a seed was interrupted, move its incomplete seed folder aside, then resume to rerun that seed from the start. Never run concurrent writers in the same output directory. New runs default to timestamped folders; explicit nonempty seed folders are rejected.

Keep source and configuration unchanged when resuming; changed code, jobs, trees or iterations require a new output directory. Each seed changes the training split, model randomness and hypothetical impact assignment together; the current multi-seed mode cannot separate those effects. Do not select a winning seed using test results. Repeated runs do not replace external validation, uncertainty analysis, or explainability evaluation. Outputs are seed-specific; cross-seed statistical aggregation is not implemented.

## Upload code from your local checkout

```bash
git add README.md experiment.py test_experiment.py requirements.txt .gitignore
git commit -m "Add validated local and RunPod experiment pipeline"
git push origin main
```

Authenticate through your normal Git credential manager if prompted. Do not put tokens in code, commands, or the repository. Generated outputs and data remain local; archive experiment outputs separately before stopping a pod.

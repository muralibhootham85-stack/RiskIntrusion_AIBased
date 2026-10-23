"""UNSW-NB15 audit and reproducible detection/decision experiment.

Run audit before training. No labels are used to construct decision context.
"""

import argparse
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "results"
FILES = ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]
EXCLUDE = ["id", "attack_cat", "label"]
SEED = 802


MIRROR_COMMIT = "44bbd02b9e70f74dd915218d5eb702f125341ffc"
EXPECTED_SHA = {
    "UNSW_NB15_training-set.csv": (
        "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa"
    ),
    "UNSW_NB15_testing-set.csv": (
        "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559"
    ),
}


def download():
    from urllib.request import urlopen

    DATA.mkdir(exist_ok=True)
    for name in FILES:
        target = DATA / name
        if (
            target.exists()
            and hashlib.sha256(target.read_bytes()).hexdigest() == EXPECTED_SHA[name]
        ):
            continue
        url = f"https://raw.githubusercontent.com/Nir-J/ML-Projects/{MIRROR_COMMIT}/UNSW-Network_Packet_Classification/{name}"
        temporary = target.with_suffix(".download")
        try:
            with urlopen(url, timeout=60) as source, temporary.open("wb") as dest:
                while chunk := source.read(1024 * 1024):
                    dest.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != EXPECTED_SHA[name]:
                raise ValueError(f"Checksum mismatch: {name}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    print("Dataset hashes verified")


def save_json(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


def load() -> list[pd.DataFrame]:
    """Reject altered files before any labels, features, or results are used."""
    for name in FILES:
        path = DATA / name
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} missing; run: python experiment.py download"
            )
        if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_SHA[name]:
            raise ValueError(f"Checksum mismatch: {name}; training refused")
    frames = [pd.read_csv(DATA / name, encoding="utf-8-sig") for name in FILES]
    if list(frames[0].columns) != list(frames[1].columns):
        raise ValueError("Train/test schemas differ")
    for frame, expected in zip(frames, [175341, 82332]):
        if len(frame) != expected:
            raise ValueError(f"Unexpected partition size: {len(frame)} != {expected}")
        if not set(EXCLUDE).issubset(frame.columns):
            raise ValueError("Required columns missing")
        if set(frame.label.unique()) != {0, 1}:
            raise ValueError("Invalid binary labels")
        normal = frame.attack_cat.str.strip().str.lower().eq("normal")
        if not normal.eq(frame.label.eq(0)).all():
            raise ValueError("Attack category and binary label disagree")
    return frames


def feature_hash(frame: pd.DataFrame) -> pd.Series:
    """Fingerprint predictors only; identifiers and targets never affect grouping."""
    return pd.util.hash_pandas_object(frame.drop(columns=EXCLUDE), index=False)


def audit():
    train, test = load()
    hashes = [feature_hash(frame) for frame in (train, test)]
    report = {
        "source": "UNSW-NB15 public benchmark; third-party GitHub mirror",
        "mirror_repository": "Nir-J/ML-Projects",
        "mirror_commit": MIRROR_COMMIT,
        "official_byte_equivalence_verified": False,
        "excluded_predictors": EXCLUDE,
        "predictor_count": len(train.columns) - len(EXCLUDE),
        "test_rows_with_train_feature_match": int(hashes[1].isin(hashes[0]).sum()),
        "files": {},
    }
    for name, frame, hashed in zip(FILES, (train, test), hashes):
        groups = (
            pd.DataFrame({"hash": hashed, "label": frame.label})
            .groupby("hash")
            .label.nunique()
        )
        numeric = frame.select_dtypes(include="number")
        report["files"][name] = {
            "sha256": hashlib.sha256((DATA / name).read_bytes()).hexdigest(),
            "rows": len(frame),
            "columns": len(frame.columns),
            "feature_duplicate_rows": int(hashed.duplicated().sum()),
            "conflicting_feature_groups": int(groups.gt(1).sum()),
            "missing_cells": int(frame.isna().sum().sum()),
            "infinite_numeric_cells": int(np.isinf(numeric.to_numpy()).sum()),
            "binary_counts": frame.label.value_counts().to_dict(),
            "category_counts": frame.attack_cat.value_counts().to_dict(),
        }
    save_json("data_audit.json", report)
    print(json.dumps(report, indent=2))


def selected_mask(score: np.ndarray, budget: float, seed: int = SEED) -> np.ndarray:
    """Select exactly floor(budget*n) rows with label-independent random tie breaks."""
    values = np.asarray(score, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Scores must be a finite one-dimensional array")
    if not np.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError("Budget must be finite and between zero and one")
    chosen = np.zeros(len(values), dtype=bool)
    tie_break = np.random.default_rng(seed).random(len(values))
    order = np.lexsort((tie_break, -values))
    chosen[order[: int(np.floor(budget * len(values)))]] = True
    return chosen


def split_training(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create three reproducible stratified, feature-disjoint training partitions."""
    idx = np.arange(len(train))
    fit, held = train_test_split(
        idx, test_size=0.4, stratify=train.label, random_state=SEED
    )
    cal, val = train_test_split(
        held, test_size=0.5, stratify=train.label.iloc[held], random_state=SEED
    )
    fingerprints = feature_hash(train)
    groups = [set(fingerprints.iloc[part]) for part in (fit, cal, val)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Feature overlap between fit/calibration/validation")
    return fit, cal, val


def run(trees: int = 200, iterations: int = 200, jobs: int = 2):
    audit()
    train, test = load()
    hashes = feature_hash(train)
    # Remove ambiguous identical feature vectors, then deduplicate before splitting.
    conflicts = pd.DataFrame({"h": hashes, "y": train.label}).groupby("h").y.nunique()
    keep = ~hashes.isin(conflicts[conflicts.gt(1)].index) & ~hashes.duplicated()
    train = train.loc[keep].reset_index(drop=True)
    fit_idx, cal_idx, val_idx = split_training(train)
    x, y = train.drop(columns=EXCLUDE), train.label
    xt, yt = test.drop(columns=EXCLUDE), test.label.to_numpy()
    cats = list(x.select_dtypes(include="object").columns)
    nums = [col for col in x if col not in cats]
    prep = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                nums,
            ),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                cats,
            ),
        ]
    )
    models = {
        "Logistic regression": LogisticRegression(max_iter=1500, random_state=SEED),
        "Random forest": RandomForestClassifier(
            n_estimators=trees, min_samples_leaf=3, n_jobs=jobs, random_state=SEED
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=iterations, random_state=SEED
        ),
    }
    fitted, validation, rows = {}, {}, []
    novel = ~feature_hash(test).isin(feature_hash(train)).to_numpy()
    for name, estimator in models.items():
        started = time.perf_counter()
        base = make_pipeline(clone(prep), clone(estimator))
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            base.fit(x.iloc[fit_idx], y.iloc[fit_idx])
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
        calibrated.fit(x.iloc[cal_idx], y.iloc[cal_idx])
        pv = calibrated.predict_proba(x.iloc[val_idx])[:, 1]
        validation[name] = float(brier_score_loss(y.iloc[val_idx], pv))
        p = calibrated.predict_proba(xt)[:, 1]
        elapsed = time.perf_counter() - started
        fitted[name] = p
        for subset, mask in [
            ("official", np.ones(len(test), dtype=bool)),
            ("nonoverlap", novel),
        ]:
            if len(np.unique(yt[mask])) != 2:
                raise ValueError("Evaluation subset must contain both classes")
            truth, prob = yt[mask], p[mask]
            rows.append(
                dict(
                    model=name,
                    subset=subset,
                    n=int(mask.sum()),
                    average_precision=average_precision_score(truth, prob),
                    roc_auc=roc_auc_score(truth, prob),
                    brier=brier_score_loss(truth, prob),
                    log_loss=log_loss(truth, prob),
                    f1=f1_score(truth, prob >= 0.5),
                    fit_and_predict_seconds=elapsed,
                )
            )
        print(name, "validation Brier:", validation[name], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "table1_detection.csv", index=False)
    best = min(validation, key=validation.get)
    p = fitted[best]
    # Hypothetical asset losses independently assigned without labels or attack type.
    impact = np.random.default_rng(SEED).choice(
        [1.0, 5.0, 20.0], len(test), p=[0.6, 0.3, 0.1]
    )
    decision = []
    for subset, mask in [
        ("official", np.ones(len(test), dtype=bool)),
        ("nonoverlap", novel),
    ]:
        truth, prob, assets = yt[mask], p[mask], impact[mask]
        for multiplier in [1.0, 5.0, 20.0]:
            loss = multiplier * assets
            for budget in [0.01, 0.05, 0.1, 0.2]:
                for policy, score in [
                    ("probability", prob),
                    ("expected_loss", prob * loss),
                    ("impact_only", loss),
                    ("random", np.random.default_rng(SEED).random(len(prob))),
                ]:
                    chosen = selected_mask(score, budget, seed=SEED)
                    realized = truth * loss * ~chosen + chosen.astype(float)
                    decision.append(
                        dict(
                            model=best,
                            subset=subset,
                            loss_multiplier=multiplier,
                            budget=budget,
                            policy=policy,
                            reviewed=int(chosen.sum()),
                            mean_scenario_cost=float(realized.mean()),
                            captured_loss_fraction=float(
                                (truth * loss * chosen).sum() / (truth * loss).sum()
                            ),
                            precision=float(truth[chosen].mean()),
                        )
                    )
    pd.DataFrame(decision).to_csv(OUT / "table2_decisions.csv", index=False)
    pd.DataFrame(
        {"label": yt, "impact_scenario": impact, "nonoverlap": novel, **fitted}
    ).to_csv(OUT / "predictions.csv", index=False)
    save_json(
        "run_metadata.json",
        dict(
            seed=SEED,
            trees=trees,
            iterations=iterations,
            jobs=jobs,
            python=platform.python_version(),
            sklearn=sklearn.__version__,
            numpy=np.__version__,
            pandas=pd.__version__,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            training_partition_sha256={
                key: hashlib.sha256(part.astype("<i8").tobytes()).hexdigest()
                for key, part in [
                    ("fit", fit_idx),
                    ("calibration", cal_idx),
                    ("validation", val_idx),
                ]
            },
            tie_policy="seeded label-independent random ties",
            fit_n=len(fit_idx),
            calibration_n=len(cal_idx),
            validation_n=len(val_idx),
            validation_brier=validation,
            selected_model=best,
            limitations=[
                "One benchmark; no field validation",
                "Impact assignments and losses are hypothetical; "
                "they are not observed financial data",
                "Review assumes perfect prevention and constant unit cost",
                "No uncertainty analysis yet; outputs are preliminary",
            ],
        ),
    )
    plot(fitted, yt, pd.DataFrame(decision))


def plot(predictions, truth, decisions):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    for name, prob in predictions.items():
        precision, recall, _ = precision_recall_curve(truth, prob)
        ax.plot(recall, precision, label=name)
    ax.axhline(truth.mean(), color="gray", linestyle="--", label="Attack prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(OUT / "figure1_precision_recall.png", dpi=400)
    plt.close(fig)
    fig, axes = plt.subplots(
        1, 3, figsize=(10, 3.5), constrained_layout=True, sharey=False
    )
    for ax, multiplier in zip(axes, [1.0, 5.0, 20.0]):
        data = decisions[
            (decisions.subset == "nonoverlap")
            & (decisions.loss_multiplier == multiplier)
        ]
        for name, group in data.groupby("policy"):
            ax.plot(
                group.budget * 100, group.mean_scenario_cost, marker="o", label=name
            )
        ax.set(
            title=f"Loss multiplier {multiplier:g}",
            xlabel="Review budget (%)",
            ylabel="Scenario cost / flow",
        )
    axes[-1].legend(fontsize=7)
    fig.savefig(OUT / "figure2_decision_cost.png", dpi=400)
    plt.close(fig)


def positive_int(value: str) -> int:
    """CLI resource and training limits must be positive integers."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def main() -> None:
    """Execute isolated runs; resume only matching, verified completed seed runs."""
    global DATA, OUT, SEED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "audit", "run"])
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[802])
    parser.add_argument("--jobs", type=positive_int, default=2)
    parser.add_argument("--trees", type=positive_int, default=200)
    parser.add_argument("--iterations", type=positive_int, default=200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(
        s < 0 or s >= 2**32 for s in args.seeds
    ):
        parser.error("Seeds must be distinct integers in [0, 2**32)")
    DATA = args.data_dir.expanduser().resolve()
    if args.command == "download":
        download()
        return
    if args.command == "audit":
        OUT = (args.output_dir or ROOT / "results").expanduser().resolve()
        audit()
        return
    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = (args.output_dir or ROOT / "results" / stamp).expanduser().resolve()
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for seed in args.seeds:
        SEED, OUT = seed, destination / f"seed_{seed}"
        config = dict(
            seed=seed,
            trees=args.trees,
            iterations=args.iterations,
            jobs=args.jobs,
            source_sha256=source_sha,
            dataset_sha256=EXPECTED_SHA,
        )
        marker = OUT / "complete.json"
        if OUT.exists() and any(OUT.iterdir()):
            if not args.resume or not marker.is_file():
                parser.error(
                    f"Nonempty or incomplete output: {OUT}; choose a new directory"
                )
            completed = json.loads(marker.read_text())
            if completed["config"] != config:
                parser.error(f"Resume configuration mismatch: {OUT}")
            for name, digest in completed["outputs"].items():
                path = OUT / name
                if (
                    not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != digest
                ):
                    parser.error(f"Resume output missing or changed: {path}")
            load()  # Verify current input bytes even when skipping a completed run.
            print(f"Verified completed seed {seed}; skipping", flush=True)
            continue
        OUT.mkdir(parents=True, exist_ok=True)
        print(f"Starting seed {seed}: {OUT}", flush=True)
        with threadpool_limits(limits=args.jobs):
            run(trees=args.trees, iterations=args.iterations, jobs=args.jobs)
        outputs = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in OUT.iterdir()
            if p.is_file()
        }
        save_json("complete.json", {"config": config, "outputs": outputs})
        print(f"Completed seed {seed}", flush=True)


if __name__ == "__main__":
    main()

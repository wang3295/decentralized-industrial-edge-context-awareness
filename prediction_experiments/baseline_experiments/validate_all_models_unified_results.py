from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RESULT_DIR = Path(__file__).resolve().parent / "results" / "all_models_unified_full_v1"
DATASETS = ["Alibaba", "GWA-T-12", "Google"]
CORE_METRICS = ["RMSE", "MAE", "R2", "MAPE_pct", "sMAPE_pct"]


def main() -> None:
    raw = pd.read_csv(RESULT_DIR / "prediction_metrics_raw.csv")
    summary = pd.read_csv(RESULT_DIR / "prediction_summary_mean_std.csv")

    assert len(raw) == 111
    assert len(summary) == 27
    assert not raw[CORE_METRICS].isna().any().any()
    assert not raw.duplicated(["dataset", "model", "run_id"]).any()

    prediction_files = 0
    for dataset in DATASETS:
        dataset_dir = RESULT_DIR / dataset
        canonical = np.load(dataset_dir / "canonical_test_targets.npz")["y_true"]
        files = list(dataset_dir.glob("*predictions.npz"))
        assert len(files) == 37, (dataset, len(files))
        prediction_files += len(files)
        for file_path in files:
            saved = np.load(file_path)
            assert np.array_equal(saved["y_true"], canonical), file_path
            assert saved["y_pred"].shape == canonical.shape, file_path
            assert np.isfinite(saved["y_pred"]).all(), file_path

    for _, row in summary.iterrows():
        group = raw[(raw.dataset == row.dataset) & (raw.model == row.model)]
        assert len(group) == int(row.runs)
        for metric in CORE_METRICS:
            expected_mean = group[metric].mean()
            expected_sd = group[metric].std(ddof=1) if len(group) > 1 else 0.0
            assert np.isclose(expected_mean, row[f"{metric}_mean"], rtol=0, atol=1e-12)
            assert np.isclose(expected_sd, row[f"{metric}_std"], rtol=0, atol=1e-12)

    report = {
        "status": "passed",
        "raw_rows": len(raw),
        "summary_rows": len(summary),
        "prediction_files": prediction_files,
        "same_test_targets_for_all_models": True,
        "summary_recomputed_exactly": True,
        "standard_deviation": "sample SD (ddof=1); deterministic methods use 0",
        "no_missing_core_metrics": True,
        "no_duplicate_run_ids": True,
        "finite_predictions": True,
    }
    (RESULT_DIR / "qa_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

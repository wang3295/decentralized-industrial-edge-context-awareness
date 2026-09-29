from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DATASET_SLUGS = {
    "Alibaba": "alibaba",
    "GWA-T-12": "gwa_t_12",
    "Google": "google",
}
LEARNED_MODELS = (
    "fits",
    "fits_delta",
    "gru",
    "dlinear",
    "s2s_lstm",
    "frets",
    "patchtst",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def config_from_payload(unified, payload: dict):
    values = dict(payload["config"])
    values["seeds"] = tuple(values["seeds"])
    values["arima_p_candidates"] = tuple(values["arima_p_candidates"])
    values["arima_d_candidates"] = tuple(values["arima_d_candidates"])
    return unified.UnifiedConfig(**values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=HERE / "results" / "all_models_unified_full_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results" / "industrial_three_dataset_package_v1",
    )
    parser.add_argument("--windows", type=int, default=257)
    args = parser.parse_args()

    benchmark_dir = args.benchmark_dir.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    unified = load_module("industrial_bundle_unified", HERE / "run_all_models_unified.py")
    base = unified.BASE
    payload = json.loads((benchmark_dir / "config.json").read_text(encoding="utf-8"))
    cfg = config_from_payload(unified, payload)
    manifest = {
        "status": "complete",
        "seed": 2026,
        "windows_per_dataset": args.windows,
        "selection": "evenly spaced across the held-out chronological test windows",
        "datasets": {},
    }

    for dataset_name, slug in DATASET_SLUGS.items():
        source_dir = benchmark_dir / dataset_name
        target_dir = output_dir / slug
        checkpoint_dir = target_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        data, metadata = base.load_dataset(base.DATASETS[dataset_name], unified.base_config(cfg))
        splits = base.split_series(data, unified.base_config(cfg))
        x_test, y_test = base.make_windows(splits["test"], cfg.seq_len, cfg.pred_len)
        indices = np.unique(
            np.linspace(0, len(x_test) - 1, num=min(args.windows, len(x_test)), dtype=int)
        )
        input_path = target_dir / "heldout_input_windows.npz"
        np.savez_compressed(
            input_path,
            dataset=np.asarray(dataset_name),
            x_raw=x_test[indices].astype(np.float32),
            y_true=y_test[indices].astype(np.float32),
            source_window_index=indices.astype(np.int64),
            pred_len=np.asarray(cfg.pred_len, dtype=np.int64),
        )

        copied = []
        for model_key in LEARNED_MODELS:
            source = source_dir / f"{model_key}_seed_2026.pt"
            target = checkpoint_dir / source.name
            shutil.copy2(source, target)
            copied.append(target)
        arima_source = source_dir / "arima_model.json"
        arima_target = checkpoint_dir / arima_source.name
        shutil.copy2(arima_source, arima_target)
        copied.append(arima_target)

        manifest["datasets"][dataset_name] = {
            "slug": slug,
            "source_file": str(base.DATASETS[dataset_name].path.resolve()),
            "processed_rows": int(len(data)),
            "processed_sha256": metadata["processed_sha256"],
            "test_windows_total": int(len(x_test)),
            "selected_windows": int(len(indices)),
            "selected_first_index": int(indices[0]),
            "selected_last_index": int(indices[-1]),
            "input_file": str(input_path.relative_to(output_dir)),
            "input_sha256": sha256_file(input_path),
            "checkpoints": {
                item.name: {"bytes": item.stat().st_size, "sha256": sha256_file(item)}
                for item in copied
            },
        }

    for filename in (
        "benchmark_models_on_vm.py",
        "run_all_models_unified.py",
        "run_baselines.py",
    ):
        shutil.copy2(HERE / filename, output_dir / filename)
    (output_dir / "bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

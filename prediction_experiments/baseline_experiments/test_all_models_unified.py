import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parent / "run_all_models_unified.py"
spec = importlib.util.spec_from_file_location("run_all_models_unified", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


EXPECTED_PARAMETERS = {
    "fits": 625,
    "fits_delta": 625,
    "gru": 3786,
    "dlinear": 1010,
    "s2s_lstm": 208066,
    "frets": 1229,
    "patchtst": 69129,
}


def test_all_learning_models_have_expected_shape_and_parameter_count():
    cfg = module.UnifiedConfig()
    sample = torch.randn(3, cfg.seq_len, 2)
    for key, expected_parameters in EXPECTED_PARAMETERS.items():
        model = module.make_model(key, cfg, channels=2).eval()
        with torch.inference_mode():
            prediction = model(sample)
        assert prediction.shape == (3, cfg.pred_len, 2), key
        assert module.count_parameters(model) == expected_parameters, key


def test_dlinear_prediction_is_independent_of_other_batch_members():
    cfg = module.UnifiedConfig()
    model = module.make_model("dlinear", cfg, channels=2).eval()
    first = torch.randn(1, cfg.seq_len, 2)
    second = torch.randn(1, cfg.seq_len, 2) * 100
    with torch.inference_mode():
        alone = model(first)
        batched = model(torch.cat([first, second], dim=0))[:1]
    torch.testing.assert_close(alone, batched)


def test_model_source_and_variant_are_declared():
    for label in module.MODEL_LABELS.values():
        assert label in module.MODEL_SOURCE
        assert label in module.IMPLEMENTATION_VARIANT


if __name__ == "__main__":
    test_all_learning_models_have_expected_shape_and_parameter_count()
    test_dlinear_prediction_is_independent_of_other_batch_members()
    test_model_source_and_variant_are_declared()
    print("All unified-model tests passed.")

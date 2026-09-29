import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parent / "run_baselines.py"
spec = importlib.util.spec_from_file_location("run_baselines", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_causal_moving_average_does_not_use_future():
    data = np.array([[1.0], [2.0], [100.0]], dtype=np.float32)
    result = module.causal_moving_average(data, 2).ravel()
    np.testing.assert_allclose(result, [1.0, 1.5, 51.0])


def test_windows_are_chronological():
    data = np.arange(12, dtype=np.float32).reshape(6, 2)
    x, y = module.make_windows(data, seq_len=2, pred_len=1)
    assert x.shape == (4, 2, 2)
    assert y.shape == (4, 1, 2)
    np.testing.assert_array_equal(x[0], data[:2])
    np.testing.assert_array_equal(y[0], data[2:3])
    np.testing.assert_array_equal(x[-1], data[3:5])
    np.testing.assert_array_equal(y[-1], data[5:6])


def test_naive_repeats_last_observation():
    x = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    pred = module.naive_predict(x, 3)
    expected = np.array([[[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]], dtype=np.float32)
    np.testing.assert_array_equal(pred, expected)


def test_arima_random_walk_forecast():
    x = np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32)
    # ARIMA(1,1,0): next difference = 1 * previous difference.
    pred = module.predict_arima_p_d_0(x, [(1, 1)], [(0.0, np.array([1.0]))], 3)
    np.testing.assert_allclose(pred.ravel(), [5.0, 6.0, 7.0])


def test_fits_matches_main_experiment_shape_and_parameter_count():
    model = module.FITSModel(seq_len=100, pred_len=5, channels=2, cut_freq=24)
    output = model(torch.randn(3, 100, 2))
    assert output.shape == (3, 5, 2)
    assert sum(parameter.numel() for parameter in model.parameters()) == 625


def test_individual_fits_shape_and_parameter_count():
    model = module.FITSModel(seq_len=100, pred_len=5, channels=2, cut_freq=24, individual=True)
    output = model(torch.randn(3, 100, 2))
    assert output.shape == (3, 5, 2)
    assert sum(parameter.numel() for parameter in model.parameters()) == 1250


if __name__ == "__main__":
    test_causal_moving_average_does_not_use_future()
    test_windows_are_chronological()
    test_naive_repeats_last_observation()
    test_arima_random_walk_forecast()
    test_fits_matches_main_experiment_shape_and_parameter_count()
    test_individual_fits_shape_and_parameter_count()
    print("All protocol tests passed.")

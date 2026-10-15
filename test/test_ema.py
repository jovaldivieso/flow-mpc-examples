from __future__ import annotations

import os
import sys
import unittest

import torch
from torch import nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.models.ema import DEFAULT_EMA_DECAY, ExponentialMovingAverage


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))


def _set_all(module: _Tiny, value: float) -> None:
    with torch.no_grad():
        module.linear.weight.fill_(value)
        module.linear.bias.fill_(value)


class ExponentialMovingAverageTests(unittest.TestCase):
    def test_default_decay_is_the_original_papers(self) -> None:
        self.assertEqual(DEFAULT_EMA_DECAY, 0.995)
        self.assertEqual(ExponentialMovingAverage(_Tiny()).decay, 0.995)

    def test_invalid_decay_is_rejected(self) -> None:
        for decay in (-0.1, 1.0, 2.0):
            with self.subTest(decay=decay):
                with self.assertRaises(ValueError):
                    ExponentialMovingAverage(_Tiny(), decay=decay)

    def test_the_average_starts_at_the_weights_and_is_a_copy(self) -> None:
        module = _Tiny()
        _set_all(module, 1.0)
        ema = ExponentialMovingAverage(module)
        torch.testing.assert_close(ema.shadow["linear.weight"], torch.ones(2, 2))

        _set_all(module, 7.0)  # changing the weights afterwards must not change the average
        torch.testing.assert_close(ema.shadow["linear.weight"], torch.ones(2, 2))

    def test_update_is_decay_times_average_plus_one_minus_decay_times_weights(self) -> None:
        module = _Tiny()
        _set_all(module, 0.0)
        ema = ExponentialMovingAverage(module, decay=0.9)

        _set_all(module, 1.0)
        ema.update(module)
        torch.testing.assert_close(ema.shadow["linear.weight"], torch.full((2, 2), 0.1))

        ema.update(module)
        torch.testing.assert_close(ema.shadow["linear.weight"], torch.full((2, 2), 0.9 * 0.1 + 0.1))

    def test_integer_buffers_are_copied_not_averaged(self) -> None:
        module = _Tiny()
        ema = ExponentialMovingAverage(module, decay=0.9)
        module.steps.fill_(5)
        ema.update(module)
        self.assertEqual(int(ema.shadow["steps"]), 5)

    def test_averaged_weights_context_runs_with_the_average_and_restores_the_weights(self) -> None:
        module = _Tiny()
        _set_all(module, 0.0)
        ema = ExponentialMovingAverage(module, decay=0.9)
        _set_all(module, 1.0)
        ema.update(module)

        with ema.averaged_weights(module):
            torch.testing.assert_close(module.linear.weight, torch.full((2, 2), 0.1))
        torch.testing.assert_close(module.linear.weight, torch.ones(2, 2))

    def test_the_weights_are_restored_even_when_the_block_raises(self) -> None:
        module = _Tiny()
        ema = ExponentialMovingAverage(module)
        _set_all(module, 3.0)

        with self.assertRaises(RuntimeError):
            with ema.averaged_weights(module):
                raise RuntimeError("boom")

        torch.testing.assert_close(module.linear.weight, torch.full((2, 2), 3.0))

    def test_state_dict_round_trips_and_rejects_a_foreign_module(self) -> None:
        module = _Tiny()
        ema = ExponentialMovingAverage(module, decay=0.5)
        _set_all(module, 2.0)
        ema.update(module)
        restored = ExponentialMovingAverage(_Tiny(), decay=0.5)

        restored.load_state_dict(ema.state_dict())

        for name, value in ema.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value)
        with self.assertRaises(ValueError):
            restored.load_state_dict({"other": torch.zeros(1)})

    def test_the_returned_state_dict_does_not_alias_the_average(self) -> None:
        ema = ExponentialMovingAverage(_Tiny())
        ema.state_dict()["linear.weight"].fill_(9.0)
        self.assertFalse(bool((ema.shadow["linear.weight"] == 9.0).all()))


if __name__ == "__main__":
    unittest.main()

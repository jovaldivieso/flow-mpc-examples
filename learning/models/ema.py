from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping

import torch
from torch import nn

DEFAULT_EMA_DECAY = 0.995


class ExponentialMovingAverage:
    """Exponential moving average of a module's parameters: ema <- decay * ema + (1 - decay) * parameter.

    The averaged weights are less noisy than the weights of the last optimizer step, and are what the policy
    is run with (SafeFlowMPC trains and evaluates its flow network this way, decay 0.995). Keeps plain tensors
    instead of a second module so it also works for policies that wrap non-copyable parts (the safe flow policy
    carries a CasADi projector): ``averaged_weights`` temporarily loads the average into the live policy.
    """

    def __init__(self, module: nn.Module, decay: float = DEFAULT_EMA_DECAY) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"'decay' must be in [0, 1), got {decay}.")
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {
            name: tensor.detach().clone() for name, tensor in module.state_dict().items()
        }

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for name, tensor in module.state_dict().items():
            shadow = self.shadow[name]
            if shadow.is_floating_point():
                shadow.mul_(self.decay).add_(tensor.detach().to(shadow.device), alpha=1.0 - self.decay)
            else:
                shadow.copy_(tensor.detach().to(shadow.device))

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        if set(state_dict) != set(self.shadow):
            raise ValueError("The EMA state dict does not match the module it was built for.")
        self.shadow = {name: tensor.detach().clone() for name, tensor in state_dict.items()}

    @contextlib.contextmanager
    def averaged_weights(self, module: nn.Module) -> Iterator[None]:
        """Run the module with the averaged weights, then put its own weights back."""
        own = {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}
        module.load_state_dict(self.shadow)
        try:
            yield
        finally:
            module.load_state_dict(own)

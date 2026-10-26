from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from learning.models.temporal_unet import TemporalUnet

TIME_BETA_ALPHA = 1.5
TIME_BETA_BETA = 1.0


class FlowVelocityNet(nn.Module):
    """Conditional flow-matching velocity-prediction network: given a noisy chunk of actions, a conditioning vector,
    and a flow time in [0, 1], predicts the denoising velocity -- trained via rectified flow matching
    (``compute_loss``) and sampled via Euler integration (``integrate``).

    The network is SafeFlowMPC's temporal U-Net (``TemporalUnet``), conditioned on the flat ``obs_cond`` vector (here the
    goal and the state) and on the flow time passed as it is, in [0, 1]. The horizon of the chunk has to be divisible by
    ``2 ** (len(unet_dim_mults) - 1)``.
    """

    def __init__(
        self,
        action_dim: int,
        obs_cond_dim: int,
        prediction_horizon: int = 16,
        unet_dim: int = 32,
        unet_dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        num_inference_steps: int = 10,
        action_scale: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        if action_dim <= 0:
            raise ValueError(f"'action_dim' must be positive, got {action_dim}.")
        if obs_cond_dim <= 0:
            raise ValueError(f"'obs_cond_dim' must be positive, got {obs_cond_dim}.")
        if prediction_horizon <= 0:
            raise ValueError(f"'prediction_horizon' must be positive, got {prediction_horizon}.")
        if unet_dim <= 0 or unet_dim % 8 != 0:
            raise ValueError(f"'unet_dim' must be a positive multiple of 8 (the group norm has 8 groups), got {unet_dim}.")
        if len(unet_dim_mults) < 1 or any(m <= 0 for m in unet_dim_mults):
            raise ValueError(f"'unet_dim_mults' must be a non-empty tuple of positive integers, got {unet_dim_mults}.")
        if num_inference_steps <= 0:
            raise ValueError(f"'num_inference_steps' must be positive, got {num_inference_steps}.")

        self.action_dim = int(action_dim)
        self.obs_cond_dim = int(obs_cond_dim)
        self.prediction_horizon = int(prediction_horizon)
        self.num_inference_steps = int(num_inference_steps)

        # Rectified flow interpolates x_t = t*x_1 + (1-t)*x_0 against x_0 ~
        # N(0, 1) (see compute_loss/integrate below) -- an isotropic,
        # unit-scale noise prior. Per-dimension action bounds are not
        # generally isotropic (e.g. joints configured with different
        # acceleration limits, since nothing physically ties them
        # together), so training x_1 = raw,
        # unnormalized actions against that fixed unit-scale prior makes the
        # loss -- and the gradient -- dominated by whichever action
        # dimension happens to have the largest physical range, independent
        # of how much that dimension actually matters for the task. Dividing
        # by a fixed, config-derived per-dimension scale before ever touching
        # x_0 puts every action dimension on comparable footing relative to
        # that same prior, and integrate() multiplies back by the same scale
        # to return actions in physical units. Defaults to all-ones (no-op)
        # only so this remains constructible without one (e.g. in isolated
        # unit tests) -- real training/eval callers must pass the actual
        # max_action so the network is trained and queried in the
        # same normalized space.
        if action_scale is None:
            action_scale_tensor = torch.ones(self.action_dim, dtype=torch.float32)
        else:
            action_scale_tensor = torch.as_tensor(action_scale, dtype=torch.float32).reshape(-1)
            if action_scale_tensor.shape != (self.action_dim,):
                raise ValueError(
                    f"'action_scale' must have shape ({self.action_dim},), got "
                    f"{tuple(action_scale_tensor.shape)}."
                )
            if bool((~torch.isfinite(action_scale_tensor)).any() or (action_scale_tensor <= 0).any()):
                raise ValueError("'action_scale' entries must all be finite and positive.")
        # Non-persistent: this is config-derived (the fleet's own max_action),
        # not learned, and every current caller already reconstructs it from
        # scratch at both train and eval time via checkpoint metadata rather
        # than through load_state_dict -- keeping it out of state_dict()
        # avoids a checkpoint saved before this option existed suddenly
        # gaining an unexpected key.
        self.register_buffer("action_scale", action_scale_tensor, persistent=False)

        self.unet = TemporalUnet(
            horizon=self.prediction_horizon,
            transition_dim=self.action_dim,
            cond_dim=self.obs_cond_dim,
            dim=unet_dim,
            dim_mults=tuple(int(m) for m in unet_dim_mults),
        )

    def predict_velocity(
        self,
        action_state_flat: torch.Tensor,
        obs_cond: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        chunk = action_state_flat.view(-1, self.prediction_horizon, self.action_dim)
        return self.unet(chunk, timesteps.float(), obs_cond).flatten(1)

    def compute_loss(self, obs_cond: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch_size = actions.shape[0]
        x_1 = actions / self.action_scale
        x_0 = torch.randn_like(actions)
        # t = 1 - Beta(1.5, 1): the density rises toward t = 0, so the early, noisy part of the flow gets more of the
        # training steps than the end of it, where the target is almost determined (SafeFlowMPC's training scripts)
        # Sampled on the CPU: the Dirichlet sampler behind Beta is not implemented on the MPS backend
        beta_distribution = torch.distributions.Beta(torch.tensor(TIME_BETA_ALPHA), torch.tensor(TIME_BETA_BETA))
        t = 1.0 - beta_distribution.sample((batch_size,)).to(device=actions.device, dtype=actions.dtype)
        t_expanded = t.view(batch_size, 1, 1)
        x_t = t_expanded * x_1 + (1.0 - t_expanded) * x_0
        target_velocity = x_1 - x_0
        pred_velocity = self.predict_velocity(x_t.flatten(1), obs_cond, t)
        return F.mse_loss(pred_velocity, target_velocity.flatten(1))

    @torch.no_grad()
    def integrate(self, obs_cond: torch.Tensor) -> torch.Tensor:
        batch_size = obs_cond.shape[0]
        device = obs_cond.device

        x = torch.randn(
            batch_size, self.prediction_horizon, self.action_dim, device=device, dtype=obs_cond.dtype
        )

        dt = 1.0 / float(self.num_inference_steps)

        for step in range(self.num_inference_steps):
            t_val = step * dt
            t_tensor = torch.full((batch_size,), t_val, device=device, dtype=obs_cond.dtype)
            pred_velocity = self.predict_velocity(x.flatten(1), obs_cond, t_tensor)
            pred_velocity = pred_velocity.view(batch_size, self.prediction_horizon, self.action_dim)
            x.add_(pred_velocity, alpha=dt)

        # x was integrated entirely in the normalized space compute_loss trains
        # in -- rescale back to physical action units before returning, since
        # every caller (apply_execution_noise, simulator.step, ...) expects those.
        return x * self.action_scale

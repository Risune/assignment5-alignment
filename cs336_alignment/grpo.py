import torch

from collections.abc import Callable
from typing import Literal



def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool):

  batch_size = len(rollout_responses)
  advantages = torch.empty((batch_size,))
  raw_rewards = torch.Tensor([reward_fn(resp, gt)["reward"] for resp, gt in zip(rollout_responses, repeated_ground_truths)])

  for i in range(0, batch_size, group_size):
    group_rewards = raw_rewards[i:i+group_size]
    mean = torch.mean(group_rewards)
    advantages[i:i+group_size] = group_rewards - mean
    if normalize_by_std:
      std = torch.std(group_rewards)
      advantages[i:i+group_size] /= (std + advantage_eps)

  return advantages, raw_rewards, {}


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
  ) -> torch.Tensor:
  return -policy_log_probs * raw_rewards_or_advantages


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  ratio = torch.exp(policy_log_probs - old_log_probs)
  clip = torch.clip(ratio, 1-cliprange, 1+cliprange)
  return -torch.min(ratio * advantages, clip * advantages), {}


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  if loss_type == "no_baseline":
    return -raw_rewards * policy_log_probs, {}
  elif loss_type == "reinforce_with_baseline":
    return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
  else:
    return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None= None,
  ) -> torch.Tensor:
  count = (mask > 0).sum(dim=dim, keepdim=True)
  mean = tensor.where(mask, 0).sum(dim=dim, keepdim=True) / count
  if dim is not None:
    return mean.squeeze(dim=dim)
  else:
    return mean.view(())


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  loss, metadata = compute_policy_gradient_loss(
    policy_log_probs,
    loss_type,
    raw_rewards,
    advantages,
    old_log_probs,
    cliprange
  )
  mean = masked_mean(loss, response_mask) / gradient_accumulation_steps
  mean.backward()
  return mean, metadata


if __name__ == "__main__":
  a = torch.Tensor([1])
  print(a)
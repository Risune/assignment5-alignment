import torch
import einops
from cs336_alignment import helper

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  B = policy_log_probs.shape[0]
  # loss本质上是mean，而不是sum，所以要除以batch_size
  loss = helper.masked_normalize(-policy_log_probs, response_mask, normalize_constant * B) / gradient_accumulation_steps
  loss.backward()
  return loss, {}


def sft_train():
  ...


if __name__ == "__main__":
  sft_train()

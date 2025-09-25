import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
import einops


def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer) -> dict[str, torch.Tensor]:
  prompt_ids = [tokenizer(s, return_tensors="pt", return_attention_mask=False)["input_ids"] for s in prompt_strs]
  output_ids = [tokenizer(s, return_tensors="pt", return_attention_mask=False)["input_ids"] for s in output_strs]
  B = len(prompt_ids)
  T = max(prompt_ids[i].shape[-1] + output_ids[i].shape[-1] for i in range(B))
  tokenized = torch.full((B, T), tokenizer.eos_token_id)
  masks = torch.full((B, T-1), False)
  for i in range(B):
    mask_start, mask_end = prompt_ids[i].shape[-1] - 1, output_ids[i].shape[-1] - 1
    concat = torch.cat((prompt_ids[i], output_ids[i]),dim=-1)
    l = concat.shape[-1]
    mask_start, mask_end = prompt_ids[i].shape[-1] - 1, l - 1
    tokenized[i, 0:l] = concat
    masks[i, mask_start:mask_end] = True
  return {
    "input_ids": tokenized[:, :-1],
    "labels": tokenized[:, 1:],
    "response_mask": masks
  }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
  max = einops.reduce(logits, "... d -> ... 1", "max")
  scaled = logits - max
  exp = torch.exp(scaled)
  sum = einops.reduce(exp, "... d -> ... 1", "sum")
  return -einops.reduce(exp / sum * (scaled - torch.log(sum)), "... d -> ...", "sum")


def log_softmax(logits: torch.Tensor) -> torch.Tensor:
  max = einops.reduce(logits, "... d -> ... 1", "max")
  scaled = logits - max
  exp = torch.exp(scaled)
  sum = einops.reduce(exp, "... d -> ... 1", "sum")
  return scaled - torch.log(sum)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False
  ) -> dict[str, torch.Tensor]:
  logits = model(input_ids).logits
  lsm = log_softmax(logits)
  probs = lsm.gather(-1, labels.unsqueeze(dim=-1).long())
  result = {
    "log_probs": probs.squeeze()
  }
  if return_token_entropy:
    result["token_entropy"] = compute_entropy(logits)
  return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None= None
  ) -> torch.Tensor:
  return torch.sum(tensor * mask.long(), dim=dim) / normalize_constant
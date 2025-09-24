import json
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


def prepare_data(tmpl, json_obj):
  prompt = tmpl.format(**json_obj)
  answer = json_obj["answer"]
  lines = answer.split("\n")
  output = f"{"".join([f"<think>{s}</think>" for s in lines[:-1]])}<answer>{lines[-1][5:]}</answer>"
  if output.startswith("<think>"):
    output = output[len("<think>"):]
  return prompt, output


def prepare_batch(tmpl, data_list, tokenizer):
  pairs = [prepare_data(tmpl, json.loads(data)) for data in data_list]
  return helper.tokenize_prompt_and_output([i[0] for i in pairs], [i[1] for i in pairs], tokenizer)


def sft_train():
  with open("cs336_alignment/prompts/r1_zero.prompt") as fp:
    tmpl = fp.read()
  with open("data/gsm8k/test.jsonl") as fp:
    line = fp.readline()
  prepare_data(tmpl, json.loads(line))


if __name__ == "__main__":
  sft_train()

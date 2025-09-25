import re
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer

from cs336_alignment import helper

def parse_mmlu_response(resp: str) -> str | None:
  m = re.findall(r"[ABCD]", resp)
  if m:
    return m[-1]
  else:
    return None


def parse_gsm8k_response(resp: str) -> str | None:
  m = re.findall(r"\d+", resp)
  if m:
    return m[-1]
  else:
    return None


class SFTDataset(Dataset):
  def __init__(self, tokenizer, dataset_path, seq_length, shuffle):
    tmpl = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
{response}"""
    with open(dataset_path) as fp:
      data = [tmpl.format(**json.loads(str)) for str in fp.readlines()]
      seqs = [tokenizer(x)["input_ids"] for x in data]
    if shuffle:
      random.shuffle(seqs)
    tokens = []
    for seq in seqs:
      tokens.extend(seq)
      tokens.append(tokenizer.eos_token_id)
    self.tokens = torch.tensor(tokens, dtype=torch.long)
    self.seq_length = seq_length

  def __len__(self):
    return (self.tokens.shape[-1] - 1) // self.seq_length

  def __getitem__(self, i):
    if i >= len(self):
      raise StopIteration(f"idx {i}")
    start = i * self.seq_length
    return {
      "input_ids": self.tokens[start:start+self.seq_length],
      "labels": self.tokens[start+1:start+self.seq_length+1]
    }


def iterate_batches(
  dataset: Dataset,
  batch_size: int,
  shuffle: bool,
):
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def per_instance_dpo_loss(
  lm: torch.nn.Module,
  lm_ref: torch.nn.Module,
  tokenizer: PreTrainedTokenizerBase,
  beta: float,
  prompt: str,
  response_chosen: str,
  response_rejected: str,
) -> torch.Tensor:
  w_data = helper.tokenize_prompt_and_output([prompt], [response_chosen], tokenizer)
  l_data = helper.tokenize_prompt_and_output([prompt], [response_rejected], tokenizer)
  w_input, w_labels, w_masks = w_data["input_ids"], w_data["labels"].unsqueeze(-1), w_data["response_mask"]
  l_input, l_labels, l_masks = l_data["input_ids"], l_data["labels"].unsqueeze(-1), l_data["response_mask"]

  w, l = helper.log_softmax(lm(w_input).logits), helper.log_softmax(lm(l_input).logits)
  w_ref, l_ref = helper.log_softmax(lm_ref(w_input).logits), helper.log_softmax(lm_ref(l_input).logits)
  Pw = helper.masked_normalize(w.gather(-1, w_labels).squeeze(-1), w_masks, 1.0)
  Pw_ref = helper.masked_normalize(w_ref.gather(-1, w_labels).squeeze(-1), w_masks, 1.0)
  Pl = helper.masked_normalize(l.gather(-1, l_labels).squeeze(-1), l_masks, 1.0)
  Pl_ref = helper.masked_normalize(l_ref.gather(-1, l_labels).squeeze(-1), l_masks, 1.0)
  loss = -torch.log(torch.sigmoid(beta * ((Pw - Pw_ref) - (Pl - Pl_ref))))
  return loss.mean()



if __name__ == '__main__':
    FIXTURES_PATH = './tests/fixtures'
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    m = AutoModelForCausalLM.from_pretrained(FIXTURES_PATH + "/tiny-gpt2")
    m_ref = AutoModelForCausalLM.from_pretrained(FIXTURES_PATH + "/tiny-gpt2-ref")

    prompt = "The quick brown fox jumps over"
    good_response = "the lazy dog."
    bad_response = "their crazy frog."

    loss = per_instance_dpo_loss(m, m_ref, tokenizer, 0.5, prompt, good_response, bad_response)
    print(loss)
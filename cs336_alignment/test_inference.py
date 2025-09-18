import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import re
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
# from vllm import LLM, SamplingParams


def reward_fn(response, gt_answer, format_response=True):
  if format_response:
    response_start = "Assistant: "
    start = response.find(response_start)
    real_response = response[start+len(response_start):]
  else:
    real_response = response
  real_answer = gt_answer.split("\n")[-1][5:]
  return r1_zero_reward_fn(real_response, real_answer)


def evaluate(model_path, output_file):
  device = "mps"
  model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map=device)
  tokenizer = AutoTokenizer.from_pretrained(model_path)

  with open("data/gsm8k/test.jsonl") as fp:
    # datasets = [json.loads(fp.readline()) for _ in range(5)]
    datasets = [json.loads(line) for line in fp.readlines() if len(line) > 2]

  with open("cs336_alignment/prompts/r1_zero.prompt") as fp:
    tmpl = fp.read()

  prompts = [(tmpl.format(**data), data["answer"]) for data in datasets]

  class StopsOnTokens(StoppingCriteria):
      def __init__(self, stop_str: str, tokenizer, start=0):
          self.stop_str = stop_str
          self.tokenizer = tokenizer
          self.stop_ids = tokenizer.encode(stop_word, add_special_tokens=False)
          self.start = start

      def __call__(self, input_ids, scores, **kwargs) -> bool:
          # 把当前序列解码，看是否包含 stop_str
          text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
          return self.stop_str in text[self.start:]

  stop_word = "</answer>"
  stop_criteria = StoppingCriteriaList([
      StopsOnTokens("</answer>", tokenizer, len(tmpl))
  ])

  total = 0
  correct = 0
  generated = []
  for prompt, real_answer in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
      print(f"==========")
      outputs = model.generate(**inputs, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id, stopping_criteria=stop_criteria)
      decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
      answers = re.findall(r"<answer>(.*?)</answer>", decoded)
      if answers is not None:
        answer = answers[-1]
      else:
        answer = None
      generated.append(decoded)
      print(decoded)
      # print("answer:", answer)
      # print("real answer:", real_answer)
      # print("is value equal", reward_fn(answer, real_answer.split("\n")[-1][5:]))
      # total += 1
      # if reward_fn(answer, real_answer.split("\n")[-1][5:]):
      #   correct += 1
      # print(f"process total {total}, correct {correct}, ratio {correct / total}")

  with open(output_file, "wb") as wp:
    import pickle
    pickle.dump(generated, wp)


def main():
   evaluate("./models/Qwen2.5-Math-1.5B", "data/outputs.pickle")

# sampling_params = SamplingParams(
#   temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
# )
# llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")
# outputs = llm.generate(prompts, sampling_params)
# for output in outputs:
#   prompt = output.prompt
#   generated_text = output.outputs[0].text
#   print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
if __name__ == "__main__":
  a = torch.Tensor([1, 2, 3, 4])
  print(a.unsqueeze(dim=-1))
  # with open("data/gsm8k/test.jsonl") as fp:
  #   # datasets = [json.loads(fp.readline()) for _ in range(5)]
  #   datasets = [json.loads(line) for line in fp.readlines() if len(line) > 2]
  # with open("data/outputs.pickle", "rb") as fp:
  #   import pickle
  #   outputs = pickle.load(fp)
  # total = 0
  # format_correct = 0
  # correct = 0
  # for gt, response in zip(datasets, outputs):
  #   answer = gt["answer"]
  #   total += 1
  #   reward = reward_fn(response, answer)
  #   reward1 = reward_fn(response, answer, False)
  #   if reward["answer_reward"] != reward1["answer_reward"]:
  #     print(gt)
  #     print(response)
  #     print(reward)
  #     print(reward1)
  #     break
    # format_correct += reward["format_reward"]
    # correct += reward["answer_reward"]
  # print(total, format_correct, correct)
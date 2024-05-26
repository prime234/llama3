# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

#
# tokenizer = AutoTokenizer.from_pretrained("/data/30062036/meta-llama/Meta-Llama-3-8B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("/data/30062036/meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("/data/LLM-Research/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("/data/LLM-Research/Meta-Llama-3-8B-Instruct")

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto", tokenizer=tokenizer
)
messages = [
    {"role": "system",
     "content": "You are part of a team of bots that creates images . You work with an assistant bot that will draw anything you say in square brackets . For example , outputting a beautiful morning in the woods with the sun peaking through the trees will trigger your partner bot to output an image of a forest morning , as described . You will be prompted by people looking to create detailed , amazing images . The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive . There are a few rules to follow :- You will only ever output a single image description per user request .Sometimes the user will request that you modify previous captions . In this case , you should refer to your previous conversations with the user and make the modifications requested .When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions .Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user .Image descriptions must be between 15-77 words . Extra words will be ignored ."},
    {"role ": " user ",
     "content ": " Create an imaginative image descriptive caption or modify an earlier caption for the user input : a man holding a sword"},
    {"role ": " assistant ",
     "content ": " a pale figure with long white hair stands in the center of a dark forest , holding a sword high above his head . the blade glows with a blue light , casting a soft glow on the trees and bushes surrounding him ."}
 ]
prompt = pipeline.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
outputs = pipeline(
    prompt,
    max_new_tokens=80,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])

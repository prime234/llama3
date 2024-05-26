import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
# tokenizer = AutoTokenizer.from_pretrained("/data/LLM-Research/Meta-Llama-3-8B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("/data/LLM-Research/Meta-Llama-3-8B-Instruct")
model_path = "/data/LLM-Research/Meta-Llama-3-8B-Instruct"
tokenizer_path = "/data/LLM-Research/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# 定义对话生成函数
def generate_response(input_text):
    # 使用模型生成回复
    response = model.generate(
        tokenizer.encode(input_text, return_tensors="pt"),
        max_length=50,  # 调整生成的最大长度
        temperature=0.7,  # 温度参数
        num_return_sequences=1,  # 生成的回复序列数量
        pad_token_id=tokenizer.eos_token_id,
        early_stopping=True
    )
    return tokenizer.decode(response[0], skip_special_tokens=True)

# 创建 Gradio 界面
inputs = gr.Textbox(lines=7, label="输入对话：")
outputs = gr.Textbox(label="回复：", readonly=True)

title = "对话生成器"
description = "输入一段对话文本，生成对应的回复。"
examples = [
    ["你好，你是谁？"],
    ["天气真好，想出去散步。"],
    ["今天的工作做得怎么样？"]
]

gr.Interface(
    fn=generate_response,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    examples=examples
).launch()

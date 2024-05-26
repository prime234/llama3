import transformers
import torch
import os
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
# 加载分词器和模型
llama_tokenizer = AutoTokenizer.from_pretrained("/data/LLM-Research/Meta-Llama-3-8B-Instruct")
llama_model = AutoModelForCausalLM.from_pretrained("/data/LLM-Research/Meta-Llama-3-8B-Instruct")

# 加载文本评分模型
scorer_model = AutoModelForSequenceClassification.from_pretrained("prithivida/parrot-finetune-prompt")
scorer_tokenizer = AutoTokenizer.from_pretrained("prithivida/parrot-finetune-prompt")

# 设置文本生成管道
pipeline = transformers.pipeline(
    "text-generation",
    model=llama_model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    tokenizer=llama_tokenizer
)

# 定义系统消息
system_message = {
    "role": "system",
    "content": "You are part of a team of bots that creates images. You work with an assistant bot that will draw anything you say in square brackets. For example, outputting a beautiful morning in the woods with the sun peaking through the trees will trigger your partner bot to output an image of a forest morning, as described. You will be prompted by people looking to create detailed, amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive. There are a few rules to follow:- You will only ever output a single image description per user request.Sometimes the user will request that you modify previous captions. In this case, you should refer to your previous conversations with the user and make the modifications requested.When modifications are requested, you should not simply make the description longer. You should refactor the entire description to integrate the suggestions.Other times the user will not want modifications, but instead want a new image. In this case, you should ignore your previous conversation with the user.Image descriptions must be between 15-77 words. Extra words will be ignored."
}

# 定义对话历史
messages = [system_message]

# 创建输出目录
output_dir = Path("prompts_output")
output_dir.mkdir(parents=True, exist_ok=True)

while True:
    user_input = input("Enter a prompt or 'exit' to quit: ")
    if user_input.lower() == 'exit':
        break

    # 添加用户输入到对话历史
    messages.append({"role": "user", "content": f"Create an imaginative image descriptive caption: {user_input}"})

    # 生成多个扩写后的prompts
    prompts = []
    for _ in range(5):  # 生成5个prompts
        prompt = llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        terminators = [llama_tokenizer.eos_token_id, llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = pipeline(
            prompt,
            max_new_tokens=80,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            num_return_sequences=1  # 每次只生成一个prompt
        )
        generated_text = outputs[0]["generated_text"][len(prompt):]
        prompts.append(generated_text)

    # 使用文本评分模型评估prompts
    prompt_scores = []
    for prompt in prompts:
        inputs = scorer_tokenizer(prompt, return_tensors="pt")
        outputs = scorer_model(**inputs)[0]
        score = outputs[0][1].item()  # 获取"好"标签的分数
        prompt_scores.append(score)

    # 对prompts进行排序,选择评分最高的prompt
    sorted_prompts = [prompt for _, prompt in sorted(zip(prompt_scores, prompts), reverse=True)]
    best_prompt = sorted_prompts[0]

    # 将输入和输出保存到文件
    input_output_file = output_dir / f"{user_input.replace(' ', '_')}.txt"
    with input_output_file.open("a", encoding="utf-8") as f:
        f.write(f"Input: {user_input}\n")
        f.write(f"Output: {best_prompt}\n\n")

    # 打印生成的最佳prompt
    print(f"Generated prompt: {best_prompt}")

    # 添加助手输出到对话历史
    messages.append({"role": "assistant", "content": best_prompt})
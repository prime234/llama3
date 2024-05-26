# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

import transformers
import torch

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("/data/LLM-Research/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("/data/LLM-Research/Meta-Llama-3-8B-Instruct")

# 设置文本生成管道
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    tokenizer=tokenizer
)

# 定义系统消息
system_message = {
    "role": "system",
    "content": "You are part of a team of bots that creates images. You work with an assistant bot that will draw anything you say in square brackets. For example, outputting a beautiful morning in the woods with the sun peaking through the trees will trigger your partner bot to output an image of a forest morning, as described. You will be prompted by people looking to create detailed, amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive. There are a few rules to follow:- You will only ever output a single image description per user request.Sometimes the user will request that you modify previous captions. In this case, you should refer to your previous conversations with the user and make the modifications requested.When modifications are requested, you should not simply make the description longer. You should refactor the entire description to integrate the suggestions.Other times the user will not want modifications, but instead want a new image. In this case, you should ignore your previous conversation with the user.Image descriptions must be between 15-77 words. Extra words will be ignored."
}

# 定义对话历史
messages = [system_message]

while True:
    user_input = input("Enter a prompt or 'exit' to quit: ")
    if user_input.lower() == 'exit':
        break

    # 添加用户输入到对话历史
    messages.append({"role": "user", "content": f"Create an imaginative image descriptive caption: {user_input}"})

    # 生成扩写后的prompt
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    outputs = pipeline(
        prompt,
        max_new_tokens=80,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    generated_text = outputs[0]["generated_text"][len(prompt):]

    # 打印生成的prompt
    print(f"Generated prompt: {generated_text}")

    # 添加助手输出到对话历史
    messages.append({"role": "assistant", "content": generated_text})
#
# 这段代码的主要流程如下:
#
# 1. 加载分词器和模型,设置文本生成管道。
# 2. 定义系统消息,描述生成图像描述的任务。
# 3. 进入一个循环,等待用户输入关键词。
# 4. 将用户输入添加到对话历史中。
# 5. 使用文本生成管道生成扩写后的prompt。
# 6. 打印生成的prompt,并将其添加到对话历史中。
# 7. 如果用户输入 'exit',则退出循环。
#
# 在运行时,用户可以输入关键词,代码会自动将其扩写成详细的图像描述prompt。生成的prompt会打印出来,并添加到对话历史中,以便后续根据对话历史进行修改。
#
# 注意,由于模型权重较大,加载过程可能需要一些时间,并且需要足够的内存。确保你有足够的计算资源来运行这个大型模型。
from threading import Thread

import gradio as gr
import spaces
from modelscope import snapshot_download
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer)
#
# model_file = 'LLM-Research/Meta-Llama-3-8B-Instruct'
#
# cache_dir = './'
#
# snapshot_download(model_file,
#                   cache_dir=cache_dir)

tokenizer = AutoTokenizer.from_pretrained("/data/LLM-Research/Meta-Llama-3-8B-Instruct")
model_file = AutoModelForCausalLM.from_pretrained("/data/LLM-Research/Meta-Llama-3-8B-Instruct", device_map="auto")
# Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(cache_dir + model_file)
# model = AutoModelForCausalLM.from_pretrained(cache_dir + model_file, device_map="auto")  # to("cuda:0")
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


@spaces.GPU(duration=120)
def chat_llama3_8b(message: str,
                   history: list,
                   temperature: float,
                   max_new_tokens: int
                   ) -> str:
    """
    Generate a streaming response using the llama3-8b model.
    Args:
        message (str): The input message.
        history (list): The conversation history used by ChatInterface.
        temperature (float): The temperature for generating the response.
        max_new_tokens (int): The maximum number of new tokens to generate.
    Returns:
        str: The generated response.
    """
    conversation = []
    for user, assistant in history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        eos_token_id=terminators,
    )
    # This will enforce greedy generation (do_sample=False) when the temperature is passed 0, avoiding the crash.
    if temperature == 0:
        generate_kwargs['do_sample'] = False

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        print(outputs)
        yield "".join(outputs)


# Gradio block
chatbot = gr.Chatbot(height=450, label='Gradio ChatInterface')

with gr.Blocks(fill_height=True, css=css) as demo:
    gr.Markdown(DESCRIPTION)
    gr.ChatInterface(
        fn=chat_llama3_8b,
        chatbot=chatbot,
        fill_height=True,
        additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
        additional_inputs=[
            gr.Slider(minimum=0,
                      maximum=1,
                      step=0.1,
                      value=0.95,
                      label="Temperature",
                      render=False),
            gr.Slider(minimum=128,
                      maximum=4096,
                      step=1,
                      value=512,
                      label="Max new tokens",
                      render=False),
        ],
        examples=[
            ['你认为生活中最重要的是什么？'],
            ['How to setup a human base on Mars? Give short answer.'],
            ['请使用中文回答，讲述一部你觉得最好的电影'],
            ['Explain theory of relativity to me like I’m 8 years old.'],
            ['梯度下降是什么意思'],
            ['What is 9,000 * 9,000?'],
            ['如何过好人生'],
            ['Write a pun-filled happy birthday message to my friend Alex.'],
            ['请讲一个你最喜欢的古代诗人和他的诗歌'],
            ['Justify why a penguin might make a good king of the jungle.']
        ],
        cache_examples=False,
    )

    gr.Markdown(LICENSE)

if __name__ == "__main__":
    demo.launch()
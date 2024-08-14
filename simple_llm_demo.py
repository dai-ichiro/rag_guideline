import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

system_prompt_text = "あなたは親切なAIアシスタントです。"
init = {
    "role": "system",
    "content": system_prompt_text,
}

model = AutoModelForCausalLM.from_pretrained(
    "calm3-22b-chat-4bit",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("cyberagent/calm3-22b-chat")
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

def call_llm(
    message: str,
    history: list[dict],
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    history_openai_format = []
    if len(history) == 0:
        history_openai_format.append(init)
        history_openai_format.append({"role": "user", "content": message})
    else:
        history_openai_format.append(init)
        for human, assistant in history:
            history_openai_format.append({"role": "user", "content": human})
            history_openai_format.append({"role": "assistant", "content": assistant})
        history_openai_format.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(
        history_openai_format,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    generation_kwargs = dict(
        inputs=input_ids,
        streamer=streamer,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text

demo = gr.ChatInterface(
    fn=call_llm,
    title="CALM3-22B-Chat-4bit",
    stop_btn="Stop Generation",
    cache_examples=False,
    multimodal=False,
    additional_inputs_accordion=gr.Accordion(
        label="Parameters", open=False, render=False
    ),
    additional_inputs=[
        gr.Slider(
            minimum=1,
            maximum=4096,
            step=1,
            value=1024,
            label="Max tokens",
            visible=True,
            render=False,
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            step=0.1,
            value=0.3,
            label="Temperature",
            visible=True,
            render=False,
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            step=0.1,
            value=1.0,
            label="Top-p",
            visible=True,
            render=False,
        ),
    ],
)
demo.launch(share=False)
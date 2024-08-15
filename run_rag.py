import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

system_prompt_base = '''あなたは親切なAIアシスタントです。以下の与えらえた文章を参照してユーザーの質問にできるだけ丁寧に日本語で回答して下さい。与えられた文章に記載されていない内容の質問には「わかりません」と回答して下さい。

```
{context}
```'''
model = AutoModelForCausalLM.from_pretrained(
    "calm3-22b-chat-4bit",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("cyberagent/calm3-22b-chat")
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

embed_model_id = "pkshatech/GLuCoSE-base-ja"
embeddings = HuggingFaceEmbeddings(model_name=embed_model_id)

db = FAISS.load_local(
    "faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
def question_extention(
        chat_history: str
) -> str:
    query = (
        "以下にチャットの履歴があります。最後の行がユーザーの質問です。チャット履歴をもとにユーザーの質問をそれ自体で内容がわかるように簡単に書き換えて下さい。"
        "\n```\n"
        f"{chat_history}"
        "\n```"
    )
    messages = [
    {"role": "system", "content": "あなたは親切なAIアシスタントです。"},
    {"role": "user", "content": query}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    tokens = model.generate(
        input_ids,
        max_new_tokens=256,
        do_sample=True,
    )
    out = tokenizer.decode(tokens[0][len(input_ids[0]):], skip_special_tokens=True)
    return out

def call_llm(
    message: str,
    history: list[dict],
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    if len(history) == 0:
        final_question = message
    else:
        chat_history = ""
        for human, assistant in history:
            chat_history += f"{human}\n"
            chat_history += f"{assistant}\n"
        chat_history += message
        final_question = question_extention(chat_history)

    print(final_question)

    history_openai_format = []
    docs = db.similarity_search(final_question, k=1)
    context = docs[0].page_content
    #metadata_ref = docs[0].metadata["source"]
    system_prompt = system_prompt_base.format(context=context)
    init = {
        "role": "system",
        "content": system_prompt
    }
    history_openai_format.append(init)
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
    
    #print(metadata_ref)

demo = gr.ChatInterface(
    fn=call_llm,
    title="calm3-22b-chat-4bit",
    stop_btn="Stop Generation",
    cache_examples=False,
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

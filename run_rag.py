import requests
import torch
from janome.tokenizer import Tokenizer
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

url = "http://192.168.11.14:5000/flask_ollama"

system_prompt_base = '''あなたは親切なAIアシスタントです。以下の与えらえた文章を参照してユーザーの質問にできるだけ丁寧に日本語で回答して下さい。与えられた文章に記載されていない内容の質問には「わかりません」と回答して下さい。

```
{context}
```'''
model = AutoModelForCausalLM.from_pretrained(
    "calm3-22b-chat-4bit",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("cyberagent/calm3-22b-chat")
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

## faiss_retriever
embed_model_id = "pkshatech/GLuCoSE-base-ja"
embeddings = HuggingFaceEmbeddings(model_name=embed_model_id)

docs = torch.load("docs.langchaindata", weights_only=False)
faiss_retriever = FAISS.from_documents(docs, embeddings).as_retriever()

## bm25_retriever
t = Tokenizer()
def preprocess_jp(text: str) -> list[str]:
    tokenized_words = [token.surface for token in t.tokenize(text)]
    return tokenized_words

vectorizer = torch.load("vectorizer.langchaindata", weights_only=False)

bm25_retriever = BM25Retriever(
    vectorizer = vectorizer,
    preprocess_func=preprocess_jp,
    docs = docs
)

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)

def question_extention(
        message: str
) -> str:
    query = (
        "以下にチャットの履歴があります。最後の行がユーザーの質問です。チャット履歴をもとにユーザーの質問をそれ自体で内容がわかるように簡単に書き換えて下さい。最後に例を示しているので参照して下さい。"
        "\n```\n"
        f"{message}"
        "\n```\n"
        "以下が例になります"
        "\n```\n"
        "チャットの履歴：\n"
        "透析患者の目標Hb値は\n"
        "透析患者の目標Hb値は、週初めの採血で10 g/dL以上12 g/dL未満を推奨します。これは、腎性貧血治療ガイドラインに基づくもので、Hb値が10 g/dL未満の患者には複数回の検査でHb値が10 g/dL未満となった時点を治療開始基準としています。Hb値の上限については、12 g/dLを推奨し、Hb値12 g/dLを超える場合には減量・休薬を考慮することが推奨されています。\n"
        "腹膜透析患者は\n"
        "申し訳ありませんが、提供された文章には腹膜透析患者の目標Hb値に関する情報は含まれていません。そのため、腹膜透析患者の目標Hb値についてはわかりません。\n"
        "では保存期CKD患者では\n"
        "回答例：\n"
        "保存期CKD患者の目標Hb値はどのくらいですか？"
        "\n```"
    )
    data = {"prompt": query}
    response = requests.post(url, json=data)

    if response.status_code == 200:
        result = response.json()
        final_answer = result["result"]
    else:
        final_answer = message
    
    return final_answer

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

    print(f"final question: {final_question}")

    history_openai_format = []

    result = ensemble_retriever.invoke(final_question)
    context = result[0].page_content
    metadata_ref = result[0].metadata["source"]
    system_prompt = system_prompt_base.format(context=context)
    init = {
        "role": "system",
        "content": system_prompt,
    }
    history_openai_format.append(init)
    history_openai_format.append({"role": "user", "content": final_question})
    
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
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text
    
    print(metadata_ref)

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

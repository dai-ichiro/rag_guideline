import torch
from janome.tokenizer import Tokenizer
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

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

def call_llm(
    message: str,
    history: list[dict],
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    history_openai_format = []

    result = ensemble_retriever.invoke(message)
    context = result[0].page_content
    metadata_ref = result[0].metadata["source"]
    system_prompt = system_prompt_base.format(context=context)
    init = {
        "role": "system",
        "content": system_prompt,
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
    
    yield f"{generated_text}\n\n参照元:  {metadata_ref}"

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

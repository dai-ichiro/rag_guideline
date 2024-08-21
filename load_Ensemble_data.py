import torch
from janome.tokenizer import Tokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

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

result = ensemble_retriever.invoke("腹膜透析患者の目標Hb値は")

print(result[0].metadata["source"])



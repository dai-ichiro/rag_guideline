import torch
from pathlib import Path
from janome.tokenizer import Tokenizer
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

text_files = Path("textdata").glob("*.txt")

text_data = []
for text_file in text_files:
    with open(text_file.as_posix(), "r", encoding="utf-8") as f:
        text_data.append(
            {
                "text": f.read(),
                "file_name": text_file.name
            }
        )

docs = [
    Document(
        page_content=x["text"],
        metadata={"source": x["file_name"] }
    ) for x in text_data
]

torch.save(docs, "docs.langchaindata")

t = Tokenizer()
def preprocess_jp(input: str) -> list[str]:
    tokenized_words = [token.surface for token in t.tokenize(input)]
    return tokenized_words

vectorizer = BM25Okapi([preprocess_jp(x["text"]) for x in text_data])

torch.save(vectorizer, "vectorizer.langchaindata")






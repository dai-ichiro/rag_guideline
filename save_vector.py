from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

embed_model_id = "pkshatech/GLuCoSE-base-ja"
embeddings = HuggingFaceEmbeddings(model_name=embed_model_id)

text_files = Path("textdata").glob("*.txt")

text_data = []
for text_file in text_files:
    with open(text_file.as_posix(), "r", encoding="utf-8") as f:
        text_data.append(f.read())

docs = [Document(page_content=x) for x in text_data]

db = FAISS.from_documents(docs, embeddings)

db.save_local("faiss_index")

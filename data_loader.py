from google import genai
from pathlib import Path
from llama_index.readers.file import PDFReader, CSVReader, DocxReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os
import csv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def load_and_chunk_csv(path: str) -> list[str]:
    file_path = Path(path)

    docs = CSVReader().load_data(file=file_path)
    texts = [d.text for d in docs if getattr(d, "text", None)]

    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))

    return chunks

def load_and_chunk_url(url: str):
    docs = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def load_and_chunk_docx(path: str):
    # Load text from Word documents and produce sentence-based chunks
    docs = DocxReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def load_and_chunk(input_path_or_url: str):
    # Auto-detect source type (URL / file extension) and route to appropriate loader
    if input_path_or_url.lower().startswith(("http://", "https://")):
        return load_and_chunk_url(input_path_or_url)
    ext = os.path.splitext(input_path_or_url)[1].lower()
    if ext == ".pdf":
        return load_and_chunk_pdf(input_path_or_url)
    if ext == ".csv":
        return load_and_chunk_csv(input_path_or_url)
    if ext == ".docx":
        return load_and_chunk_docx(input_path_or_url)
    raise ValueError("Unsupported input type")

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
    )
    return [e.values for e in response.embeddings]



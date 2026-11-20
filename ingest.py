
import os

from dotenv import load_dotenv
from supabase import create_client

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials are missing from .env")


# Connect to Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# Step 1: Load the PDF
pdf_path = "data/company_policy.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"Loaded {len(documents)} PDF pages.")


# Step 2: Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} document chunks.")


# Step 3: Initialize the existing embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Step 4: Prevent accidental duplicate ingestion
existing = (
    supabase.table("documents")
    .select("id", count="exact")
    .limit(1)
    .execute()
)

if (existing.count or 0) > 0:
    raise RuntimeError(
        "Supabase documents table is not empty. "
        "Ingestion stopped to prevent duplicate records. "
        "Inspect the existing records before proceeding."
    )


# Step 5: Generate embeddings
texts = [chunk.page_content for chunk in chunks]

vectors = embeddings.embed_documents(texts)

if len(vectors) != len(chunks):
    raise RuntimeError("Embedding count does not match chunk count.")

if any(len(vector) != 384 for vector in vectors):
    raise RuntimeError("Unexpected embedding dimension. Expected 384.")


# Step 6: Prepare rows for Supabase
rows = []

for index, (chunk, vector) in enumerate(zip(chunks, vectors)):

    metadata = dict(chunk.metadata)
    metadata["chunk_index"] = index

    rows.append({
        "content": chunk.page_content,
        "metadata": metadata,
        "embedding": vector
    })


# Step 7: Insert document chunks in batches
batch_size = 50

for start in range(0, len(rows), batch_size):

    batch = rows[start:start + batch_size]

    supabase.table("documents").insert(batch).execute()

    print(
        f"Uploaded {min(start + batch_size, len(rows))}"
        f"/{len(rows)} chunks."
    )


print("\nDocument ingestion completed successfully.")
print(f"Total chunks stored in Supabase: {len(rows)}")

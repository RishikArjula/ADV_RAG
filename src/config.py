
import os
from typing import Any

from dotenv import load_dotenv
from supabase import create_client
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Read Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing from .env")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")

# Connect to Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize the existing embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Custom retriever using Supabase vector similarity search
class SupabaseRetriever(BaseRetriever):

    client: Any
    embedding_model: Any
    k: int = 3

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:

        # Convert the user's question into a 384-dimensional vector
        query_embedding = self.embedding_model.embed_query(query)

        # Call the PostgreSQL similarity search function
        response = self.client.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_count": self.k,
                "filter": {}
            }
        ).execute()

        results = response.data or []

        # Convert Supabase rows into LangChain Documents
        documents = []

        for row in results:
            metadata = row.get("metadata") or {}

            documents.append(
                Document(
                    page_content=row.get("content", ""),
                    metadata=metadata
                )
            )

        return documents


# Create the retriever
retriever = SupabaseRetriever(
    client=supabase,
    embedding_model=embeddings,
    k=3
)

# Initialize the existing Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=GROQ_API_KEY
)

print("Supabase connection configured successfully.")
print("Embedding model initialized successfully.")
print("Supabase retriever initialized successfully.")

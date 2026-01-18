import os
import sys
import time
from typing import Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
# Prefer token-based splitting; fallback to character splitter
try:
    from langchain_text_splitters import TokenTextSplitter
except ImportError:
    TokenTextSplitter = None
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

# Ensure the project root is on sys.path so 'src.*' imports work when running as a script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.embedding_factory import get_embedding_model, get_embedding_dimension
from pinecone import Pinecone, ServerlessSpec
from src.logging_utils import configure_logging, get_logger, log_call, log_operation

# (path already configured above)

from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME 

# Load environment variables from .env file
load_dotenv()
configure_logging()
_logger = get_logger(__name__)


@log_call
def ingest_data(pdf_path: str = "data/brihat-parashara-hora-shastra-english-v.pdf") -> Optional[None]:
    # 1. Load and Split
    if not os.path.exists(pdf_path):
        _logger.error(f"PDF not found at {pdf_path}")
        return

    _logger.info("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    with log_operation(_logger, "load_pdf"):
        raw_docs = loader.load()
    
    _logger.info(f"Splitting {len(raw_docs)} pages...")
    # Configure splitter: token-based if available, else character-based
    splitter_info = "TokenTextSplitter" if TokenTextSplitter else "RecursiveCharacterTextSplitter"
    _logger.info(f"Using splitter: {splitter_info}")
    if TokenTextSplitter:
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
        # Split per page to preserve metadata faithfully
        with log_operation(_logger, "split_documents_token"):
            docs = []
            for d in raw_docs:
                chunks = text_splitter.split_text(d.page_content)
                page = d.metadata.get("page")
                src = d.metadata.get("source")
                for idx, ch in enumerate(chunks):
                    docs.append(
                        type(d)(
                            page_content=ch,
                            metadata={**d.metadata, "page": page, "source": src, "chunk_index": idx},
                        )
                    )
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        with log_operation(_logger, "split_documents_char"):
            docs = text_splitter.split_documents(raw_docs)
            # Ensure chunk_index metadata exists for deterministic IDs
            for idx, d in enumerate(docs):
                if "chunk_index" not in d.metadata:
                    d.metadata["chunk_index"] = idx
    _logger.info(f"Total Chunks to process: {len(docs)}")

    # 2. Configure Pinecone
    with log_operation(_logger, "pinecone_client_init"):
        pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Determine index dimension via embedding_factory
    dimension = get_embedding_dimension()
    _logger.info(f"Embedding dimension selected: {dimension}")
    
    region = os.getenv("PINECONE_REGION", "us-east-1")
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        _logger.info(f"Creating Index '{PINECONE_INDEX_NAME}' with dimension {dimension}...")
        with log_operation(_logger, "pinecone_create_index"):
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=dimension, 
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region=region)
            )
    else:
        # Verify dimension matches
        index_info = pc.describe_index(PINECONE_INDEX_NAME)
        if index_info.dimension != dimension:
            _logger.critical(
                f"Index dimension mismatch: got {index_info.dimension}, expected {dimension}."
            )
            _logger.error("Please delete the index in Pinecone console and run this script again.")
            return
        # Best-effort region check for awareness (non-fatal)
        try:
            idx_region = getattr(index_info, "spec", {}).get("region", None)
            if idx_region and idx_region != region:
                _logger.warning(f"Index region {idx_region} differs from configured {region}")
        except Exception:
            pass

    # 3. Batch Ingestion with robust retry
    with log_operation(_logger, "init_embeddings"):
        embeddings = get_embedding_model()

    # Initialize VectorStore once (avoid per-batch overhead)
    namespace = "bphs"
    docsearch = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace,
    )

    batch_size = 20
    total_docs = len(docs)
    _logger.info(f"Starting ingestion with batch size {batch_size}...")

    for i in range(0, total_docs, batch_size):
        batch = docs[i : i + batch_size]
        _logger.info(f"Processing batch {i} to {i+len(batch)} / {total_docs}...")

        # Deterministic IDs: source + page + stable chunk index
        ids = []
        for j, d in enumerate(batch):
            src = os.path.basename(d.metadata.get("source", pdf_path))
            page = d.metadata.get("page", "na")
            local_idx = d.metadata.get("chunk_index", j)
            ids.append(f"{namespace}:{src}:p{page}:c{local_idx}")

        # Retry this batch until success or max retries
        retries = 0
        max_retries = 3
        backoff = 1.0
        while retries < max_retries:
            try:
                with log_operation(_logger, f"pinecone_upsert_batch_{i}"):
                    docsearch.add_documents(batch, ids=ids)
                _logger.info(f"Batch {i}/{total_docs}: Upserted {len(batch)} docs")
                time.sleep(1.0)  # gentle pacing
                break
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    _logger.critical(
                        f"Failed to ingest batch {i} after {max_retries} attempts. Skipping. Error: {e}"
                    )
                    # Optionally: persist failed batch for inspection
                    break
                _logger.warning(
                    f"Batch {i} failed (Attempt {retries}/{max_retries}). Retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)
                backoff *= 2  # exponential backoff

    _logger.info("Ingestion Complete!")

if __name__ == "__main__":
    _logger.info("Running ingest_data from __main__")
    ingest_data()

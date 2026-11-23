from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Split extracted PDF text
def split_text(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []

    for doc in documents:

        text_chunks = splitter.split_text(doc["text"])

        for chunk in text_chunks:

            chunks.append({
                "text": chunk,
                "page": doc["page"],
                "source": doc["source"]
            })

    return chunks


# Generate embeddings
def generate_embeddings(chunks):

    texts = [chunk["text"] for chunk in chunks]

    vectors = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    results = []

    for chunk, vector in zip(chunks, vectors):

        results.append({
            "text": chunk["text"],
            "embedding": vector.tolist(),
            "page": chunk["page"],
            "source": chunk["source"]
        })

    return results

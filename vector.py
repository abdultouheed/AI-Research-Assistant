
import chromadb
import hashlib


# =====================================
# 1. CONNECT TO EXISTING DATABASE
# =====================================

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="research_papers"
)


# =====================================
# 2. GENERATE UNIQUE CHUNK ID
# =====================================

def generate_chunk_id(item):

    unique_text = (
        item["source"]
        + str(item["page"])
        + item["text"]
    )

    return hashlib.sha256(
        unique_text.encode("utf-8")
    ).hexdigest()


# =====================================
# 3. STORE OR UPDATE EMBEDDINGS
# =====================================

def store_embeddings(vectors):

    if not vectors:
        return "No embeddings available."

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for item in vectors:

        chunk_id = generate_chunk_id(item)

        ids.append(chunk_id)

        embeddings.append(item["embedding"])

        documents.append(item["text"])

        metadatas.append({
            "source": item["source"],
            "page": item["page"]
        })

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    return f"{len(ids)} chunks stored or updated successfully."


# =====================================
# 4. SEARCH DOCUMENTS
# =====================================

def search_documents(query_vector, top_k=3):

    total = collection.count()

    if total == 0:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "ids": [[]]
        }

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, total)
    )

    return results


# =====================================
# 5. LIST UPLOADED PDFS
# =====================================

def list_pdfs():

    results = collection.get(
        include=["metadatas"]
    )

    pdf_counts = {}

    for metadata in results["metadatas"]:

        source = metadata["source"]

        pdf_counts[source] = (
            pdf_counts.get(source, 0) + 1
        )

    return pdf_counts


# =====================================
# 6. DELETE A SELECTED PDF
# =====================================

def delete_pdf(source):

    results = collection.get(
        where={"source": source},
        include=[]
    )

    ids = results["ids"]

    if not ids:
        return f"No chunks found for {source}."

    collection.delete(ids=ids)

    return f"Deleted {len(ids)} chunks from {source}."


# =====================================
# 7. DATABASE INFORMATION
# =====================================

def database_info():

    return {
        "total_chunks": collection.count(),
        "total_pdfs": len(list_pdfs())
    }



# =====================================
# 8. CHECK IF PDF ALREADY EXISTS
# =====================================

def pdf_exists(source):

    results = collection.get(
        where={"source": source},
        include=[]
    )

    return len(results["ids"]) > 0

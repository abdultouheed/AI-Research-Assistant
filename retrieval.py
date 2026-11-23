
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from embedding import model
from vector import collection


# =====================================
# 1. GET DOCUMENTS FOR A SPECIFIC PAPER
# =====================================

def get_all_documents(source=None):

    if source:
        results = collection.get(
            where={"source": source},
            include=["documents", "metadatas"]
        )
    else:
        results = collection.get(
            include=["documents", "metadatas"]
        )

    return results["documents"], results["metadatas"]


# =====================================
# 2. KEYWORD SEARCH - PAPER WISE
# =====================================

def keyword_search(question, documents, metadata):

    if not documents:
        return []

    tokenized_docs = [
        doc.lower().split() for doc in documents
    ]

    bm25 = BM25Okapi(tokenized_docs)

    query_tokens = question.lower().split()

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices:

        results.append({
            "text": documents[index],
            "metadata": metadata[index],
            "score": float(scores[index])
        })

    return results


# =====================================
# 3. SEMANTIC SEARCH - PAPER WISE
# =====================================

def semantic_search(question, source):

    # Count chunks belonging to this PDF
    results_count = collection.get(
        where={"source": source},
        include=[]
    )

    total = len(results_count["ids"])

    if total == 0:
        return []

    query_vector = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=total,
        where={"source": source}
    )

    output = []

    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        output.append({
            "text": doc,
            "metadata": meta,
            "score": float(distance)
        })

    return output


# =====================================
# 4. HYBRID SEARCH - PAPER WISE
# =====================================

def hybrid_search(question, source):

    documents, metadata = get_all_documents(source)

    if not documents:
        return []

    semantic_results = semantic_search(
        question, source
    )

    keyword_results = keyword_search(
        question,
        documents,
        metadata
    )

    scores = {}
    passages = {}

    for result_list in [
        semantic_results,
        keyword_results
    ]:

        for rank, item in enumerate(
            result_list, start=1
        ):

            key = (
                item["metadata"]["source"],
                item["metadata"]["page"],
                item["text"]
            )

            scores[key] = scores.get(key, 0) + (
                1 / (60 + rank)
            )

            passages[key] = item

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    final_results = []

    for key, score in ranked:

        item = passages[key].copy()

        item["rrf_score"] = score

        final_results.append(item)

    return final_results


# =====================================
# 5. CROSS-ENCODER RERANKING
# =====================================

reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_documents(question, documents):

    if not documents:
        return []

    pairs = [
        [question, item["text"]]
        for item in documents
    ]

    scores = reranker_model.predict(
        pairs,
        batch_size=32
    )

    ranked_documents = []

    for item, score in zip(documents, scores):

        new_item = item.copy()

        new_item["rerank_score"] = float(score)

        ranked_documents.append(new_item)

    ranked_documents.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return ranked_documents


# =====================================
# 6. ADAPTIVE PASSAGE SELECTION
# =====================================

def adaptive_top_k(ranked_results):

    if not ranked_results:
        return []

    if len(ranked_results) == 1:
        return ranked_results

    score_gaps = []

    for i in range(len(ranked_results) - 1):

        current_score = ranked_results[i]["rerank_score"]

        next_score = ranked_results[i + 1]["rerank_score"]

        difference = current_score - next_score

        score_gaps.append(difference)

    largest_gap_index = score_gaps.index(
        max(score_gaps)
    )

    selected = ranked_results[
        :largest_gap_index + 1
    ]

    print("Largest score gap:", max(score_gaps))
    print("Selected passages:", len(selected))

    return selected


# =====================================
# 7. PAPER-WISE RETRIEVAL PIPELINE
# =====================================

def retrieve_documents(question):

    print("\n===== PAPER-WISE RETRIEVAL =====")

    # Get all uploaded paper names
    all_results = collection.get(
        include=["metadatas"]
    )

    sources = sorted(set(
        metadata["source"]
        for metadata in all_results["metadatas"]
    ))

    print("Total papers:", len(sources))

    final_results = []

    # Process each paper independently
    for source in sources:

        print("\nProcessing:", source)

        # 1. Hybrid search for this paper
        results = hybrid_search(
            question,
            source
        )

        print("Hybrid candidates:", len(results))

        if not results:
            continue

        # 2. CrossEncoder reranking
        ranked_results = rerank_documents(
            question,
            results
        )

        print("Reranked candidates:", len(ranked_results))

        if not ranked_results:
            continue

        # 3. Adaptive selection for this paper
        selected = adaptive_top_k(
            ranked_results
        )

        print("Selected passages:", len(selected))

        final_results.extend(selected)

    print("\nTotal selected passages:", len(final_results))
    print("================================")

    return final_results

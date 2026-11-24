import os
import streamlit as st
from openai import OpenAI

from data import load_multiple_pdfs
from embedding import split_text, generate_embeddings
from vector import store_embeddings, list_pdfs, delete_pdf
from retrieval import retrieve_documents

# =====================================
# 1. CONFIGURATION
# =====================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODEL = "nvidia/nemotron-3.5-super-vl:free"

os.makedirs("uploads", exist_ok=True)


# =====================================
# 2. PDF PROCESSING
# =====================================

def process_pdfs(uploaded_files):

    existing_pdfs = list_pdfs()

    new_paths = []
    skipped = []

    for uploaded_file in uploaded_files:

        filename = os.path.basename(uploaded_file.name)

        if filename in existing_pdfs:
            skipped.append(filename)
            continue

        file_path = os.path.join("uploads", filename)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        new_paths.append(file_path)

    if not new_paths:
        return 0, skipped

    # Extract text
    documents = load_multiple_pdfs(new_paths)

    if not documents:
        return 0, skipped

    # Create chunks
    chunks = split_text(documents)

    # Generate embeddings
    vectors = generate_embeddings(chunks)

    # Store in ChromaDB
    result = store_embeddings(vectors)

    return len(vectors), skipped



# =====================================
# CONVERSATION MEMORY
# =====================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =====================================
# 3. RETRIEVE CONTEXT
# =====================================

def retrieve_context(question):

    final_results = retrieve_documents(question)

    if not final_results:
        return ""

    context = ""

    for item in final_results:

        context += f"""
Source: {item["metadata"].get("source")}
Page: {item["metadata"].get("page")}

Content:
{item["text"]}

----------------------------
"""

    return context



def rewrite_question(question):

    history = st.session_state.chat_history[-6:]

    if not history:
        return question

    conversation = ""

    for item in history:
        conversation += f"""
User: {item["question"]}
Assistant: {item["answer"]}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are a search query rewriting assistant.

Convert the user's latest question into a
standalone research question using conversation history.

Rules:
1. Resolve references such as it, they, this paper.
2. Preserve the original meaning.
3. Do not answer the question.
4. Do not introduce new information.
5. If the question is already clear, keep it unchanged.
6. Return only the rewritten question.
"""
            },
            {
                "role": "user",
                "content": f"""
Conversation history:
{conversation}

Latest question:
{question}

Rewrite this question for document retrieval.
"""
            }
        ],
        temperature=0
    )

    rewritten = response.choices[0].message.content

    return rewritten.strip() if rewritten else question



# =====================================
# 4. GENERATE ADVANCED RESEARCH ANSWER
# =====================================

def research_answer(question):

    try:

        # Retrieve relevant research passages
        retrieval_question = rewrite_question(question)

        context = retrieve_context(retrieval_question)

        if not context:
            yield "No relevant research information found in the uploaded PDFs."
            return

        # Get previous conversation history
        history = st.session_state.chat_history[-6:]

        # Convert history into messages
        conversation_messages = []

        for item in history:
            conversation_messages.append({
                "role": "user",
                "content": item["question"]
            })

            conversation_messages.append({
                "role": "assistant",
                "content": item["answer"]
            })

        response = client.chat.completions.create(

            model=MODEL,

            messages=[

                {
                                "role": "system",

                                "content": """
            You are an advanced AI Research Assistant
            and Scientific Research Analyst.

            Your responsibility is to understand, analyze,
            explain and compare research papers using
            the provided research context.

            ========================================
            1. RESEARCH PAPER UNDERSTANDING
            ========================================

            Identify and explain:

            - Research objectives
            - Problem statements
            - Research gaps
            - Proposed solutions
            - Novel contributions
            - Methodologies
            - Algorithms and architectures
            - Datasets
            - Experimental setup
            - Results and performance metrics
            - Advantages and limitations
            - Future research directions

            Only report information supported by the context.

            ========================================
            2. METHODOLOGY ANALYSIS
            ========================================

            When asked about methodology:

            - Explain the complete approach step by step.
            - Identify the models and algorithms used.
            - Explain why each technique is used.
            - Describe the overall workflow.
            - Explain how different components interact.
            - Explain mathematical equations when available.
            - Simplify complex technical concepts with examples.

            ========================================
            3. RESULTS AND PERFORMANCE ANALYSIS
            ========================================

            When asked about results:

            - Identify reported performance metrics.
            - Explain accuracy, precision, recall and F1-score.
            - Explain ROC-AUC and other relevant metrics.
            - Compare model performance when data is available.
            - Interpret experimental results.
            - Never invent numerical values.

            ========================================
            4. MULTIPLE PAPER COMPARISON
            ========================================

            When multiple research papers are provided:

            - Compare their research objectives.
            - Compare methodologies and algorithms.
            - Compare datasets and experimental approaches.
            - Compare reported results.
            - Identify similarities and differences.
            - Explain the strengths of each approach.

            Clearly identify which PDF supports each finding.

            ========================================
            5. CRITICAL RESEARCH ANALYSIS
            ========================================

            When asked to evaluate research:

            - Identify limitations mentioned by authors.
            - Explain research challenges.
            - Identify strengths of the proposed approach.
            - Discuss possible research gaps.
            - Suggest potential improvements when requested.

            Distinguish author-stated limitations from
            your own analytical observations.

            ========================================
            6. TECHNICAL EXPLANATIONS
            ========================================

            You should be capable of explaining:

            - Machine learning algorithms
            - Deep learning architectures
            - CNN, RNN, Transformers and other models
            - Feature extraction and selection
            - Classification and regression
            - Data preprocessing
            - Model evaluation
            - Mathematical formulas
            - System architectures
            - Experimental workflows

            Use examples and step-by-step explanations
            where they improve understanding.

            ========================================
            7. FACTUAL ACCURACY
            ========================================

            Follow these rules strictly:

            1. Never fabricate research findings.
            2. Never invent statistics or experimental results.
            3. Never generate fake citations or page numbers.
            4. Use the provided context as your primary evidence.
            5. Clearly identify missing information.
            6. Distinguish facts from assumptions and inferences.
            7. Do not claim the paper contains information
            that is not present in the retrieved context.
            8. If evidence is insufficient, explain the limitation.
            9. Do not blindly agree with the user's assumptions.
            10. Do not confuse information from different PDFs.

            ========================================
            8. RESPONSE STYLE
            ========================================

            - Be professional and technically accurate.
            - Use clear and understandable English.
            - Give direct answers first.
            - Provide detailed explanations when necessary.
            - Use headings, bullet points and tables when useful.
            - Avoid unnecessary repetition.
            - Do not make every answer excessively long.
            - Adapt the response to the actual question.

            ========================================
            9. SOURCE ATTRIBUTION
            ========================================

            For research-related answers:

            - Mention PDF filenames.
            - Mention relevant page numbers.
            - Connect findings to their respective sources.
            - Never invent source references.

            ========================================
            10. RESPONSE STRUCTURE
            ========================================

            Adapt the structure based on the question.

            General question:
            Answer:
            Explanation:
            Sources:

            Methodology question:
            Objective:
            Methodology:
            Working Process:
            Techniques Used:
            Sources:

            Results question:
            Experimental Setup:
            Results:
            Performance Analysis:
            Sources:

            Comparison question:
            Overview:
            Similarities:
            Differences:
            Comparison:
            Conclusion:
            Sources:

            Limitations question:
            Identified Limitations:
            Research Challenges:
            Possible Improvements:
            Sources:

            Do not force these structures when a simple
            answer is more appropriate.

            Your goal is to function as a reliable,
            analytical and knowledgeable research assistant.
            """
                },
                
                *conversation_messages,

                {
                    "role": "user",

                    "content": f"""
            RESEARCH PAPER CONTEXT:

            {context}

            USER QUESTION:

            {question}

            Analyze the available research evidence carefully.

            Answer the question accurately and clearly.
            Provide technical explanations where relevant.
            Use examples when they improve understanding.

            Mention the supporting PDF filenames and
            page numbers.

            If the provided context does not contain
            sufficient information, clearly acknowledge it.
            """
                }

            ],

            temperature=0.2,
            stream = True,
            max_tokens=2000

        )

        try:
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        finally:
            response.close()

    except Exception as e:
        yield f"Error generating answer: {str(e)}"

    #     answer = response.choices[0].message.content

    #     if not answer:
    #         return "The AI model returned an empty response."

    #     # Save current conversation
    #     st.session_state.chat_history.append({
    #         "question": question,
    #         "answer": answer
    #     })


    #     return answer

    # except Exception as e:

    #     return f"Error generating answer: {str(e)}"


# =====================================
# 5. STREAMLIT GUI
# =====================================

st.title("📚 AI Research Assistant")

st.caption(
    "Upload research papers and ask questions "
    "across your documents."
)

st.divider()

# PDF upload section
st.subheader("Upload Research Papers")

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Process PDFs", type="primary"):

    if not uploaded_files:
        st.warning("Please select at least one PDF.")

    else:
        with st.spinner("Processing your research papers..."):

            try:
                count, skipped = process_pdfs(uploaded_files)

                if count:
                    st.success(
                        f"Successfully uploaded"
                    )

                if skipped:
                    st.info(
                        "Already processed (skipped): "
                        + ", ".join(skipped)
                    )

                if not count and not skipped:
                    st.warning("No text could be extracted.")

            except Exception as e:
                st.error(f"Processing error: {str(e)}")


# =====================================
# 6. DISPLAY UPLOADED PDFS
# =====================================

st.divider()

st.subheader("Your Research Library")

pdfs = list_pdfs()

col1, col2 = st.columns(2)

with col1:
    st.metric("Total PDFs", len(pdfs))

# with col2:
#     st.metric("Total Chunks", sum(pdfs.values()))

if pdfs:

    for filename, chunks in pdfs.items():

        col1, col2, col3 = st.columns([5, 2, 1])

        with col1:
            st.write(f"📄 **{filename}**")

        # with col2:
        #     st.write(f"{chunks} chunks")

        with col3:
            if st.button("🗑️", key=f"delete_{filename}"):

                # Delete vectors from ChromaDB
                result = delete_pdf(filename)

                # Delete original PDF
                file_path = os.path.join("uploads", filename)

                if os.path.exists(file_path):
                    os.remove(file_path)

                st.success(result)
                st.rerun()

else:
    st.info("No research papers have been processed yet.")


# =====================================
# 7. ASK QUESTIONS
# =====================================

# st.divider()

# st.subheader("Ask Your Research Assistant")

# # New Chat button
# if st.button("＋ New Chat"):
#     st.session_state.chat_history = []
#     st.rerun()
# # Display previous conversation
# for item in st.session_state.chat_history:

#     with st.chat_message("user"):
#         st.markdown(item["question"])

#     with st.chat_message("assistant"):
#         st.markdown(item["answer"])


# # New question input
# question = st.chat_input("Ask your research question...")

# if question:

#     if not pdfs:
#         st.warning("Upload and process PDFs first.")

#     else:

#         # Display current question
#         with st.chat_message("user"):
#             st.markdown(question)

#         # Generate and display answer
#         with st.chat_message("assistant"):

#             with st.spinner("Searching and generating answer..."):

#                 try:
#                     answer = research_answer(question)
#                     st.markdown(answer)

#                 except Exception as e:
#                     st.error(f"Error: {str(e)}")



st.divider()

st.subheader("Ask Your Research Assistant")

# NEW: state
st.session_state.setdefault("pending_question", None)
st.session_state.setdefault("partial", "")

# NEW: stop callback
def stop_generation():
    ss = st.session_state
    if ss.pending_question:
        ss.chat_history.append({
            "question": ss.pending_question,
            "answer": ss.partial + "\n\n*(generation stopped)*",
        })
    ss.pending_question = None
    ss.partial = ""

# New Chat button
if st.button("＋ New Chat"):
    st.session_state.chat_history = []
    st.session_state.pending_question = None    # NEW
    st.session_state.partial = ""               # NEW
    st.rerun()

# Display previous conversation (unchanged)
for item in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(item["question"])
    with st.chat_message("assistant"):
        st.markdown(item["answer"])

# CHANGED: disable input while generating
question = st.chat_input(
    "Ask your research question...",
    disabled=st.session_state.pending_question is not None,
)

if question:
    if not pdfs:
        st.warning("Upload and process PDFs first.")
    else:
        # CHANGED: don't generate here, just store and rerun
        st.session_state.pending_question = question
        st.session_state.partial = ""
        st.rerun()

# NEW: generation block (replaces your old "Display current question / Generate" block)
if st.session_state.pending_question:
    q = st.session_state.pending_question

    with st.chat_message("user"):
        st.markdown(q)

    st.button("⏹ Stop generating", on_click=stop_generation)   # must be BEFORE streaming

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            with st.spinner("Searching and generating answer..."):
                for token in research_answer(q):
                    st.session_state.partial += token
                    placeholder.markdown(st.session_state.partial + "▌")

            placeholder.markdown(st.session_state.partial)
            st.session_state.chat_history.append(
                {"question": q, "answer": st.session_state.partial}
            )
            st.session_state.pending_question = None
            st.session_state.partial = ""
            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.pending_question = None
# import fitz
# import os


# def load_pdf(file_path):

#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"PDF not found: {file_path}")

#     if not file_path.lower().endswith(".pdf"):
#         raise ValueError("Only PDF files are supported.")

#     documents = []

#     pdf = fitz.open(file_path)

#     try:
#         for page_number, page in enumerate(pdf, start=1):

#             text = page.get_text("text").strip()

#             if text:

#                 documents.append({
#                     "text": text,
#                     "page": page_number,
#                     "source": os.path.basename(file_path)
#                 })

#     finally:
#         pdf.close()

#     return documents




import fitz
import os


def load_pdf(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    documents = []

    pdf = fitz.open(file_path)

    try:
        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text").strip()

            if text:

                documents.append({
                    "text": text,
                    "page": page_number,
                    "source": os.path.basename(file_path)
                })

    finally:
        pdf.close()

    return documents


# Load multiple PDFs
def load_multiple_pdfs(file_paths):

    all_documents = []

    for file_path in file_paths:

        try:
            documents = load_pdf(file_path)

            all_documents.extend(documents)

            print(
                f"Processed: {os.path.basename(file_path)} "
                f"| Pages extracted: {len(documents)}"
            )

        except Exception as e:

            print(f"Error processing {file_path}: {e}")

    return all_documents

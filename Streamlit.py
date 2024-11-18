import streamlit as st
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Load a multilingual embedding model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Directory to store PDFs
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""  # Use an empty string if extract_text() returns None
    return text

# Step 2: Split text into chunks
def split_text_into_chunks(text, chunk_size=200):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Step 3: Embed text chunks
def embed_chunks(chunks):
    return model.encode(chunks)

# Step 4: Build FAISS index with existing PDFs
def build_faiss_index(pdf_paths, chunk_size=200):
    chunk_info = []
    all_embeddings = []
    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text_into_chunks(text, chunk_size)
        embeddings = embed_chunks(chunks)
        all_embeddings.extend(embeddings)
        chunk_info.extend([(pdf_path, chunk) for chunk in chunks])
    all_embeddings = np.array(all_embeddings).astype("float32")
    faiss_index = faiss.IndexFlatL2(all_embeddings.shape[1])
    faiss_index.add(all_embeddings)
    return faiss_index, chunk_info

# Step 5: Check for similarity with submitted PDF
def check_plagiarism(submitted_pdf_path, faiss_index, chunk_info, threshold=0.20, top_k=3):
    submitted_text = extract_text_from_pdf(submitted_pdf_path)
    submitted_chunks = split_text_into_chunks(submitted_text)
    submitted_embeddings = embed_chunks(submitted_chunks)
    results = []
    for chunk, embedding in zip(submitted_chunks, submitted_embeddings):
        embedding = np.array([embedding]).astype("float32")
        distances, indices = faiss_index.search(embedding, k=top_k)
        for dist, idx in zip(distances[0], indices[0]):
            similarity = (1 - (dist / 2)) * 100  # Convert to percentage
            if similarity >= threshold * 100:
                matched_pdf_path, matched_chunk = chunk_info[idx]
                results.append({
                    "submitted_chunk": chunk,
                    "matched_pdf": matched_pdf_path,
                    "matched_chunk": matched_chunk,
                    "similarity": similarity
                })
    return results

# Streamlit interface
st.set_page_config(page_title="Program Similarity Checker", layout="centered")
st.title("Program Similarity Checker")

# Sidebar to show uploaded files
st.sidebar.header("Uploaded Files")
uploaded_files = os.listdir(DATA_DIR)
if uploaded_files:
    st.sidebar.write("### Stored PDFs:")
    for file in uploaded_files:
        st.sidebar.write(f"- {file}")
else:
    st.sidebar.write("No files uploaded yet.")

# Add custom CSS for styling the tabs
st.markdown(
    """
    <style>
    .tab-style {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        font-family: Arial, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Upload and store PDFs
st.subheader("Add PDFs to the Dataset")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file:
    file_name = uploaded_file.name
    if not file_name.endswith(".pdf"):
        st.error("Only PDF format is accepted.")
    else:
        save_path = os.path.join(DATA_DIR, file_name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{file_name}' has been added to the data folder.")

# Dynamically fetch all PDF files in the data folder
dataset_pdf_paths = [os.path.join(DATA_DIR, pdf) for pdf in os.listdir(DATA_DIR) if pdf.endswith(".pdf")]

if dataset_pdf_paths:
    # Build FAISS index with all available PDFs
    faiss_index, chunk_info = build_faiss_index(dataset_pdf_paths)

    # Upload a PDF for similarity check
    st.subheader("Check Similarity")
    similarity_file = st.file_uploader("Upload a PDF file to check similarity", type="pdf")
    if similarity_file:
        with open("temp_submitted.pdf", "wb") as f:
            f.write(similarity_file.getbuffer())
        similar_content = check_plagiarism("temp_submitted.pdf", faiss_index, chunk_info)

        if similar_content:
            st.success("Similar Content Found:")
            # Display results in tabs with Arabic support
            for match in similar_content[:2]:  # Limit to top 2 results
                tabs = st.tabs(["Submitted Chunk", "Matched Chunk"])
                with tabs[0]:
                    st.markdown(
                        f"""
                        <div class='tab-style' style="direction: rtl; text-align: right; font-family: 'Arial', sans-serif;">
                        {match['submitted_chunk']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with tabs[1]:
                    st.markdown(
                        f"""
                        <div class='tab-style' style="direction: rtl; text-align: right; font-family: 'Arial', sans-serif;">
                        Matched PDF: {os.path.basename(match['matched_pdf'])}<br>{match['matched_chunk']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f"""
                    <p style='color: red; font-weight: bold; direction: rtl; text-align: right; font-family: "Arial", sans-serif;">
                    Similarity Score: {match['similarity']:.2f}%
                    </p>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning("No similar content found.")
else:
    st.warning("No PDF files found in the data directory.")

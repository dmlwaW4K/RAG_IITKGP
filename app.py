from config import GOOGLE_API_KEY
import streamlit as st
import os
import fitz  # pymupdf
import pytesseract
from PIL import Image
import io
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# 1. Setup API Key - Securely
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("API Key not found in Streamlit secrets!")
    st.stop()
# Tesseract path for Windows — adjust if installed elsewhere
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── 2. Prompt ─────────────────────────────────────────────────────────────────
template = """You are an expert Financial Compliance Officer. Use the provided context to answer the question.
If the answer is not in the context, say 'Information not found in regulatory documentation.'
If the question is not related to finance, refuse to answer.
Always cite the source document name and page number.

Context: {context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

# ── 3. PDF Extractor (text-first, OCR fallback) ───────────────────────────────
def extract_docs_from_pdf(uploaded_file):
    """Try native text extraction first; fall back to OCR page by page."""
    pdf_bytes = uploaded_file.read()
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    docs = []

    for i, page in enumerate(pdf):
        # Try native text first
        text = page.get_text().strip()

        # If page has no text (scanned), OCR it
        if not text:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img).strip()

        if text:
            docs.append(Document(
                page_content=text,
                metadata={"source": uploaded_file.name, "page": i + 1}
            ))

    pdf.close()
    return docs

# ── 4. Page Setup ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="ComplianceGuard", page_icon="⚖️")
st.title("⚖️ ComplianceGuard: Finance Regulatory Agent")

# ── 5. Session State ──────────────────────────────────────────────────────────
if "chain" not in st.session_state:
    st.session_state.chain = None
if "ready" not in st.session_state:
    st.session_state.ready = False
if "last_file" not in st.session_state:
    st.session_state.last_file = None

# ── 6. File Upload ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Finance Regulation PDF", type="pdf")

# Reset if new file uploaded
if uploaded_file and uploaded_file.name != st.session_state.last_file:
    st.session_state.ready = False
    st.session_state.chain = None

if uploaded_file and not st.session_state.ready:
    # ── 7. Extract Text ───────────────────────────────────────────────────────
    with st.spinner("📖 Reading PDF (OCR if needed — may take ~30s for scanned docs)..."):
        docs = extract_docs_from_pdf(uploaded_file)

    if not docs:
        st.error("❌ Could not extract any text even with OCR. The PDF may be corrupted.")
        st.stop()

    st.info(f"📄 Extracted text from {len(docs)} pages")

    # ── 8. Split ──────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = splitter.split_documents(docs)
    st.info(f"🔪 Split into {len(texts)} chunks")

    # ── 9. Embed ──────────────────────────────────────────────────────────────
    with st.spinner("🔢 Creating embeddings..."):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
            db = Chroma.from_documents(texts, embeddings)
        except Exception as e:
            st.error(f"❌ Embedding failed: {e}")
            st.stop()

    # ── 10. Build Chain ───────────────────────────────────────────────────────
    retriever = db.as_retriever(search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.1,
    google_api_key=GOOGLE_API_KEY
)

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'doc')}, Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in docs
        )

    st.session_state.chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    st.session_state.ready = True
    st.session_state.last_file = uploaded_file.name
    st.success(f"✅ Ready! {len(texts)} chunks indexed.")

# ── 11. Query UI — ALWAYS rendered, never inside upload block ─────────────────
st.divider()

if st.session_state.ready:
    st.subheader("💬 Ask a Compliance Question")
else:
    st.info("⬆️ Upload a PDF above to get started.")

query = st.text_area(
    "Your question:",
    placeholder="e.g. What are the disclosure requirements for listed companies?",
    height=120,
    key="query_input",
    disabled=not st.session_state.ready
)

# Button is ALWAYS outside any conditional block
clicked = st.button(
    "🔍 Ask ComplianceGuard",
    type="primary",
    use_container_width=True,
    disabled=not st.session_state.ready
)

if clicked:
    if not query.strip():
        st.warning("⚠️ Please type a question first.")
    else:
        with st.spinner("🔎 Analyzing regulation..."):
            try:
                response = st.session_state.chain.invoke(query)
                st.write("### 📋 Answer")
                st.write(response)
            except Exception as e:
                st.error(f"❌ Query failed: {e}")

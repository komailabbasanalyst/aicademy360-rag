import os
import tempfile
import uuid
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(
    page_title="Aicademy 360 RAG Knowledge Studio",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(106, 92, 255, 0.15), transparent 24%),
            radial-gradient(circle at 90% 10%, rgba(0, 198, 174, 0.12), transparent 22%),
            #07111f;
        color: #eef4ff;
    }

    [data-testid="stSidebar"] {
        background: rgba(8, 20, 36, 0.96);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 4rem;
        max-width: 1450px;
    }

    .hero {
        padding: 1.4rem 1.5rem;
        border-radius: 24px;
        background: linear-gradient(120deg, rgba(88,76,255,.30), rgba(0,190,170,.17));
        border: 1px solid rgba(255,255,255,.10);
        box-shadow: 0 24px 70px rgba(0,0,0,.22);
        margin-bottom: 1rem;
    }

    .hero-kicker {
        color: #9fafff;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
    }

    .hero h1 {
        margin: .25rem 0 .4rem 0;
        font-size: clamp(2rem, 4vw, 3.6rem);
        line-height: 1.02;
        color: white;
    }

    .hero p {
        color: #c7d3e6;
        font-size: 1rem;
        max-width: 850px;
        margin: 0;
    }

    .glass-card {
        background: rgba(12, 26, 45, 0.72);
        border: 1px solid rgba(255,255,255,.09);
        border-radius: 20px;
        padding: 1.05rem 1.1rem;
        min-height: 118px;
        box-shadow: 0 18px 50px rgba(0,0,0,.16);
    }

    .card-label {
        color: #8fa3be;
        font-size: .76rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-weight: 700;
    }

    .card-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: white;
        margin-top: .25rem;
    }

    .card-sub {
        color: #8fa3be;
        font-size: .82rem;
        margin-top: .2rem;
    }

    .pipeline {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: .55rem;
        margin: .6rem 0 1rem 0;
    }

    .pipe-step {
        text-align: center;
        padding: .75rem .45rem;
        border-radius: 14px;
        background: rgba(255,255,255,.045);
        border: 1px solid rgba(255,255,255,.07);
        color: #dce8fa;
        font-size: .82rem;
        font-weight: 700;
    }

    .evidence-card {
        border-radius: 16px;
        padding: .9rem 1rem;
        margin: .55rem 0;
        background: rgba(255,255,255,.035);
        border-left: 4px solid #6f7cff;
        border-top: 1px solid rgba(255,255,255,.06);
        border-right: 1px solid rgba(255,255,255,.06);
        border-bottom: 1px solid rgba(255,255,255,.06);
    }

    .source-chip {
        display: inline-block;
        padding: .2rem .55rem;
        border-radius: 999px;
        margin-right: .35rem;
        background: rgba(111,124,255,.16);
        border: 1px solid rgba(111,124,255,.28);
        color: #cbd1ff;
        font-size: .75rem;
        font-weight: 700;
    }

    .small-note {
        color: #8296b1;
        font-size: .82rem;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 12px;
        font-weight: 700;
    }

    @media (max-width: 900px) {
        .pipeline { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
</style>
""",
    unsafe_allow_html=True,
)
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load the local embedding model once and reuse it across Streamlit reruns."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )


def build_knowledge_base(files, chunk_size, overlap):
    """Read PDFs, add source metadata, split them, embed them, and create Chroma."""
    all_pages = []

    for uploaded_pdf in files:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(uploaded_pdf.getvalue())
        temp_file.close()

        try:
            pages = PyPDFLoader(temp_file.name).load()
        finally:
            os.unlink(temp_file.name)

        for page in pages:
            page.metadata["source_file"] = uploaded_pdf.name

        all_pages.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    chunks = splitter.split_documents(all_pages)

    
    collection_name = f"aicademy_{uuid.uuid4().hex[:10]}"

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=load_embedding_model(),
        collection_name=collection_name,
    )

    return vectorstore, all_pages, chunks


def retrieve_documents(vectorstore, query, search_mode, top_k):
    """Retrieve document chunks using either similarity search or MMR."""
    if search_mode == "MMR - diverse evidence":
        return vectorstore.max_marginal_relevance_search(
            query,
            k=top_k,
            fetch_k=max(12, top_k * 4),
        )

    return vectorstore.similarity_search(query, k=top_k)


def create_context(docs, max_chars=12000):
    """Turn retrieved chunks into one labeled context block for the LLM."""
    parts = []
    used = 0

    for number, doc in enumerate(docs, start=1):
        file_name = doc.metadata.get("source_file", "Unknown file")
        page_number = doc.metadata.get("page", 0) + 1
        piece = (
            f"[S{number} | {file_name} | page {page_number}]\n"
            f"{doc.page_content}\n"
        )

        if used + len(piece) > max_chars:
            break

        parts.append(piece)
        used += len(piece)

    return "\n".join(parts)


def run_grounded_answer(llm, question, docs, strict_mode=True):
    """Ask the LLM to answer from retrieved evidence and cite source labels."""
    context = create_context(docs)

    if strict_mode:
        rule = (
            "Use only the supplied evidence. If the answer is not supported by the "
            "evidence, say: I could not find that answer in the uploaded knowledge base."
        )
    else:
        rule = (
            "Prioritize the supplied evidence. Clearly separate any general explanation "
            "from facts that come from the documents."
        )

    prompt = f"""
You are the research assistant inside Aicademy 360 RAG Knowledge Studio.
{rule}
Write clearly for a student.
Cite useful evidence with labels such as [S1], [S2].
Do not invent page numbers or sources.

EVIDENCE:
{context}

QUESTION:
{question}
"""

    return llm.invoke(prompt).content


def document_sample(chunks, selected_file, max_chars=15000):
    """Create a balanced sample from the beginning and end of a selected document."""
    if selected_file == "All documents":
        selected = chunks
    else:
        selected = [
            chunk
            for chunk in chunks
            if chunk.metadata.get("source_file") == selected_file
        ]

    if not selected:
        return ""

    
    sample_chunks = selected[:6]
    if len(selected) > 8:
        sample_chunks += selected[-3:]

    text = "\n\n".join(chunk.page_content for chunk in sample_chunks)
    return text[:max_chars]



for key, value in {
    "vectorstore": None,
    "pages": [],
    "chunks": [],
    "file_names": [],
    "last_question": "",
    "last_answer": "",
    "last_evidence": [],
    "chunk_size_used": None,
    "overlap_used": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value



with st.sidebar:
    st.markdown("### RAG Control Center")

    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        help="You can also place GROQ_API_KEY inside a .env file.",
    )
    api_key = api_key_input or os.getenv("GROQ_API_KEY")

    st.markdown("#### Generation")
    model_name = st.selectbox(
        "Groq model",
        ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
        index=0,
    )
    temperature = st.slider("Creativity", 0.0, 1.0, 0.1, 0.1)
    strict_mode = st.toggle("Strict document-only answers", value=True)

    st.markdown("#### Retrieval")
    search_mode = st.selectbox(
        "Search strategy",
        ["MMR - diverse evidence", "Similarity - closest chunks"],
    )
    top_k = st.slider("Top-K chunks", 2, 8, 4)

    st.markdown("#### Indexing")
    chunk_size = st.slider("Chunk size", 400, 1600, 900, 100)
    overlap = st.slider("Chunk overlap", 0, 300, 120, 20)
    st.caption("Chunk size and overlap take effect when you build or rebuild the knowledge base.")

    st.divider()

    if st.button("Reset Knowledge Base", use_container_width=True):
        for key in [
            "vectorstore",
            "pages",
            "chunks",
            "file_names",
            "last_question",
            "last_answer",
            "last_evidence",
            "chunk_size_used",
            "overlap_used",
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

st.markdown(
    """
<div class="hero">
  <div class="hero-kicker">Aicademy 360 | Generative AI Engineering</div>
  <h1>RAG Knowledge Studio</h1>
  <p>
    Turn documents into a searchable knowledge base, inspect what the retriever found,
    generate grounded answers, and transform the same knowledge into study tools.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="pipeline">
  <div class="pipe-step">1. Documents</div>
  <div class="pipe-step">2. Chunks</div>
  <div class="pipe-step">3. Embeddings</div>
  <div class="pipe-step">4. Vector Search</div>
  <div class="pipe-step">5. Evidence</div>
  <div class="pipe-step">6. Grounded Answer</div>
</div>
""",
    unsafe_allow_html=True,
)

if not api_key:
    st.info("Add your Groq API key in the sidebar to activate the portal.")
    st.stop()


os.environ["GROQ_API_KEY"] = api_key

llm = ChatGroq(
    model=model_name,
    temperature=temperature,
)

with st.expander("Build or replace the knowledge base", expanded=st.session_state.vectorstore is None):
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True,
        help="For class, try the supplied Attention Is All You Need paper first.",
    )

    build_col, note_col = st.columns([1, 2])

    with build_col:
        build_clicked = st.button(
            "Build Knowledge Base",
            type="primary",
            use_container_width=True,
        )

    with note_col:
        st.caption(
            "The portal will load pages, create chunks, generate embeddings, and store them in Chroma."
        )

    if build_clicked:
        if not uploaded_files:
            st.error("Upload at least one PDF first.")
            st.stop()

        with st.status("Building the RAG knowledge base...", expanded=True) as status:
            st.write("Reading PDF pages")
            st.write("Splitting text into chunks")
            st.write("Creating embeddings")
            st.write("Indexing chunks in Chroma")

            vectorstore, pages, chunks = build_knowledge_base(
                uploaded_files,
                chunk_size,
                overlap,
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.pages = pages
            st.session_state.chunks = chunks
            st.session_state.file_names = [file.name for file in uploaded_files]
            st.session_state.last_question = ""
            st.session_state.last_answer = ""
            st.session_state.last_evidence = []
            st.session_state.chunk_size_used = chunk_size
            st.session_state.overlap_used = overlap

            status.update(label="Knowledge base ready", state="complete")

        st.rerun()

if st.session_state.vectorstore is None:
    st.warning("Build a knowledge base to unlock the workspace.")
    st.stop()

vectorstore = st.session_state.vectorstore
chunks = st.session_state.chunks
file_names = st.session_state.file_names

(
    dashboard_tab,
    ask_tab,
    search_tab,
    study_tab,
    compare_tab,
    lab_tab,
) = st.tabs(
    [
        "Dashboard",
        "Ask + Evidence",
        "Semantic Search",
        "Study Studio",
        "Compare Docs",
        "RAG Lab",
    ]
)

with dashboard_tab:
    c1, c2, c3, c4 = st.columns(4)

    cards = [
        (c1, "Documents", len(file_names), "Files inside this knowledge base"),
        (c2, "Pages", len(st.session_state.pages), "Pages read from PDFs"),
        (c3, "Chunks", len(chunks), "Searchable pieces of text"),
        (c4, "Top-K", top_k, "Evidence chunks per question"),
    ]

    for column, label, value, sub in cards:
        with column:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="card-label">{label}</div>
                    <div class="card-value">{value}</div>
                    <div class="card-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Knowledge Base")
    for name in file_names:
        pages_in_file = sum(
            1 for page in st.session_state.pages if page.metadata.get("source_file") == name
        )
        chunks_in_file = sum(
            1 for chunk in chunks if chunk.metadata.get("source_file") == name
        )

        st.markdown(
            f"""
            <div class="evidence-card">
                <span class="source-chip">PDF</span>
                <strong>{name}</strong><br>
                <span class="small-note">{pages_in_file} pages | {chunks_in_file} searchable chunks</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### What this portal demonstrates")
    st.write(
        "The interface is more advanced, but the AI pipeline is the same one students build in the simple version. "
        "This is the lesson: strong products grow from clear fundamentals."
    )

with ask_tab:
    st.markdown("### Ask the knowledge base")
    st.caption("The answer appears beside the evidence used to create it.")

    with st.form("qa_form"):
        question = st.text_area(
            "Question",
            height=90,
            placeholder="Example: Why does the Transformer allow more parallelization?",
        )
        ask_clicked = st.form_submit_button("Retrieve Evidence and Answer", type="primary")

    if ask_clicked:
        if not question.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Searching the knowledge base and grounding the answer..."):
                docs = retrieve_documents(vectorstore, question, search_mode, top_k)
                answer = run_grounded_answer(llm, question, docs, strict_mode)

            st.session_state.last_question = question
            st.session_state.last_answer = answer
            st.session_state.last_evidence = docs

    if st.session_state.last_answer:
        answer_col, evidence_col = st.columns([1.05, 0.95], gap="large")

        with answer_col:
            st.markdown("#### Grounded Answer")
            st.info(st.session_state.last_answer)

            report = (
                f"# RAG Research Note\n\n"
                f"## Question\n{st.session_state.last_question}\n\n"
                f"## Answer\n{st.session_state.last_answer}\n"
            )

            st.download_button(
                "Download Answer as Markdown",
                data=report,
                file_name="rag_answer.md",
                mime="text/markdown",
            )

        with evidence_col:
            st.markdown("#### Retrieved Evidence")

            for number, doc in enumerate(st.session_state.last_evidence, start=1):
                source = doc.metadata.get("source_file", "Unknown")
                page = doc.metadata.get("page", 0) + 1

                with st.expander(f"S{number} | {source} | page {page}", expanded=number <= 2):
                    st.write(doc.page_content)

with search_tab:
    st.markdown("### Search by meaning, without asking the LLM")
    st.caption(
        "This separates retrieval from generation. Students can see that vector search is useful even without a chatbot."
    )

    search_query = st.text_input(
        "Semantic search",
        placeholder="Example: architecture without recurrence or convolution",
        key="advanced_semantic_search",
    )

    if st.button("Run Semantic Search", key="advanced_search_button"):
        if not search_query.strip():
            st.warning("Type a search query first.")
        else:
            try:
                scored_results = vectorstore.similarity_search_with_relevance_scores(
                    search_query,
                    k=top_k,
                )

                for number, (doc, score) in enumerate(scored_results, start=1):
                    source = doc.metadata.get("source_file", "Unknown")
                    page = doc.metadata.get("page", 0) + 1

                    st.markdown(
                        f"""
                        <div class="evidence-card">
                            <span class="source-chip">Result {number}</span>
                            <span class="source-chip">{source}</span>
                            <span class="source-chip">Page {page}</span>
                            <span class="source-chip">Relevance {score:.2f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.write(doc.page_content)

            except Exception:
                # Some vector-store configurations may not expose normalized
                # relevance scores. The search itself can still work.
                results = vectorstore.similarity_search(search_query, k=top_k)

                for number, doc in enumerate(results, start=1):
                    source = doc.metadata.get("source_file", "Unknown")
                    page = doc.metadata.get("page", 0) + 1
                    st.markdown(f"**Result {number} | {source} | page {page}**")
                    st.write(doc.page_content)
                    st.divider()

with study_tab:
    st.markdown("### Turn the same knowledge base into learning tools")
    st.caption(
        "RAG is not only question answering. Retrieval can power many document intelligence products."
    )

    selected_file = st.selectbox(
        "Choose document",
        ["All documents"] + file_names,
        key="study_file",
    )

    tool_col1, tool_col2, tool_col3 = st.columns(3)
    summary_clicked = tool_col1.button("Create Smart Summary", use_container_width=True)
    quiz_clicked = tool_col2.button("Generate 5-Question Quiz", use_container_width=True)
    flash_clicked = tool_col3.button("Create Flashcards", use_container_width=True)

    if summary_clicked or quiz_clicked or flash_clicked:
        sample = document_sample(chunks, selected_file)

        if not sample:
            st.warning("No document text is available for this selection.")
        else:
            if summary_clicked:
                instruction = (
                    "Create a clear student-friendly summary. Use headings and 5 to 8 key points. "
                    "Do not add facts that are not in the supplied document text."
                )
                title = "Smart Summary"

            elif quiz_clicked:
                instruction = (
                    "Create exactly 5 useful questions from this document text. "
                    "After the questions, provide a separate answer key."
                )
                title = "Document Quiz"

            else:
                instruction = (
                    "Create 8 flashcards. Format every card as Question: ... and Answer: ... . "
                    "Keep the answers short and grounded in the text."
                )
                title = "Flashcards"

            prompt = f"""
You are an educational document assistant.
{instruction}

DOCUMENT TEXT:
{sample}
"""

            with st.spinner("Creating study material..."):
                output = llm.invoke(prompt).content

            st.markdown(f"#### {title}")
            st.write(output)

with compare_tab:
    st.markdown("### Compare two documents using retrieved evidence")

    if len(file_names) < 2:
        st.info("Upload at least two PDFs to unlock document comparison.")
    else:
        left, right = st.columns(2)
        with left:
            doc_a = st.selectbox("Document A", file_names, index=0)
        with right:
            doc_b = st.selectbox("Document B", file_names, index=1)

        compare_question = st.text_input(
            "What should we compare?",
            placeholder="Example: How do the two documents describe attention?",
        )

        if st.button("Compare Evidence", type="primary"):
            if not compare_question.strip():
                st.warning("Type a comparison question first.")
            elif doc_a == doc_b:
                st.warning("Choose two different documents.")
            else:
                docs_a = vectorstore.similarity_search(
                    compare_question,
                    k=3,
                    filter={"source_file": doc_a},
                )
                docs_b = vectorstore.similarity_search(
                    compare_question,
                    k=3,
                    filter={"source_file": doc_b},
                )

                context_a = create_context(docs_a, max_chars=6000)
                context_b = create_context(docs_b, max_chars=6000)

                prompt = f"""
Compare the two document evidence sets below for the user's question.
Only use the evidence provided.
Explain similarities and differences clearly.

QUESTION:
{compare_question}

DOCUMENT A: {doc_a}
{context_a}

DOCUMENT B: {doc_b}
{context_b}
"""

                comparison = llm.invoke(prompt).content
                st.success(comparison)

with lab_tab:
    st.markdown("### RAG Playground")
    st.write(
        "A good AI engineer does not only run RAG. They experiment with retrieval and inspect what changes."
    )

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Chunk Size Used", st.session_state.chunk_size_used)
    e2.metric("Overlap Used", st.session_state.overlap_used)
    e3.metric("Top-K", top_k)
    e4.metric("Search", "MMR" if search_mode.startswith("MMR") else "Similarity")

    experiment_query = st.text_input(
        "Experiment question",
        placeholder="Try one question and compare Similarity with MMR",
        key="experiment_query",
    )

    if st.button("Compare Retrieval Strategies"):
        if not experiment_query.strip():
            st.warning("Type an experiment question first.")
        else:
            sim_docs = vectorstore.similarity_search(experiment_query, k=top_k)
            mmr_docs = vectorstore.max_marginal_relevance_search(
                experiment_query,
                k=top_k,
                fetch_k=max(12, top_k * 4),
            )

            sim_col, mmr_col = st.columns(2, gap="large")

            with sim_col:
                st.markdown("#### Similarity")
                for i, doc in enumerate(sim_docs, start=1):
                    source = doc.metadata.get("source_file", "Unknown")
                    page = doc.metadata.get("page", 0) + 1
                    with st.expander(f"{i}. {source} | page {page}"):
                        st.write(doc.page_content[:900])

            with mmr_col:
                st.markdown("#### MMR")
                for i, doc in enumerate(mmr_docs, start=1):
                    source = doc.metadata.get("source_file", "Unknown")
                    page = doc.metadata.get("page", 0) + 1
                    with st.expander(f"{i}. {source} | page {page}"):
                        st.write(doc.page_content[:900])

    with st.expander("Advanced RAG roadmap"):
        st.markdown(
            """
- Metadata filtering: search only a chosen department, date, document, or category.
- Hybrid search: combine semantic vector search with keyword search.
- Reranking: retrieve a larger set, then use another model to reorder the best evidence.
- Better chunking: split by headings, sections, tables, or document structure.
- Evaluation: measure retrieval quality, answer faithfulness, and citation quality.
- Multimodal RAG: retrieve from images, charts, audio, and scanned documents.
- Agentic RAG: let an agent decide when, where, and how many times to search.
            """
        )

st.caption(
    "Aicademy 360 classroom showcase. The portal is advanced, but every feature still depends on the same RAG fundamentals."
)

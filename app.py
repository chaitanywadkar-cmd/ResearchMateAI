import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

    # Dashboard activity tracking
if "research_search_history" not in st.session_state:
    st.session_state["research_search_history"] = []

if "uploaded_pdf_files" not in st.session_state:
    st.session_state["uploaded_pdf_files"] = set()

if "uploaded_dataset_files" not in st.session_state:
    st.session_state["uploaded_dataset_files"] = set()

def display_research_history():
    st.divider()
    st.subheader("🕘 Research History")

    if st.session_state.chat_history:
        for index, chat in enumerate(
            reversed(st.session_state.chat_history), start=1
        ):
            with st.expander(f"Research {index}: {chat['question']}"):
                st.markdown("**Question:**")
                st.write(chat["question"])

                st.markdown("**AI Answer:**")
                st.write(chat["answer"])

                st.markdown("**Supporting Sources:**")
                for source in chat["sources"]:
                    st.write(source["passage"])
                    st.caption(
                        f"PDF Page: {source['page']} | "
                        f"Similarity: {source['score']:.3f}"
                    )

        st.download_button(
            label="⬇️ Download Research History as PDF",
            data=generate_pdf_report(st.session_state.chat_history),
            file_name="researchmate_report.pdf",
            mime="application/pdf",
            key="download_research_history_pdf",
        )

        if st.button("🗑️ Clear Research History", key="clear_research_history"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("No research history yet. Ask your first question!")


# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="ResearchMate AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: bold;
    color: #2563eb;
}

.subtitle {
    font-size: 17px;
    color: gray;
}

.stButton button {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="main-title">📚 ResearchMate AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Your Intelligent AI Research Assistant</p>',
    unsafe_allow_html=True
)

st.divider()

@st.cache_resource
def load_semantic_model():
    """Load the embedding model once and reuse it."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def split_into_chunks(text, max_words=180, overlap=40):
    """Split document text into overlapping chunks."""
    if max_words <= 0 or overlap < 0 or overlap >= max_words:
        raise ValueError("overlap must be non-negative and less than max_words")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


def create_page_chunks(pdf_bytes, max_words=180, overlap=40):
    """Split PDF text into overlapping chunks while retaining page numbers."""
    if max_words <= 0 or overlap < 0 or overlap >= max_words:
        raise ValueError("overlap must be non-negative and less than max_words")

    import fitz

    chunks = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            words = page.get_text("text").split()
            start = 0

            while start < len(words):
                end = min(start + max_words, len(words))
                chunk_text = " ".join(words[start:end])

                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "page": page_number,
                    })

                if end == len(words):
                    break

                start = end - overlap

    return chunks


def retrieve_relevant_passages(question, chunks, model, top_k=3):
    """Retrieve passages with the highest cosine similarity."""
    if not question.strip() or not chunks:
        return []
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    chunk_texts = [chunk["text"] for chunk in chunks]
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    chunk_embeddings = model.encode(
        chunk_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    scores = chunk_embeddings @ question_embedding
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "passage": chunks[int(index)]["text"],
            "page": chunks[int(index)]["page"],
            "score": float(scores[index]),
        }
        for index in ranked_indices
    ]


def display_research_qa(extracted_text, page_chunks):
    """Display the research question-answering interface."""
    st.divider()
    st.subheader("Research Question Answering")
    st.write(
        "Ask questions about your uploaded research paper. "
        "ResearchMate AI will retrieve the most relevant "
        "passages from the document."
    )

    if not extracted_text or not extracted_text.strip():
        st.warning(
            "No text is available. Upload a text-readable PDF first."
        )
        return

    question = st.text_input(
        "Enter your research question",
        placeholder="What are the main findings of this research?",
        key="research_qa_question",
    )
    top_k = st.slider(
        "Number of passages to retrieve",
        min_value=1,
        max_value=5,
        value=3,
        key="research_qa_top_k",
    )

    if st.button(
        "Find Research Answers",
        type="primary",
        key="research_qa_submit",
    ):
        if not question.strip():
            st.warning("Please enter a research question.")
            return

        try:
            with st.spinner("Searching the research paper..."):
                if not page_chunks:
                    st.warning("No usable text chunks were found.")
                    return

                model = load_semantic_model()
                results = retrieve_relevant_passages(
                    question=question,
                    chunks=page_chunks,
                    model=model,
                    top_k=top_k,
                )

                st.session_state["research_qa_results"] = results
                st.session_state["research_qa_last_question"] = question
                with st.spinner("Generating answer..."):
                    answer = generate_research_answer(
                        question=question,
                        passages=results,
                    )
                    st.markdown(answer)
                st.session_state["research_qa_answer"] = answer
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "sources": results,
                })
        except Exception as error:
            st.error(f"Research search failed: {error}")
            return

    results = st.session_state.get("research_qa_results", [])
    last_question = st.session_state.get("research_qa_last_question", "")

    if results:
        st.subheader("🤖 AI Research Answer")
        st.write(st.session_state.get("research_qa_answer", ""))
        st.write(f"*Question:* {last_question}")
        st.subheader("📚 Verified PDF Sources")

        for index, result in enumerate(results, start=1):
            with st.expander(f"Source {index}"):
                st.write(result["passage"])
                st.success(f"PDF Page Number: {result['page']}")
                st.caption(f"Similarity Score: {result['score']:.3f}")

def generate_research_answer(question, passages):
    """
    Generate an answer grounded in retrieved PDF passages.
    Uses a locally running Ollama model.
    """

    if not passages:
        return "No relevant passages were found in the document."

    context = "\n\n".join(
        f"Source {i + 1}:\n{item['passage']}"
        for i, item in enumerate(passages)
    )

    prompt = f"""
You are ResearchMate AI, a research assistant.

Answer the question using only the supplied research passages.

Rules:
1. Do not invent facts.
2. If the passages do not contain the answer, say so clearly.
3. Explain the answer in simple, clear language.
4. Mention the source numbers that support your answer.
5. Distinguish information stated in the paper from inference.

RESEARCH PASSAGES:
{context}

QUESTION:
{question}

ANSWER:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2
                }
            },
            timeout=180
        )

        response.raise_for_status()

        data = response.json()

        return data.get(
            "response",
            "The model returned an empty answer."
        ).strip()

    except requests.exceptions.ConnectionError:
        return (
            "Cannot connect to Ollama. "
            "Open Ollama and ensure the local service is running."
        )

    except requests.exceptions.Timeout:
        return (
            "The model took too long to respond. "
            "Try a shorter question or a smaller model."
        )

    except requests.exceptions.RequestException as error:
        return f"Answer generation failed: {error}"
def generate_pdf_report(chat_history):

    buffer = BytesIO()

    pdf = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()
    story = []

    # Project title
    story.append(
        Paragraph(
            "<b>ResearchMate AI - Research Report</b>",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 20))

    # Report generation date
    from datetime import datetime

    report_date = datetime.now().strftime("%d-%m-%Y %H:%M")

    story.append(
        Paragraph(f"<b>Generated on:</b> {report_date}", styles["Normal"])
    )

    story.append(Spacer(1, 20))

    # Add every research question and answer
    for index, chat in enumerate(chat_history, start=1):

        story.append(
            Paragraph(
                f"<b>Research Question {index}</b>",
                styles["Heading2"]
            )
        )

        story.append(
            Paragraph(
                chat["question"],
                styles["BodyText"]
            )
        )

        story.append(Spacer(1, 12))

        story.append(
            Paragraph("<b>AI Generated Answer</b>", styles["Heading3"])
        )

        story.append(
            Paragraph(
                chat["answer"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>"),
                styles["BodyText"]
            )
        )

        story.append(Spacer(1, 12))

        story.append(
            Paragraph("<b>Supporting PDF Sources</b>", styles["Heading3"])
        )

        for source_index, source in enumerate(
            chat["sources"], start=1
        ):

            passage = (
                source["passage"]
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

            story.append(
                Paragraph(
                    f"<b>Source {source_index} - Page {source['page']}</b>",
                    styles["Normal"]
                )
            )

            story.append(
                Paragraph(passage, styles["BodyText"])
            )

            story.append(
                Paragraph(
                    f"Similarity Score: {source['score']:.3f}",
                    styles["Italic"]
                )
            )

            story.append(Spacer(1, 10))

        story.append(Spacer(1, 20))

    # Build PDF
    pdf.build(story)

    buffer.seek(0)

    return buffer.getvalue()

display_research_history()

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 38px;
        font-weight: bold;
        color: #4F8BF9;
    }

    .subtitle {
        font-size: 17px;
        color: #808495;
    }

    .info-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #444444;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("🔬 ResearchMate AI")
st.sidebar.caption("Intelligent Research Assistant")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔍 AI Research",
        "📄 PDF Analysis",
        "📊 Data Analytics",
        "ℹ️ About Project"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Version 1.0 | Development Mode")


# -------------------------------
# Dashboard
# -------------------------------
if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">ResearchMate AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Your intelligent research and data analysis assistant.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    research_count = (
        len(st.session_state["research_search_history"])
        + len(st.session_state.chat_history)
    )

    pdf_count = len(st.session_state["uploaded_pdf_files"])

    dataset_count = len(st.session_state["uploaded_dataset_files"])

    with col1:
        st.metric("Research Sessions", research_count)

    with col2:
        st.metric("Documents Uploaded", pdf_count)

    with col3:
        st.metric("Datasets Analyzed", dataset_count)

    st.markdown("<div style='text-align: center;'><h3 style='margin-bottom: 0.5rem;'>🚀 Explore Your Workspace</h3></div>", unsafe_allow_html=True)

    workspace_cols = st.columns(2)

    with workspace_cols[0]:
        with st.container(border=True):
            st.subheader("🔍 AI Research")
            st.write(
                "Explore research topics and organize "
                "information from reliable sources."
            )
            st.caption("Status: Coming in the next development phase")

        with st.container(border=True):
            st.subheader("📊 Data Analytics")
            st.write(
                "Upload CSV or Excel datasets and "
                "discover useful patterns and insights."
            )
            st.caption("Status: Interface ready")

    with workspace_cols[1]:
        with st.container(border=True):
            st.subheader("📄 PDF Intelligence")
            st.write(
                "Upload research papers and prepare "
                "for document summarization and question answering."
            )
            st.caption("Status: Interface ready")

        with st.container(border=True):
            st.subheader("📝 Research Reports")
            st.write(
                "Prepare structured research reports "
                "with findings and source references."
            )
            st.caption("Status: Planned")

    st.info(
        "Choose a feature from the sidebar to explore the application."
    )

# -------------------------------
# AI Research - LIVE WEB SEARCH
# -------------------------------
elif page == "🔍 AI Research":

    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    st.title("🔍 AI Research Workspace")

    st.write(
        "Search the web for research topics and explore "
        "relevant information from online sources."
    )

    topic = st.text_input(
        "Research Topic",
        placeholder="Example: Applications of AI in healthcare",
        key="live_research_topic"
    )

    research_depth = st.selectbox(
        "Research Depth",
        [
            "Quick Overview",
            "Detailed Research",
            "Comprehensive Research"
        ],
        key="live_research_depth"
    )

    if research_depth == "Quick Overview":
        max_results = 5
    elif research_depth == "Detailed Research":
        max_results = 10
    else:
        max_results = 15

    if st.button("🔎 Search Research Topic", type="primary"):

        if not topic.strip():
            st.warning("Please enter a research topic.")

        else:
            try:
                with st.spinner("Searching the web..."):

                    with DDGS() as search_engine:
                        search_results = list(
                            search_engine.text(
                                topic.strip(),
                                max_results=max_results
                            )
                        )

                st.session_state["research_topic"] = topic.strip()
                st.session_state["research_depth"] = research_depth
                st.session_state["live_search_results"] = search_results
                
                # Save each successful research search
                st.session_state["research_search_history"].append({
                 "topic": topic.strip(),
                 "depth": research_depth,
                 "results_count": len(search_results)
                })

                st.success(
                    f"Found {len(search_results)} search results!"
                )

            except Exception as error:
                st.error(f"Web search failed: {error}")

    # Display saved search results
    if "live_search_results" in st.session_state:

        results = st.session_state["live_search_results"]

        st.divider()
        st.subheader("📚 Research Search Results")

        st.write(
            "**Research Topic:**",
            st.session_state.get("research_topic", "")
        )

        st.write(
            "**Research Depth:**",
            st.session_state.get("research_depth", "")
        )

        if results:

            for index, result in enumerate(results, start=1):

                title = result.get("title", "Untitled")
                link = result.get("href", "")
                description = result.get("body", "No description available.")

                with st.container(border=True):

                    st.markdown(f"### {index}. {title}")

                    st.write(description)

                    if link:
                        st.markdown(f"[🔗 Read Full Source]({link})")

        else:
            st.info(
                "No results found. Try another research topic."
            )

        st.caption(
            "Search results are collected from the web. "
            "Verify important information using the original sources."
        )

# -------------------------------
# PDF Analysis
# -------------------------------
elif page == "📄 PDF Analysis":

    import fitz
    import re

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    st.title("📄 PDF Intelligence")

    st.write(
        "Upload a research paper to extract text, "
        "generate summaries, and ask questions about the document."
    )

    # -------------------------------
    # PDF Upload
    # -------------------------------
    uploaded_pdf = st.file_uploader(
        "Upload your research PDF",
        type=["pdf"],
        key="research_pdf"
    )

    if uploaded_pdf is not None:
        pdf_key = f"{uploaded_pdf.name}:{uploaded_pdf.size}"
        st.session_state["uploaded_pdf_files"].add(pdf_key)

        try:
            # Extract PDF text
            pdf_bytes = uploaded_pdf.getvalue()

            page_chunks = create_page_chunks(pdf_bytes)

            with fitz.open(
                stream=pdf_bytes,
                filetype="pdf"
            ) as document:

                total_pages = len(document)

                extracted_pages = [
                    pdf_page.get_text()
                    for pdf_page in document
                ]

            extracted_text = "\n".join(extracted_pages)

            if not extracted_text.strip():

                st.warning(
                    "No selectable text was found. "
                    "This PDF may be scanned. OCR support "
                    "will be added in a future phase."
                )

            else:

                # -------------------------------
                # Document Statistics
                # -------------------------------
                words = extracted_text.split()
                characters = len(extracted_text)

                st.success("PDF processed successfully!")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Pages",
                    total_pages
                )

                col2.metric(
                    "Total Words",
                    f"{len(words):,}"
                )

                col3.metric(
                    "Characters",
                    f"{characters:,}"
                )

                # -------------------------------
                # Extracted Text
                # -------------------------------
                st.divider()

                st.subheader("📖 Extracted Text")

                with st.expander("View full document text"):

                    st.text_area(
                        "PDF Content",
                        value=extracted_text,
                        height=300
                    )

                st.download_button(
                    label="⬇️ Download Extracted Text",
                    data=extracted_text,
                    file_name="extracted_research.txt",
                    mime="text/plain",
                    key="download_extracted_text"
                )

                # -------------------------------
                # Prepare Sentences
                # -------------------------------
                sentences = re.split(
                    r'(?<=[.!?])\s+',
                    extracted_text
                )

                sentences = [
                    sentence.strip()
                    for sentence in sentences
                    if len(sentence.split()) >= 5
                ]

                # -------------------------------
                # Research Summarizer
                # -------------------------------
                st.divider()

                st.subheader("🧠 Research Summarizer")

                summary_length = st.selectbox(
                    "Choose summary length",
                    ["Short", "Medium", "Detailed"],
                    key="summary_length"
                )

                if st.button(
                    "Generate Summary",
                    type="primary",
                    key="generate_summary"
                ):

                    if len(sentences) < 2:

                        st.warning(
                            "The document contains too little "
                            "text to generate a useful summary."
                        )

                    else:

                        sentence_limits = {
                            "Short": 3,
                            "Medium": 6,
                            "Detailed": 10
                        }

                        number_of_sentences = min(
                            sentence_limits[summary_length],
                            len(sentences)
                        )

                        try:

                            vectorizer = TfidfVectorizer(
                                stop_words="english"
                            )

                            sentence_matrix = (
                                vectorizer.fit_transform(sentences)
                            )

                            sentence_scores = (
                                sentence_matrix.sum(axis=1).A1
                            )

                            ranked_indices = sorted(
                                range(len(sentences)),
                                key=lambda index: (
                                    sentence_scores[index]
                                ),
                                reverse=True
                            )

                            selected_indices = sorted(
                                ranked_indices[
                                    :number_of_sentences
                                ]
                            )

                            summary = " ".join(
                                sentences[index]
                                for index in selected_indices
                            )

                            st.session_state["pdf_summary"] = summary

                        except ValueError:

                            st.warning(
                                "Unable to summarize this document. "
                                "Try a longer PDF with more text."
                            )

                # Display the saved summary across reruns
                if st.session_state.get("pdf_summary"):

                    st.success("Summary is ready!")

                    st.subheader("📝 Research Summary")

                    st.write(
                        st.session_state["pdf_summary"]
                    )

                    st.download_button(
                        label="⬇️ Download Summary",
                        data=st.session_state["pdf_summary"],
                        file_name="research_summary.txt",
                        mime="text/plain",
                        key="download_pdf_summary"
                    )

                display_research_qa(extracted_text, page_chunks)

        except Exception as error:

            st.error(
                f"Unable to process the PDF: {error}"
            )

            st.caption(
                "Check that the file is a valid PDF and "
                "that the required libraries are installed."
            )
# -------------------------------
# Advanced Data Analytics
# -------------------------------
elif page == "📊 Data Analytics":

    st.title("📊 Advanced Data Analytics")
    st.write(
        "Upload, explore, clean and prepare your dataset "
        "for professional data analysis."
    )

    uploaded_data = st.file_uploader(
        "Upload CSV or Excel Dataset",
        type=["csv", "xlsx"],
        key="research_dataset"
    )

    if uploaded_data is not None:
        dataset_key = f"{uploaded_data.name}:{uploaded_data.size}"
        st.session_state["uploaded_dataset_files"].add(dataset_key)

        try:
            if uploaded_data.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_data)
            else:
                df = pd.read_excel(uploaded_data)

            st.success("Dataset uploaded successfully!")

            # Store original dataset
            st.session_state["original_df"] = df.copy()

            # Initialize cleaned dataset
            if "cleaned_df" not in st.session_state:
                st.session_state["cleaned_df"] = df.copy()

            st.subheader("📌 Dataset Overview")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Rows", len(df))
            col2.metric("Total Columns", len(df.columns))
            col3.metric(
                "Missing Values",
                int(df.isna().sum().sum())
            )
            col4.metric(
                "Duplicate Rows",
                int(df.duplicated().sum())
            )

            st.divider()

            st.subheader("👀 Dataset Preview")

            st.dataframe(
                st.session_state["cleaned_df"].head(20),
                use_container_width=True
            )

            # Missing value analysis
            st.divider()
            st.subheader("🔍 Missing Value Analysis")

            missing_data = pd.DataFrame({
                "Column": df.columns,
                "Missing Values": df.isna().sum().values,
                "Missing Percentage": (
                    df.isna().mean().values * 100
                ).round(2)
            })

            st.dataframe(
                missing_data,
                use_container_width=True
            )

            # Duplicate analysis
            st.divider()
            st.subheader("🔁 Duplicate Analysis")

            duplicate_count = int(
                st.session_state["cleaned_df"].duplicated().sum()
            )

            st.metric("Duplicate Records", duplicate_count)

            # Data Cleaning
            st.divider()
            st.subheader("🧹 Data Cleaning")

            cleaned_df = st.session_state["cleaned_df"]

            if st.button("Remove Duplicate Rows"):

                cleaned_df = cleaned_df.drop_duplicates()

                st.session_state["cleaned_df"] = cleaned_df

                st.success("Duplicate rows removed!")
                st.rerun()

            st.markdown("### Fill Missing Values")

            numeric_columns = cleaned_df.select_dtypes(
                include=np.number
            ).columns.tolist()

            if numeric_columns:

                selected_column = st.selectbox(
                    "Select Numerical Column",
                    numeric_columns
                )

                fill_method = st.selectbox(
                    "Select Filling Method",
                    ["Mean", "Median", "Zero"]
                )

                if st.button("Fill Missing Values"):

                    if fill_method == "Mean":
                        fill_value = cleaned_df[
                            selected_column
                        ].mean()

                    elif fill_method == "Median":
                        fill_value = cleaned_df[
                            selected_column
                        ].median()

                    else:
                        fill_value = 0

                    cleaned_df[selected_column] = (
                        cleaned_df[selected_column].fillna(
                            fill_value
                        )
                    )

                    st.session_state["cleaned_df"] = cleaned_df

                    st.success(
                        f"Missing values filled using {fill_method}!"
                    )
                    st.rerun()

            else:
                st.info(
                    "No numerical columns available for filling."
                )

            # Download cleaned dataset
            st.divider()
            st.subheader("📥 Download Cleaned Dataset")

            csv_data = st.session_state[
                "cleaned_df"
            ].to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Cleaned CSV",
                data=csv_data,
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )

            st.success(
                "Your data cleaning workspace is ready!"
            )
            # -------------------------------
            # ADVANCED EXPLORATORY DATA ANALYSIS
            # -------------------------------

            st.divider()
            st.title("📈 Exploratory Data Analysis")

            analysis_df = st.session_state["cleaned_df"]

            # 1. Descriptive Statistics
            st.subheader("📊 Statistical Summary")

            st.dataframe(
                analysis_df.describe().T,
                use_container_width=True
            )

            # 2. Column Selection
            numeric_columns = analysis_df.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_columns = analysis_df.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            if numeric_columns:

                st.divider()
                st.subheader("📉 Numerical Analysis")

                selected_numeric = st.selectbox(
                    "Select Numerical Column",
                    numeric_columns,
                    key="eda_numeric"
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Mean",
                    f"{analysis_df[selected_numeric].mean():.2f}"
                )

                col2.metric(
                    "Median",
                    f"{analysis_df[selected_numeric].median():.2f}"
                )

                col3.metric(
                    "Standard Deviation",
                    f"{analysis_df[selected_numeric].std():.2f}"
                )

                # Histogram
                import plotly.express as px

                fig = px.histogram(
                    analysis_df,
                    x=selected_numeric,
                    title=f"Distribution of {selected_numeric}",
                    marginal="box"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # 3. Outlier Detection using IQR

                st.subheader("🚨 Outlier Detection")

                Q1 = analysis_df[selected_numeric].quantile(0.25)
                Q3 = analysis_df[selected_numeric].quantile(0.75)

                IQR = Q3 - Q1

                lower_limit = Q1 - 1.5 * IQR
                upper_limit = Q3 + 1.5 * IQR

                outliers = analysis_df[
                    (analysis_df[selected_numeric] < lower_limit) |
                    (analysis_df[selected_numeric] > upper_limit)
                ]

                c1, c2, c3 = st.columns(3)

                c1.metric("Outlier Count", len(outliers))
                c2.metric("Lower Limit", f"{lower_limit:.2f}")
                c3.metric("Upper Limit", f"{upper_limit:.2f}")

                st.dataframe(
                    outliers.head(20),
                    use_container_width=True
                )

                # 4. Correlation Analysis

                st.divider()
                st.subheader("🔗 Correlation Analysis")

                if len(numeric_columns) >= 2:

                    correlation = analysis_df[
                        numeric_columns
                    ].corr()

                    fig_corr = px.imshow(
                        correlation,
                        text_auto=True,
                        color_continuous_scale="RdBu_r",
                        title="Correlation Heatmap",
                        aspect="auto"
                    )

                    st.plotly_chart(
                        fig_corr,
                        use_container_width=True
                    )

                else:
                    st.info(
                        "At least two numerical columns "
                        "are required for correlation analysis."
                    )

            # 5. Categorical Analysis

            if categorical_columns:

                st.divider()
                st.subheader("📋 Categorical Analysis")

                selected_category = st.selectbox(
                    "Select Categorical Column",
                    categorical_columns,
                    key="eda_category"
                )

                category_counts = (
                    analysis_df[selected_category]
                    .fillna("Missing")
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )

                category_counts.columns = [
                    "Category",
                    "Count"
                ]

                st.dataframe(
                    category_counts,
                    use_container_width=True
                )

                fig_category = px.bar(
                    category_counts.head(15),
                    x="Category",
                    y="Count",
                    title=f"Frequency of {selected_category}"
                )

                st.plotly_chart(
                    fig_category,
                    use_container_width=True
                )

            st.success(
                "Exploratory Data Analysis completed!"
            )
            # ---------------------------------
            # STEP 11C: INTERACTIVE VISUALIZATION
            # ---------------------------------

            import plotly.express as px

            st.divider()
            st.title("📊 Interactive Data Visualization")

            visual_df = st.session_state["cleaned_df"].copy()

            numeric_cols = visual_df.select_dtypes(
                include=np.number
            ).columns.tolist()

            category_cols = visual_df.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            # 1. Dynamic Filters
            st.subheader("🔎 Dynamic Dataset Filters")

            filtered_df = visual_df.copy()

            with st.expander("Apply Filters", expanded=True):

                for index, column in enumerate(category_cols):

                    filter_values = (
                        visual_df[column]
                        .fillna("Missing")
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    selected_values = st.multiselect(
                        f"Filter by {column}",
                        options=filter_values,
                        key=f"visual_filter_{index}"
                    )

                    if selected_values:
                        filtered_df = filtered_df[
                            filtered_df[column]
                            .fillna("Missing")
                            .astype(str)
                            .isin(selected_values)
                        ]

            st.metric(
                "Filtered Records",
                len(filtered_df)
            )

            st.dataframe(
                filtered_df.head(20),
                use_container_width=True
            )

            if not filtered_df.empty:

                # 2. Scatter Plot

                st.divider()
                st.subheader("🔵 Scatter Plot")

                if len(numeric_cols) >= 2:

                    col1, col2 = st.columns(2)

                    with col1:
                        scatter_x = st.selectbox(
                            "Select X Axis",
                            numeric_cols,
                            key="scatter_x"
                        )

                    with col2:
                        scatter_y = st.selectbox(
                            "Select Y Axis",
                            numeric_cols,
                            index=1,
                            key="scatter_y"
                        )

                    fig_scatter = px.scatter(
                        filtered_df,
                        x=scatter_x,
                        y=scatter_y,
                        title="Relationship Between Variables",
                        hover_data=category_cols[:2]
                    )

                    st.plotly_chart(
                        fig_scatter,
                        use_container_width=True
                    )

                else:
                    st.info(
                        "Two numerical columns are required "
                        "for scatter plots."
                    )

                # 3. Box Plot

                st.divider()
                st.subheader("📦 Box Plot")

                if numeric_cols:

                    box_column = st.selectbox(
                        "Select Numerical Column",
                        numeric_cols,
                        key="box_column"
                    )

                    box_group = st.selectbox(
                        "Group By (Optional)",
                        ["None"] + category_cols,
                        key="box_group"
                    )

                    fig_box = px.box(
                        filtered_df,
                        y=box_column,
                        color=(
                            box_group
                            if box_group != "None"
                            else None
                        ),
                        title=f"Box Plot of {box_column}",
                        points="outliers"
                    )

                    st.plotly_chart(
                        fig_box,
                        use_container_width=True
                    )

                # 4. Pie Chart

                st.divider()
                st.subheader("🥧 Category Distribution")

                if category_cols:

                    pie_column = st.selectbox(
                        "Select Category",
                        category_cols,
                        key="pie_column"
                    )

                    pie_data = (
                        filtered_df[pie_column]
                        .fillna("Missing")
                        .astype(str)
                        .value_counts()
                        .reset_index()
                    )

                    pie_data.columns = [
                        "Category",
                        "Count"
                    ]

                    fig_pie = px.pie(
                        pie_data,
                        names="Category",
                        values="Count",
                        title=f"Distribution of {pie_column}",
                        hole=0.35
                    )

                    st.plotly_chart(
                        fig_pie,
                        use_container_width=True
                    )

                # 5. Customizable Bar Chart

                st.divider()
                st.subheader("📊 Customizable Bar Chart")

                if category_cols and numeric_cols:

                    col1, col2 = st.columns(2)

                    with col1:
                        bar_category = st.selectbox(
                            "Select Category",
                            category_cols,
                            key="bar_category"
                        )

                    with col2:
                        bar_value = st.selectbox(
                            "Select Numerical Value",
                            numeric_cols,
                            key="bar_value"
                        )

                    aggregation = st.selectbox(
                        "Aggregation Method",
                        ["Sum", "Mean", "Count"],
                        key="bar_aggregation"
                    )

                    grouped_df = filtered_df.groupby(
                        bar_category,
                        dropna=False
                    )

                    if aggregation == "Sum":
                        bar_data = grouped_df[
                            bar_value
                        ].sum().reset_index()

                    elif aggregation == "Mean":
                        bar_data = grouped_df[
                            bar_value
                        ].mean().reset_index()

                    else:
                        bar_data = grouped_df.size().reset_index(
                            name="Record Count"
                        )

                    if aggregation == "Count":
                        y_axis = "Record Count"
                    else:
                        y_axis = bar_value

                    bar_data[bar_category] = (
                        bar_data[bar_category]
                        .fillna("Missing")
                        .astype(str)
                    )

                    fig_bar = px.bar(
                        bar_data,
                        x=bar_category,
                        y=y_axis,
                        title=f"{aggregation} of {bar_value}"
                    )

                    st.plotly_chart(
                        fig_bar,
                        use_container_width=True
                    )

                # 6. Download Filtered Dataset

                st.divider()
                st.subheader("📥 Export Filtered Dataset")

                filtered_csv = filtered_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Filtered CSV",
                    data=filtered_csv,
                    file_name="filtered_dataset.csv",
                    mime="text/csv",
                    key="download_filtered_data"
                )

            else:
                st.warning(
                    "No records match the selected filters. "
                    "Please change your filter selections."
                )

            st.success(
                "Interactive visualization module completed!"
            )
            # ---------------------------------
            # STEP 11D: ADVANCED DATA SCIENCE
            # ---------------------------------

            from scipy import stats
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import (
                mean_absolute_error,
                mean_squared_error,
                r2_score
            )

            st.divider()
            st.title("🧪 Advanced Data Science Laboratory")

            ds_df = st.session_state["cleaned_df"].copy()

            numeric_cols = ds_df.select_dtypes(
                include=np.number
            ).columns.tolist()

            category_cols = ds_df.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            # 1. Hypothesis Testing

            st.subheader("🔬 Statistical Hypothesis Testing")
            st.write(
                "Perform an independent two-sample t-test "
                "to compare two groups."
            )

            if numeric_cols and category_cols:

                test_column = st.selectbox(
                    "Select Numerical Variable",
                    numeric_cols,
                    key="test_numeric"
                )

                test_group = st.selectbox(
                    "Select Group Column",
                    category_cols,
                    key="test_group"
                )

                groups = (
                    ds_df[test_group]
                    .dropna()
                    .unique()
                    .tolist()
                )

                if len(groups) >= 2:

                    group_a = st.selectbox(
                        "First Group",
                        groups,
                        key="group_a"
                    )

                    group_b_options = [
                        value for value in groups
                        if value != group_a
                    ]

                    group_b = st.selectbox(
                        "Second Group",
                        group_b_options,
                        key="group_b"
                    )

                    if st.button(
                        "Run T-Test",
                        key="run_ttest"
                    ):

                        a = ds_df.loc[
                            ds_df[test_group] == group_a,
                            test_column
                        ].dropna()

                        b = ds_df.loc[
                            ds_df[test_group] == group_b,
                            test_column
                        ].dropna()

                        if len(a) >= 2 and len(b) >= 2:

                            result = stats.ttest_ind(
                                a, b, equal_var=False
                            )

                            st.metric(
                                "T-Test P-Value",
                                f"{result.pvalue:.6f}"
                            )

                            if result.pvalue < 0.05:
                                st.success(
                                    "Statistically significant "
                                    "difference detected at 5% level."
                                )
                            else:
                                st.info(
                                    "Insufficient evidence of a "
                                    "significant difference at 5% level."
                                )

                            st.write(
                                "T-Statistic:",
                                round(result.statistic, 4)
                            )

                        else:
                            st.warning(
                                "Each group requires at least "
                                "two valid numerical observations."
                            )

                else:
                    st.info(
                        "At least two groups are required."
                    )

            # 2. Pearson Correlation

            st.divider()
            st.subheader("🔗 Pearson Correlation Testing")

            if len(numeric_cols) >= 2:

                corr_col1 = st.selectbox(
                    "First Variable",
                    numeric_cols,
                    key="pearson_col1"
                )

                corr_col2_options = [
                    col for col in numeric_cols
                    if col != corr_col1
                ]

                corr_col2 = st.selectbox(
                    "Second Variable",
                    corr_col2_options,
                    key="pearson_col2"
                )

                if st.button(
                    "Calculate Correlation",
                    key="calculate_pearson"
                ):

                    pair_data = ds_df[
                        [corr_col1, corr_col2]
                    ].dropna()

                    if len(pair_data) >= 3:

                        result = stats.pearsonr(
                            pair_data[corr_col1],
                            pair_data[corr_col2]
                        )

                        st.metric(
                            "Pearson Correlation",
                            f"{result.statistic:.4f}"
                        )

                        st.metric(
                            "Correlation P-Value",
                            f"{result.pvalue:.6f}"
                        )

                        st.write(
                            "Correlation ranges from -1 to +1."
                        )

                    else:
                        st.warning(
                            "At least three valid paired observations "
                            "are required."
                        )

            # 3. Feature Engineering

            st.divider()
            st.subheader("⚙️ Feature Engineering")

            if len(numeric_cols) >= 2:

                feature_col1 = st.selectbox(
                    "First Numerical Feature",
                    numeric_cols,
                    key="feature_col1"
                )

                feature_col2 = st.selectbox(
                    "Second Numerical Feature",
                    [
                        col for col in numeric_cols
                        if col != feature_col1
                    ],
                    key="feature_col2"
                )

                feature_operation = st.selectbox(
                    "Choose Operation",
                    [
                        "Addition",
                        "Subtraction",
                        "Multiplication",
                        "Division"
                    ],
                    key="feature_operation"
                )

                if st.button(
                    "Create New Feature",
                    key="create_feature"
                ):

                    new_feature = "engineered_feature"

                    if feature_operation == "Addition":
                        ds_df[new_feature] = (
                            ds_df[feature_col1] +
                            ds_df[feature_col2]
                        )

                    elif feature_operation == "Subtraction":
                        ds_df[new_feature] = (
                            ds_df[feature_col1] -
                            ds_df[feature_col2]
                        )

                    elif feature_operation == "Multiplication":
                        ds_df[new_feature] = (
                            ds_df[feature_col1] *
                            ds_df[feature_col2]
                        )

                    else:
                        ds_df[new_feature] = (
                            ds_df[feature_col1] /
                            ds_df[feature_col2].replace(0, np.nan)
                        )

                    st.session_state["engineered_df"] = (
                        ds_df.copy()
                    )

                    st.success(
                        "New feature created successfully!"
                    )

                    st.dataframe(
                        ds_df.head(10),
                        use_container_width=True
                    )

            # 4. Linear Regression

            st.divider()
            st.subheader("🤖 Linear Regression Model")

            st.write(
                "Train a machine learning model to predict "
                "a numerical target variable."
            )

            model_df = st.session_state.get(
                "engineered_df",
                st.session_state["cleaned_df"]
            ).copy()

            model_numeric_cols = model_df.select_dtypes(
                include=np.number
            ).columns.tolist()

            if len(model_numeric_cols) >= 2:

                target = st.selectbox(
                    "Select Target Variable",
                    model_numeric_cols,
                    key="model_target"
                )

                predictor_options = [
                    col for col in model_numeric_cols
                    if col != target
                ]

                predictors = st.multiselect(
                    "Select Input Features",
                    predictor_options,
                    default=predictor_options[:min(
                        3, len(predictor_options)
                    )],
                    key="model_predictors"
                )

                test_size = st.slider(
                    "Test Data Percentage",
                    min_value=10,
                    max_value=40,
                    value=20,
                    step=5,
                    key="model_test_size"
                )

                if st.button(
                    "Train Regression Model",
                    type="primary",
                    key="train_regression"
                ):

                    if not predictors:
                        st.warning(
                            "Select at least one input feature."
                        )

                    else:

                        model_data = model_df[
                            predictors + [target]
                        ].replace(
                            [np.inf, -np.inf],
                            np.nan
                        ).dropna()

                        if len(model_data) < 10:

                            st.warning(
                                "At least 10 complete records "
                                "are required for training."
                            )

                        else:

                            X = model_data[predictors]
                            y = model_data[target]

                            X_train, X_test, y_train, y_test = (
                                train_test_split(
                                    X,
                                    y,
                                    test_size=test_size / 100,
                                    random_state=42
                                )
                            )

                            model = LinearRegression()

                            model.fit(X_train, y_train)

                            predictions = model.predict(X_test)

                            r2 = r2_score(
                                y_test,
                                predictions
                            )

                            mae = mean_absolute_error(
                                y_test,
                                predictions
                            )

                            rmse = np.sqrt(
                                mean_squared_error(
                                    y_test,
                                    predictions
                                )
                            )

                            col1, col2, col3 = st.columns(3)

                            col1.metric(
                                "R² Score",
                                f"{r2:.4f}"
                            )

                            col2.metric(
                                "MAE",
                                f"{mae:.4f}"
                            )

                            col3.metric(
                                "RMSE",
                                f"{rmse:.4f}"
                            )

                            st.success(
                                "Regression model trained successfully!"
                            )

                            # Actual vs Predicted

                            prediction_df = pd.DataFrame({
                                "Actual": y_test.values,
                                "Predicted": predictions
                            })

                            fig_prediction = px.scatter(
                                prediction_df,
                                x="Actual",
                                y="Predicted",
                                title="Actual vs Predicted Values"
                            )

                            st.plotly_chart(
                                fig_prediction,
                                use_container_width=True
                            )

                            # Coefficients

                            coefficient_df = pd.DataFrame({
                                "Feature": predictors,
                                "Coefficient": model.coef_
                            })

                            st.subheader(
                                "Model Feature Coefficients"
                            )

                            st.dataframe(
                                coefficient_df,
                                use_container_width=True
                            )

            else:
                st.info(
                    "At least two numerical columns are "
                    "required for regression."
                )

            st.success(
                "Advanced Data Science module completed!"
            )
            # ---------------------------------
            # STEP 12A: ML CLASSIFICATION MODELS
            # ---------------------------------

            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import (
                accuracy_score,
                classification_report,
                confusion_matrix
            )

            st.divider()
            st.title("🤖 Machine Learning Classification Laboratory")

            ml_df = st.session_state["cleaned_df"].copy()

            st.write(
                "Train classification models to predict categorical outcomes."
            )

            # Select target
            target_candidates = [
                col for col in ml_df.columns
                if ml_df[col].nunique(dropna=True) >= 2
                and ml_df[col].nunique(dropna=True) <= 20
            ]

            if target_candidates:

                target_col = st.selectbox(
                    "Select Target Column",
                    target_candidates,
                    key="classification_target"
                )

                if st.button("Prepare Classification Dataset"):

                    features = ml_df.drop(columns=[target_col])
                    target = ml_df[target_col]

                    valid_rows = target.notna()
                    features = features.loc[valid_rows]
                    target = target.loc[valid_rows]

                    if target.nunique() < 2:
                        st.error("Target must contain at least two classes.")

                    elif len(target) < 20:
                        st.warning(
                            "Please use at least 20 valid records."
                        )

                    else:
                        X = features.select_dtypes(
                            include=np.number
                        )
                        categorical_features = features.select_dtypes(
                            exclude=np.number
                        )

                        X = pd.concat(
                            [X, categorical_features],
                            axis=1
                        )

                        X = X.replace(
                            [np.inf, -np.inf],
                            np.nan
                        )
                        
                        label_encoder = LabelEncoder()
                        y = LabelEncoder().fit_transform(target.astype(str))

                        numeric_columns = X.select_dtypes(
                            include=np.number
                        ).columns.tolist()

                        categorical_columns = X.select_dtypes(
                            exclude=np.number
                        ).columns.tolist()

                        numeric_pipeline = Pipeline([
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler())
                        ])

                        categorical_pipeline = Pipeline([
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore"))
                        ])

                        preprocessing = ColumnTransformer([
                            ("numeric", numeric_pipeline, numeric_columns),
                            ("categorical", categorical_pipeline, categorical_columns)
                        ])

                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y,
                            test_size=0.2,
                            random_state=42,
                            stratify=y
                        )

                        models = {
                            "Logistic Regression": LogisticRegression(
                                max_iter=1000
                            ),
                            "Decision Tree": DecisionTreeClassifier(
                                random_state=42
                            ),
                            "Random Forest": RandomForestClassifier(
                                n_estimators=100,
                                random_state=42
                            )
                        }
                
               # STEP 12B: MODEL EVALUATION

                        from sklearn.metrics import (
                            precision_score,
                            recall_score,
                            f1_score,
                            confusion_matrix,
                            classification_report
                        )

                        results = []
                        evaluation_models = {}
                        fitted_models = {}

                        class_names = sorted(
                            target.astype(str).unique().tolist()
                        )

                        for model_name, classifier in models.items():
                            
                            pipeline = Pipeline([
                                ("preprocessing", preprocessing),
                                ("classifier", classifier)
                            ])

                            pipeline.fit(X_train, y_train)

                            fitted_models[model_name] = pipeline

                            predictions = pipeline.predict(X_test)

                            accuracy = accuracy_score(
                                y_test, predictions
                            )

                            precision = precision_score(
                                y_test,
                                predictions,
                                average="weighted",
                                zero_division=0
                            )

                            recall = recall_score(
                                y_test,
                                predictions,
                                average="weighted",
                                zero_division=0
                            )

                            f1 = f1_score(
                                y_test,
                                predictions,
                                average="weighted",
                                zero_division=0
                            )

                            results.append({
                                "Model": model_name,
                                "Accuracy": round(accuracy * 100, 2),
                                "Precision": round(precision * 100, 2),
                                "Recall": round(recall * 100, 2),
                                "F1 Score": round(f1 * 100, 2)
                            })

                            evaluation_models[model_name] = {
                                "predictions": predictions,
                                "confusion_matrix": confusion_matrix(
                                    y_test, predictions
                                ),
                                "report": classification_report(
                                    y_test,
                                    predictions,
                                    output_dict=True,
                                    zero_division=0
                                )
                            }

                        # Create model comparison DataFrame
                        results_df = pd.DataFrame(results)

                        st.session_state["classification_results"] = results_df
                        st.session_state["classification_evaluations"] = evaluation_models
                        st.session_state["classification_class_names"] = class_names

                        # Save trained models and prediction information
                        st.session_state["classification_models"] = fitted_models
                        st.session_state["classification_label_encoder"] = label_encoder
                        st.session_state["classification_feature_columns"] = X.columns.tolist()
                        st.session_state["classification_numeric_columns"] = numeric_columns
                        st.session_state["classification_categorical_columns"] = categorical_columns
                        st.session_state["classification_training_data"] = X.copy()

                        st.success("All models evaluated successfully!")

                        st.subheader("📊 Model Performance Comparison")

                        st.dataframe(
                            results_df,
                            use_container_width=True
                        )

                        st.bar_chart(
                            results_df.set_index("Model")
                        )

                        best_model = results_df.loc[
                            results_df["Accuracy"].idxmax()
                        ]

                        st.metric(
                            "Best Performing Model",
                            best_model["Model"],
                            f'{best_model["Accuracy"]}% Accuracy'
                        )

                        # Detailed evaluation

                        st.divider()
                        st.subheader("🔍 Detailed Model Evaluation")

                        selected_model = st.selectbox(
                            "Select Model for Detailed Evaluation",
                            list(evaluation_models.keys()),
                            key="evaluation_model"
                        )

                        selected_evaluation = evaluation_models[
                            selected_model
                        ]

                        st.write("**Confusion Matrix**")

                        cm = selected_evaluation["confusion_matrix"]

                        cm_df = pd.DataFrame(
                            cm,
                            index=[
                                f"Actual {i}" for i in range(len(cm))
                            ],
                            columns=[
                                f"Predicted {i}" for i in range(len(cm))
                            ]
                        )

                        st.dataframe(
                            cm_df,
                            use_container_width=True
                        )

                        st.write("**Classification Report**")

                        report_df = pd.DataFrame(
                            selected_evaluation["report"]
                        ).transpose()

                        st.dataframe(
                            report_df.round(3),
                            use_container_width=True
                        )

            # ---------------------------------
            # STEP 12C: ML PREDICTION SYSTEM
            # ---------------------------------

            st.divider()
            st.title("🔮 Machine Learning Prediction System")

            if "classification_models" in st.session_state:

                st.write(
                    "Enter new record details to predict the outcome "
                    "using your trained machine learning model."
                )

                # Load saved model information
                trained_models = st.session_state["classification_models"]
                results_df = st.session_state["classification_results"]
                label_encoder = st.session_state["classification_label_encoder"]
                feature_columns = st.session_state["classification_feature_columns"]
                numeric_columns = st.session_state["classification_numeric_columns"]
                categorical_columns = st.session_state["classification_categorical_columns"]
                training_data = st.session_state["classification_training_data"]

                # Identify the best model
                best_model_name = results_df.loc[
                    results_df["Accuracy"].idxmax(), "Model"
                ]

                st.success(f"Best Model: {best_model_name}")

                model_names = list(trained_models.keys())

                selected_prediction_model = st.selectbox(
                    "Select Model for Prediction",
                    model_names,
                    index=model_names.index(best_model_name),
                    key="prediction_model_choice"
                )

                prediction_pipeline = trained_models[
                    selected_prediction_model
                ]

                st.subheader("📝 Enter New Record Details")

                with st.form("prediction_form"):

                    input_values = {}

                    st.markdown("**Numerical Features**")

                    for column in numeric_columns:

                        default_value = training_data[column].median()

                        if pd.isna(default_value):
                            default_value = 0.0

                        input_values[column] = st.number_input(
                            f"Enter {column}",
                            value=float(default_value),
                            key=f"prediction_num_{column}"
                        )

                    st.markdown("**Categorical Features**")

                    for column in categorical_columns:

                        options = sorted(
                            training_data[column]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )

                        if not options:
                            options = ["Unknown"]

                        input_values[column] = st.selectbox(
                            f"Select {column}",
                            options,
                            key=f"prediction_cat_{column}"
                        )

                    predict_button = st.form_submit_button(
                        "🚀 Predict Outcome"
                    )

                if predict_button:

                    new_record = pd.DataFrame(
                        [input_values],
                        columns=feature_columns
                    )

                    predicted_value = prediction_pipeline.predict(
                        new_record
                    )[0]

                    predicted_label = label_encoder.inverse_transform(
                        [int(predicted_value)]
                    )[0]

                    prediction_result = pd.DataFrame({
                        "Selected Model": [selected_prediction_model],
                        "Predicted Outcome": [predicted_label]
                    })

                    st.session_state["prediction_results"] = prediction_result

                    st.success("Prediction completed successfully!")

                    st.metric(
                        "Predicted Outcome",
                        str(predicted_label)
                    )

                    st.dataframe(
                        prediction_result,
                        use_container_width=True
                    )

                    # Prediction probability
                    if hasattr(prediction_pipeline, "predict_proba"):

                        probabilities = prediction_pipeline.predict_proba(
                            new_record
                        )[0]

                        probability_df = pd.DataFrame({
                            "Class": label_encoder.classes_,
                            "Probability (%)": probabilities * 100
                        })

                        st.subheader("📊 Prediction Probability")

                        st.dataframe(
                            probability_df.round(2),
                            use_container_width=True
                        )

                # Download prediction
                if "prediction_results" in st.session_state:

                    st.download_button(
                        label="⬇️ Download Prediction as CSV",
                        data=st.session_state["prediction_results"].to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="researchmate_prediction.csv",
                        mime="text/csv",
                        key="download_prediction_csv"
                    )

            else:
                st.info(
                    "First prepare the classification dataset and train "
                    "the models in Step 12A and Step 12B."
                    "No suitable target column found. "
                    "Choose a dataset with a categorical column "
                    "containing at least two classes."
                )

        except Exception as error:

            st.error(f"Dataset processing failed: {error}")

# -------------------------------
# About Project
# -------------------------------
elif page == "ℹ️ About Project":

    st.title("ℹ️ About ResearchMate AI")

    st.write("""
    **ResearchMate AI** is an educational AI application
    designed to support research, document understanding,
    and data analysis.

    ### Planned Capabilities

    - Intelligent research assistance
    - PDF text extraction and summarization
    - Document-based question answering
    - Source references and citations
    - CSV and Excel data analysis
    - Interactive visualizations
    - Downloadable research reports

    ### Technology Stack

    ResearchMate AI is built with a modern Python data science and AI stack,
    including:

    - Python for application logic and automation
    - Streamlit for the interactive web dashboard and UI
    - Pandas and NumPy for dataset processing and analysis
    - Plotly for charts and interactive visualizations
    - PyMuPDF (fitz) for PDF text extraction and document processing
    - scikit-learn for summary scoring, correlation analysis, and ML workflows
    - sentence-transformers for semantic search and embeddings
    - DuckDuckGo Search for web research retrieval
    - ReportLab for generating downloadable PDF reports
    - Ollama + local LLMs for research question answering
    - Git and GitHub for version control and project collaboration

    This project is under active development.
    """)

    st.info(
        "Our next goal is to implement real PDF text extraction."
    )
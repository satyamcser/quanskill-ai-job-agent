import sys
import os
import streamlit as st
import tempfile
import pandas as pd

# ✅ Add project root for imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from modules.resume_parser import ResumeParser
from modules.job_scraper import JobScraper
from modules.vector_store import Vectorizer
from modules.matcher import JobMatcher
from modules.qa_agent import LocalLLM

# --- Streamlit UI ---
st.set_page_config(page_title="AI Job Agent", layout="wide")
st.title("🤖 AI Job Agent")
st.write("Upload your resume, search for jobs, and interactively ask questions about them!")

# --- Sidebar Inputs ---
st.sidebar.header("Job Search Settings")
job_query = st.sidebar.text_input("Job search query", value="AI Engineer")
job_location = st.sidebar.text_input("Location", value="Remote")
num_results = st.sidebar.slider("Number of jobs to fetch", 1, 20, 5)

# ✅ Session state variables
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None
if "ranked_jobs" not in st.session_state:
    st.session_state.ranked_jobs = None

uploaded_resume = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded_resume and st.session_state.resume_text is None:
    # ✅ Parse resume only once
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_resume.read())
        tmp_path = tmp.name

    st.success("✅ Resume uploaded successfully!")
    with st.spinner("Parsing resume..."):
        resume_parser = ResumeParser(tmp_path)
        st.session_state.resume_text = resume_parser.extract_text()

    # ✅ Remove temp file but keep text in session
    os.remove(tmp_path)

# ✅ If resume parsed successfully, enable job search
if st.session_state.resume_text:
    if st.button("🔍 Find Matching Jobs"):
        with st.spinner("Scraping job listings..."):
            scraper = JobScraper(search_term=job_query, location=job_location, num_results=num_results)
            df_jobs = scraper.scrape()

        # ✅ Fallback if scraping returns nothing
        if df_jobs.empty:
            st.warning("⚠ No jobs found! Using sample dummy jobs instead.")
            df_jobs = pd.DataFrame([
                {
                    "title": "AI Engineer",
                    "company": "OpenAI",
                    "location": "Remote",
                    "description": "Work on cutting-edge AI models and deployment.",
                    "job_url": "https://example.com/job1"
                },
                {
                    "title": "Machine Learning Engineer",
                    "company": "Google DeepMind",
                    "location": "London, UK",
                    "description": "Build ML pipelines, optimize deep learning models.",
                    "job_url": "https://example.com/job2"
                },
                {
                    "title": "Data Scientist",
                    "company": "Meta AI",
                    "location": "Remote",
                    "description": "Analyze large datasets, create AI-powered insights.",
                    "job_url": "https://example.com/job3"
                }
            ])

        with st.spinner("Generating embeddings & ranking matches..."):
            vectorizer = Vectorizer()
            resume_emb = vectorizer.embed_text(st.session_state.resume_text)
            job_descriptions = df_jobs["description"].tolist()
            job_embs = vectorizer.embed_texts(job_descriptions) if len(job_descriptions) > 0 else []

            if len(job_embs) == 0:
                st.error("❌ No job embeddings generated. Cannot rank jobs.")
            else:
                matcher = JobMatcher()
                ranked_jobs = matcher.rank_jobs(resume_emb, job_embs, df_jobs, top_k=5)
                st.session_state.ranked_jobs = ranked_jobs

# ✅ Display ranked jobs if available
if st.session_state.ranked_jobs is not None:
    st.subheader("📌 Top Matching Jobs")
    st.dataframe(st.session_state.ranked_jobs[["title", "company", "location", "similarity", "job_url"]])

    job_index = st.selectbox(
        "Select a job for Q&A",
        st.session_state.ranked_jobs.index,
        format_func=lambda i: st.session_state.ranked_jobs.loc[i, "title"]
    )

    if job_index is not None:
        selected_job_desc = st.session_state.ranked_jobs.loc[job_index, "description"]

        st.subheader("💬 Ask a Question about this Job")
        user_question = st.text_input("Type your question", value="What are the key responsibilities?")

        if st.button("🤖 Get Answer"):
            with st.spinner("Thinking..."):
                llm = LocalLLM()
                answer = llm.answer_question(selected_job_desc, user_question)
            st.success(answer)
else:
    if not uploaded_resume:
        st.info("Please upload your resume to start.")

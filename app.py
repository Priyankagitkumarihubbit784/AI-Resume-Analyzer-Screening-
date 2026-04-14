import streamlit as st
import pickle
import pdfplumber
import plotly.graph_objects as go
import tempfile

from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide"
)

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# -------------------------------
# LOAD MODELS
# -------------------------------

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_rewriter():
    return pipeline("text2text-generation", model="google/flan-t5-base")

@st.cache_resource
def load_ml_model():
    model = pickle.load(open("model/model.pkl", "rb"))
    vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
    return model, vectorizer


summarizer = load_summarizer()
embed_model = load_embedding_model()
rewriter = load_rewriter()
model, vectorizer = load_ml_model()


# -------------------------------
# SKILLS DATABASE
# -------------------------------

skills_database = [
    "python","machine learning","deep learning","data science","sql",
    "java","c++","html","css","javascript","react","node",
    "tensorflow","pytorch","nlp"
]


# -------------------------------
# PDF TEXT EXTRACTION
# -------------------------------

def extract_text_from_pdf(file):

    text = ""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text


# -------------------------------
# SKILL DETECTION
# -------------------------------

def detect_skills_bert(resume_text):

    detected = []

    resume_lower = resume_text.lower()

    for skill in skills_database:
        if skill in resume_lower:
            detected.append(skill)

    resume_embedding = embed_model.encode([resume_text])

    for skill in skills_database:

        skill_embedding = embed_model.encode([skill])

        similarity = cosine_similarity(
            resume_embedding,
            skill_embedding
        )[0][0]

        if similarity > 0.4 and skill not in detected:
            detected.append(skill)

    return detected


# -------------------------------
# ATS SCORE
# -------------------------------

def calculate_ats_score(found_skills, word_count, confidence, job_similarity=0):

    skill_score = min(len(found_skills) * 8, 40)

    if word_count < 150:
        length_score = 5
    elif word_count < 300:
        length_score = 10
    elif word_count < 600:
        length_score = 20
    else:
        length_score = 15

    confidence_score = min(confidence * 0.2, 20)

    job_score = job_similarity * 20

    total_score = skill_score + length_score + confidence_score + job_score

    return min(total_score, 100)


# -------------------------------
# RESUME REWRITER
# -------------------------------

def rewrite_resume_line(line):

    prompt = f"""
You are a professional resume writer.

Rewrite the following weak resume bullet point into a strong, professional, ATS-optimized bullet point.

Weak bullet point:
{line}

Improved bullet point:
"""

    result = rewriter(
        prompt,
        max_length=80,
        temperature=0.7,
        do_sample=True
    )

    text = result[0]["generated_text"]

    return text


# -------------------------------
# INTERVIEW QUESTIONS
# -------------------------------

def generate_interview_questions(skills):

    if len(skills) == 0:
        skills = ["programming", "software development"]

    prompt = f"""
You are a technical interviewer.

Generate 5 technical interview questions for a candidate with these skills:
{', '.join(skills)}

Rules:
- Only technical questions
- No HR questions
- Questions related to programming, machine learning, or software development
- Return numbered questions
"""

    result = rewriter(prompt, max_length=200, do_sample=False)

    return result[0]["generated_text"]


# -------------------------------
# PDF REPORT
# -------------------------------

def generate_pdf_report(prediction, confidence, ats_score, skills, word_count):

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    c = canvas.Canvas(temp_file.name, pagesize=letter)

    y = 750

    c.drawString(50, y, "AI Resume Analysis Report")
    y -= 40

    c.drawString(50, y, f"Predicted Role: {prediction}")
    y -= 30

    c.drawString(50, y, f"Confidence: {confidence:.2f}%")
    y -= 30

    c.drawString(50, y, f"ATS Score: {ats_score:.2f}%")
    y -= 30

    c.drawString(50, y, f"Word Count: {word_count}")
    y -= 30

    c.drawString(50, y, f"Skills: {', '.join(skills)}")

    c.save()

    return temp_file.name


# -------------------------------
# UI
# -------------------------------

st.title("🤖 AI Resume Analyzer")

uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

job_desc = st.text_area("Paste Job Description (optional)")


# -------------------------------
# ANALYSIS
# -------------------------------

if st.button("Analyze Resume"):

    if uploaded_file is None:

        st.warning("Please upload a resume first")

    else:

        with st.spinner("Analyzing resume..."):

            resume_text = extract_text_from_pdf(uploaded_file)

            resume_text = resume_text.replace("\n", " ").strip()

            word_count = len(resume_text.split())

            summary = summarizer(
                resume_text[:1200],
                max_length=120,
                min_length=40,
                do_sample=False
            )

            resume_vector = vectorizer.transform([resume_text])

            prediction = model.predict(resume_vector)[0]

            probabilities = model.predict_proba(resume_vector)

            confidence = max(probabilities[0]) * 100

            found_skills = detect_skills_bert(resume_text)

            similarity = 0
            if job_desc:
                job_vec = vectorizer.transform([job_desc])
                similarity = cosine_similarity(resume_vector, job_vec)[0][0]

            ats_score = calculate_ats_score(
                found_skills,
                word_count,
                confidence,
                similarity if job_desc else 0
            )

            st.session_state.analysis_done = True
            st.session_state.skills = found_skills
            st.session_state.prediction = prediction
            st.session_state.confidence = confidence
            st.session_state.ats = ats_score
            st.session_state.summary = summary[0]["summary_text"]
            st.session_state.word_count = word_count


# -------------------------------
# RESULTS
# -------------------------------

if st.session_state.analysis_done:

    st.subheader("AI Resume Summary")
    st.write(st.session_state.summary)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted Role", st.session_state.prediction)

    with col2:
        st.metric("Confidence", f"{st.session_state.confidence:.2f}%")

    with col3:
        st.metric("Skills Found", len(st.session_state.skills))


    st.subheader("Detected Skills")

    for skill in st.session_state.skills:
        st.markdown(f"- {skill}")


    st.subheader("ATS Score")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=st.session_state.ats,
        gauge={'axis': {'range': [0, 100]}}
    ))

    st.plotly_chart(fig)


    st.subheader("AI Interview Question Generator")

    if st.button("Generate Interview Questions"):

        questions = generate_interview_questions(st.session_state.skills)

        st.success("Interview Questions")

        st.write(questions)


    st.subheader("Download Resume Report")

    if st.button("Download PDF Report"):

        pdf_file = generate_pdf_report(
            st.session_state.prediction,
            st.session_state.confidence,
            st.session_state.ats,
            st.session_state.skills,
            st.session_state.word_count
        )

        with open(pdf_file, "rb") as f:

            st.download_button(
                label="Download PDF",
                data=f,
                file_name="resume_report.pdf",
                mime="application/pdf"
            )


    st.subheader("AI Resume Rewriter")

    weak_point = st.text_area("Paste a weak resume bullet point")

    if st.button("Rewrite with AI"):

        if weak_point:

            improved = rewrite_resume_line(weak_point)

            st.success("Improved Resume Line")

            st.write(improved)
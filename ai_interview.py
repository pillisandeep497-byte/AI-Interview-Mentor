import streamlit as st
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()
api_key = st.secrets["OPEN_API_KEY"]
st.set_page_config(
    page_title="AI Interview Mentor",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 AI Interview Mentor")

st.markdown(
    "Practice technical interviews with AI-powered feedback."
)

# -----------------------------
# LLM
# -----------------------------

llm = ChatOpenRouter(
    model="openai/gpt-oss-20b",
    api_key=api_key,
    temperature=0.7
)

# -----------------------------
# Structured Output
# -----------------------------

class InterviewModel(BaseModel):
    interview_question: str = Field(
        description="Interview Question"
    )

    expert_answer: str = Field(
        description="Expert Level Answer"
    )

    improvement_tips: str = Field(
        description="Improvement Suggestions"
    )

parser = PydanticOutputParser(
    pydantic_object=InterviewModel
)

# -----------------------------
# Sidebar
# -----------------------------

role = st.sidebar.selectbox(
    "Select Role",
    [
        "Generative AI Engineer",
        "AI Engineer",
        "Machine Learning Engineer",
        "Python Developer",
        "Data Scientist"
    ]
)

experience = st.sidebar.selectbox(
    "Experience Level",
    [
        "Beginner",
        "Intermediate",
        "Advanced"
    ]
)

topic = st.text_input(
    "Interview Topic",
    placeholder="Example: RAG, LangChain, Python, FAISS"
)

# -----------------------------
# Prompt
# -----------------------------

prompt = ChatPromptTemplate.from_template(
"""
You are a senior interviewer.

Role:
{role}

Experience:
{experience}

Topic:
{topic}

Generate:

1. One challenging interview question.
2. An expert answer.
3. Suggestions to improve candidate response.

{format_instructions}
"""
)

chain = prompt | llm

# -----------------------------
# Generate
# -----------------------------

if st.button("Generate Interview Question"):

    with st.spinner("Preparing Interview..."):

        response = chain.invoke(
            {
                "role": role,
                "experience": experience,
                "topic": topic,
                "format_instructions":
                parser.get_format_instructions()
            }
        )

        result = parser.parse(
            response.content
        )

    st.subheader("🎤 Interview Question")
    st.info(result.interview_question)

    st.subheader("✅ Expert Answer")
    st.success(result.expert_answer)

    st.subheader("📈 Improvement Tips")
    st.warning(result.improvement_tips)
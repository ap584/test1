from pydantic_ai import Agent
from pydantic import BaseModel
from typing import Optional
from pydantic_ai.models.openai import OpenAIModel
import streamlit as st
from pymupdf4llm import PdfReader

# Define your models
class Resume(BaseModel):
    name: Optional[str]
    email: Optional[str]
    phone_number: Optional[str]
    resume_text: Optional[str]

class Continuous(BaseModel):
    continous_learning: Optional[str]

class professional_opputunities(BaseModel):
    pl: Optional[str]

class Self_learning(BaseModel):
    self_learning: Optional[str]

class Hackathon(BaseModel):
    hackathons: Optional[str]

class Recent_learning(BaseModel):
    rec_learn: Optional[str]

class Intellectual_curiosity(BaseModel):
    intellectual_curiosity: Optional[str]

# Load LLM
model = OpenAIModel("gpt-4o")

# Agents
agent = Agent(model, result_type=Resume)
learning_agent = Agent(model, result_type=Continuous)
professional_agent = Agent(model, result_type=professional_opputunities)
self_learning_agent = Agent(model, result_type=Self_learning)
hackathon_agent = Agent(model, result_type=Hackathon)
recent_learning_agent = Agent(model, result_type=Recent_learning)
intellectual_curiosity_agent = Agent(model, result_type=Intellectual_curiosity)

# Streamlit app
st.title("Resume Parser with pymupdf4llms")
uploaded_file = st.file_uploader("Upload a PDF Resume", type=["pdf"])

if uploaded_file is not None:
    reader = PdfReader.from_path(uploaded_file)
    full_text = reader.text

    st.subheader("Extracted Resume Text")
    st.text_area("Text", full_text, height=300)

    # Primary resume info
    prompt = f"""
    You are a resume analyser. You need to extract the following:
    - Name
    - Email
    - Phone number
    - Resume text
    
    Resume:
    {full_text}
    """
    res = agent.run_sync(prompt)

    if res.data:
        st.subheader("Parsed Resume Data")
        st.json(res.data)
    else:
        st.error("Resume parsing failed.")

    # Continuous learning
    learning_prompt = f"""
    You are a resume analyst. Does the resume demonstrate continuous learning via:
    - Formal education (degrees, diplomas)
    - Certifications
    - Online courses

    Resume:
    {full_text}
    """
    learning_res = learning_agent.run_sync(learning_prompt)
    if learning_res.data:
        st.subheader("Continuous Learning")
        st.write(learning_res.data)

    # Professional Opportunities
    professional_prompt = f"""
    Answer with "Yes" or "No". If "Yes", provide the relevant details.
    Q: Does the candidate frequently pursue professional development opportunities?

    Resume:
    {full_text}
    """
    professional_res = professional_agent.run_sync(professional_prompt)
    if professional_res.data:
        st.subheader("Professional Development Opportunities")
        st.write(professional_res.data)

    # Self-Learning
    self_learning_prompt = f"""
    Are there self-directed learning projects or independent study that show intrinsic motivation?

    Resume:
    {full_text}
    """
    self_res = self_learning_agent.run_sync(self_learning_prompt)
    if self_res.data:
        st.subheader("Self-Directed Learning")
        st.write(self_res.data)

    # Hackathons
    hackathon_prompt = f"""
    Has the candidate participated in hackathons or competitions?

    Resume:
    {full_text}
    """
    hackathon_res = hackathon_agent.run_sync(hackathon_prompt)
    if hackathon_res.data:
        st.subheader("Hackathon Participation")
        st.write(hackathon_res.data)

    # Recent Learning
    rec_learn_prompt = f"""
    Does the resume show recent learning activities that align with current industry trends?

    Resume:
    {full_text}
    """
    rec_learn_res = recent_learning_agent.run_sync(rec_learn_prompt)
    if rec_learn_res.data:
        st.subheader("Recent Learning Activities")
        st.write(rec_learn_res.data)

    # Intellectual Curiosity
    intellectual_curiosity_prompt = f"""
    Is there evidence of learning skills unrelated to the candidate's job, showing intellectual curiosity?

    Resume:
    {full_text}
    """
    intellectual_curiosity_res = intellectual_curiosity_agent.run_sync(intellectual_curiosity_prompt)
    if intellectual_curiosity_res.data:
        st.subheader("Intellectual Curiosity")
        st.write(intellectual_curiosity_res.data)

from pydantic_ai import Agent
from pydantic import BaseModel, Field
from typing import Optional
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import streamlit as st
import pdfplumber
from termcolor import colored


class Resume(BaseModel):
    name: Optional[str]
    email: Optional[str]
    phone_number: Optional[str]
    resume_text: Optional[str]

class Frequency(BaseModel):
    answer: str = Field(..., description="Answer the question")
    reason: Optional[str] = Field(None, description="Justify the answer")

class Job_change(BaseModel):
    answer: str = Field(..., description="Answer the question")
    reason: Optional[str] = Field(None, description="Justify the answer")

class YesNoAnswer(BaseModel):
    answer: str = Field(..., description="Answer as 'Yes' or 'No'")
    reason: Optional[str] = Field(None, description="Justify the answer")

class Hackathon(BaseModel):
    answer: str = Field(..., description="Answer the question")
    reason: Optional[str] = Field(None, description="Justify the answer")


model = OpenAIModel('gpt-4o', provider=OpenAIProvider())


agent = Agent(model, result_type=Resume)
learning_agent = Agent(model, result_type=YesNoAnswer)
professional_agent = Agent(model, result_type=YesNoAnswer)
recent_learning_agent = Agent(model, result_type=YesNoAnswer)
process_improvement_agent = Agent(model, result_type=YesNoAnswer)
overcoming_challenges_agent = Agent(model, result_type=YesNoAnswer)
leadership_progression_agent = Agent(model, result_type=YesNoAnswer)
strategic_influence_agent = Agent(model, result_type=YesNoAnswer)
mentorship_agent = Agent(model, result_type=YesNoAnswer)
career_advancement_agent = Agent(model, result_type=YesNoAnswer)
promotions_agent = Agent(model, result_type=YesNoAnswer)
work_gaps_agent = Agent(model, result_type=YesNoAnswer)
#freq_job_agent = Agent(model, result_type=YesNoAnswer)
frequency_agent = Agent(model, result_type=Frequency)
job_change_agent = Agent(model, result_type=Job_change)
hackathon_agent = Agent(model, result_type=Hackathon)



st.title("Resume Parser")
st.markdown("Upload Resume")

uploaded_file = st.file_uploader("Choose a pdf file", type=["pdf"])

if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        full_text = "".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    st.subheader("Extracted text")
    st.text_area("Resume Text", full_text, height=300)

    
    prompt = f"""
    You are a resume analyser. Extract:
    - Name
    - Email
    - Phone number
    - Resume text

    Resume:
    {full_text}
    """
    res = agent.run_sync(prompt)
    if res.data:
        st.subheader("Parsed Data")
        st.json(res.data)

    
    weights = {
        "learning_agent": 1.0,
        "professional_agent": 1.0,
        "recent_learning_agent": 1.0,
        "process_improvement_agent": 1.0,
        "overcoming_challenges_agent": 1.0,
        "leadership_progression_agent": 1.0,
        "strategic_influence_agent": 1.0,
        "mentorship_agent": 1.0,
        "career_advancement_agent": 1.0,
        "promotions_agent": 1.0,
        "work_gaps_agent": 1.0,
    }

    agents_info = [
        (learning_agent, "Does the resume demonstrate a pattern of continuous learning through certifications or online courses relevant to their career?", "learning_agent"),
        (professional_agent, "How frequently does the candidate pursue professional development opportunities compared to industry standards?", "professional_agent"),
        (recent_learning_agent, "Does the resume show evidence of staying current with industry trends through recent learning?", "recent_learning_agent"),
        (process_improvement_agent, "Are there examples of the candidate creating new processes, tools, or methodologies to improve efficiency?", "process_improvement_agent"),
        (overcoming_challenges_agent, "Does the candidate mention overcoming significant obstacles or challenges in their career path?", "overcoming_challenges_agent"),
        (leadership_progression_agent, "Has the candidate progressively taken on more leadership responsibilities?", "leadership_progression_agent"),
        (strategic_influence_agent, "Is there evidence of the candidate influencing organizational decisions or strategic direction?", "strategic_influence_agent"),
        (mentorship_agent, "Does the resume reflect experiences mentoring junior team members or leading knowledge-sharing initiatives?", "mentorship_agent"),
        (career_advancement_agent, "Is there a clear progression in the complexity or scope of responsibilities over time that indicates deliberate career advancement?", "career_advancement_agent"),
        (promotions_agent, "Does the resume show evidence of promotions or increased responsibilities within the same organization?", "promotions_agent"),
        (work_gaps_agent, "Are there any significant gaps in the candidate's work history that require explanation?", "work_gaps_agent"),
    ]

    def ask_question(agent, question_text, key, weight):
        prompt = f"""
        You are a resume analyst. Answer according to the following format:

        Answer: Yes or No
        Reason: A brief justification from the resume

        Question:
        {question_text}

        Resume:
        {full_text}
        """
        res = agent.run_sync(prompt)
        score = 0
        if res.data:
            st.subheader(question_text)
            st.write("Answer:", res.data.answer)
            st.write("Reason:", res.data.reason)

            answer = res.data.answer.strip().lower()
            if answer == "yes":
                score = 1 * weight
            elif answer == "no":
                score = 0 * weight
            score_color = "green" if score > 0 else "red"
        st.markdown(f"<span style='color:{score_color}; font-weight:bold;'>Score: +{score}</span>", unsafe_allow_html=True)
            
        return score

    total_score = 0
    max_score = 0

    for agent_instance, question, key in agents_info:
        weight = weights.get(key, 1.0)
        score = ask_question(agent_instance, question, key, weight)
        total_score += score
        max_score += abs(weight)
        #st.write("Score: ", score)
        

    

    
    freq_prompt = f"""
    How frequently does the candidate pursue professional development opportunities compared to industry standards?
    Answer according to the instructions below:
    Answer: How frequently
    Reason: A brief justification

    Resume:
    {full_text}
    """
    freq_res = frequency_agent.run_sync(freq_prompt)

    if freq_res.data:
        st.subheader("How frequently does the candidate pursue professional development opportunities compared to industry standards?")
        st.write("Answer: ", freq_res.data.answer)
        st.write("Reason: ", freq_res.data.reason)

    hackathon_prompt = f"""
    Has the candidate participated in hackathons, competitions, or challenges related to their field?
    Answer according to the instructions below:
    Answer: How many (a number)
    Reason: A brief justification

    Resume:
    {full_text}
    """

    hackathon_res = hackathon_agent.run_sync(hackathon_prompt)

    if(hackathon_res.data):
        st.subheader("Has the candidate participated in hackathons, competitions, or challenges related to their field?")
        st.write("Answer:", hackathon_res.data.answer)
        st.write("Reason:", hackathon_res.data.reason)

    job_change_prompt = f"""
    Are there any frequent job changes? A job change can be a change in organization in less than 2 years.
    Answer according to the instructions below:
    Answer: How frequently (a number)
    Reason: A brief justification

    Resume:
    {full_text}
    """
    job_res = job_change_agent.run_sync(job_change_prompt)
    if job_res.data:
        st.subheader("Are there any frequent job changes?")
        st.write("Answer: ", job_res.data.answer, " job changes")
        st.write("Reason: ", job_res.data.reason)

    
    st.subheader("Resume Score")
    st.write(f"Score: {total_score:.2f} / {max_score:.2f}")
    
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from typing import Optional
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import streamlit as st
import pdfplumber

class Resume(BaseModel):
    name: Optional[str]
    email: Optional[str]
    phone_number: Optional[str]
    resume_text: Optional[str]

class YesNoAnswer(BaseModel):
    answer: str = Field(..., description="Answer as 'Yes' or 'No'")
    reason: Optional[str] = Field(None, description="Justification for the answer")

model = OpenAIModel('gpt-4o', provider=OpenAIProvider())


agent = Agent(model, result_type=Resume)
learning_agent = Agent(model, result_type=YesNoAnswer)
professional_agent = Agent(model, result_type=YesNoAnswer)
hackathon_agent = Agent(model, result_type=YesNoAnswer)
recent_learning_agent = Agent(model, result_type=YesNoAnswer)
process_improvement_agent = Agent(model, result_type=YesNoAnswer)
overcoming_challenges_agent = Agent(model, result_type=YesNoAnswer)
leadership_progression_agent = Agent(model, result_type=YesNoAnswer)
strategic_influence_agent = Agent(model, result_type=YesNoAnswer)
mentorship_agent = Agent(model, result_type=YesNoAnswer)
career_advancement_agent = Agent(model, result_type=YesNoAnswer)
promotions_agent = Agent(model, result_type=YesNoAnswer)
work_gaps_agent = Agent(model, result_type=YesNoAnswer)

st.title("Resume Parser")
st.markdown("Upload resume")

uploaded_file = st.file_uploader("Choose a PDF File...", type=["pdf"])

if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        full_text = "".join([page.extract_text() for page in pdf.pages])

    st.subheader("Extracted text from resume:")
    st.text_area("Resume text", full_text, height=300)

    prompt = f"""
    you are a resume analyser. Extract:
    - Name
    - Email
    - Phone number
    - Resume text

    Resume:
    {full_text}
    """
    res = agent.run_sync(prompt)
    if res.data:
        st.subheader("Parsed data:")
        st.json(res.data)
    else:
        st.error("Failed")

    def ask_question(agent, question_text, subheader):
        prompt = f"""
        You are a resume analyst. Based on the resume below, answer the following question strictly in this format:

        Answer: Yes or No
        Reason: A brief justification from the resume.

        Question:
        {question_text}

        Resume:
        {full_text}
        """
        res = agent.run_sync(prompt)
        if res.data:
            st.subheader(question_text)
            st.write("Answer:", res.data.answer)
            st.write("Reason:", res.data.reason)

    ask_question(learning_agent, "Does the resume demonstrate a pattern of continuous learning through certifications or online courses relevant to their career?", "Continuous Learning")
    ask_question(professional_agent, "How frequently does the candidate pursue professional development opportunities compared to industry standards?", "Professional Development Opportunities")
    ask_question(hackathon_agent, "Has the candidate participated in hackathons, competitions, or challenges related to their field?", "Hackathon Participation")
    ask_question(recent_learning_agent, "Does the resume show evidence of staying current with industry trends through recent learning?", "Recent Learning")
    ask_question(process_improvement_agent, "Are there examples of the candidate creating new processes, tools, or methodologies to improve efficiency?", "Process Improvement")
    ask_question(overcoming_challenges_agent, "Does the candidate mention overcoming significant obstacles or challenges in their career path?", "Overcoming Challenges")
    ask_question(leadership_progression_agent, "Has the candidate progressively taken on more leadership responsibilities?", "Leadership Progression")
    ask_question(strategic_influence_agent, "Is there evidence of the candidate influencing organizational decisions or strategic direction?", "Strategic Influence")
    ask_question(mentorship_agent, "Does the resume reflect experiences mentoring junior team members or leading knowledge-sharing initiatives?", "Mentorship")
    ask_question(career_advancement_agent, "Is there a clear progression in the complexity or scope of responsibilities over time that indicates deliberate career advancement?", "Career Advancement")
    ask_question(promotions_agent, "Does the resume show evidence of promotions or increased responsibilities within the same organization?", "Promotions")
    ask_question(work_gaps_agent, "Are there any significant gaps in the candidate's work history that require explanation?", "Work Gaps")

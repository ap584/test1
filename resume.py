from pydantic_ai import Agent
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import streamlit as st
import pdfplumber

class Resume(BaseModel):
    name:Optional[str]
    email:Optional[str]
    phone_number:Optional[str]
    resume_text:Optional[str]

class Continuous(BaseModel):
    continous_learning:Optional[str]

class professional_opputunities(BaseModel):
    pl:Optional[str]

class Self_learning(BaseModel):
    self_learning:Optional[str]

class Hackathon(BaseModel):
    hackathons:Optional[str]

class Recent_learning(BaseModel):
    rec_learn:Optional[str]

class Intellectual_curiosity(BaseModel):
    intellectual_curiosity:Optional[str]

class Learning_platforms(BaseModel):
    learning_platforms:Optional[str]

class Challenging_assignments(BaseModel):
    challenging_assignments:Optional[str]
class ProcessImprovement(BaseModel):
    process_improvement: Optional[str]

class OvercomingChallenges(BaseModel):
    overcoming_challenges: Optional[str]

class LeadershipProgression(BaseModel):
    leadership_progression: Optional[str]

class StrategicInfluence(BaseModel):
    strategic_influence: Optional[str]

class Mentorship(BaseModel):
    mentorship: Optional[str]

class CareerAdvancement(BaseModel):
    career_advancement: Optional[str]

class Promotions(BaseModel):
    promotions: Optional[str]

class WorkGaps(BaseModel):
    work_gaps: Optional[str]



model = OpenAIModel('gpt-4o', provider=OpenAIProvider())
agent = Agent(model, result_type=Resume)
learning_agent = Agent(model, result_type=Continuous)
professional_agent = Agent(model, result_type=professional_opputunities)
self_learning_agent = Agent(model, result_type=Self_learning)
hackathon_agent = Agent(model, result_type=Hackathon)
recent_learning_agent = Agent(model, result_type=Recent_learning)
intellectual_curiosity_agent = Agent(model, result_type=Intellectual_curiosity)
learning_platform_agent = Agent(model, result_type=Learning_platforms)
challenging_assignment_agent = Agent(model, result_type=Challenging_assignments)
process_improvement_agent = Agent(model, result_type=ProcessImprovement)
overcoming_challenges_agent = Agent(model, result_type=OvercomingChallenges)
leadership_progression_agent = Agent(model, result_type=LeadershipProgression)
strategic_influence_agent = Agent(model, result_type=StrategicInfluence)
mentorship_agent = Agent(model, result_type=Mentorship)
career_advancement_agent = Agent(model, result_type=CareerAdvancement)
promotions_agent = Agent(model, result_type=Promotions)
work_gaps_agent = Agent(model, result_type=WorkGaps)


st.title("Resume Parser")
st.markdown("Upload resume")

uploaded_file = st.file_uploader("Choose a PDF File...", type=["pdf"])

if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        full_text = ""

        for page in pdf.pages:
            full_text += page.extract_text()


    st.subheader("Extracted text from resume:")
    st.text_area("Resume text", full_text, height = 300)

    prompt = f"""
    you are a resume analyser. You need to extract the following:
    -Name
    -email
    -Phone number
    -resume text

    
    {full_text}
    
    """
    res = agent.run_sync(prompt)

    if res.data:
        st.subheader("Parsed data:")
        st.json(res.data)
    else:
        st.error("Failed")

    learning_prompt = f"""
    You are a resume analyst. Please determine if the following resume demonstrates a pattern of continuous learning through:
    (Ignore the Formal education like degrees, diplomas, etc.)
    - Certifications
    - Online courses or other learning experiences relevant to the career path

    Resume Text:
    {full_text}

    Answer: "Yes" or "No"  
    Reason: A brief justification from the resume content.
    """
    learning_res = learning_agent.run_sync(learning_prompt)

    if learning_res.data:
        st.subheader("Does the resume demonstrate a pattern of continuous learning through formal education, certifications, or courses relevant to their career path?")
        st.write(learning_res.data)

    professional_prompt = f"""
    Answer: "Yes" or "No"   
    Reason: A brief justification from the resume content.
    Q. How frequently does the candidate pursue professional development opportunities compared to industry standards?
    {full_text}
    """

    professional_res = professional_agent.run_sync(professional_prompt)

    if professional_res.data:
        st.subheader("How frequently does the candidate pursue professional development opportunities compared to industry standards?")
        st.write(professional_res.data)

    
    
    hackathon_prompt = f"""
    Answer: "Yes" or "No"  
    Reason: A brief justification from the resume content.
    
    Has the candidate participated in hackathons, competitions, or challenges related to their field?
    {full_text}
    """
    hackathon_res = hackathon_agent.run_sync(hackathon_prompt)

    if hackathon_res.data:
        st.subheader("Has the candidate participated in hackathons, competitions, or challenges related to their field?")
        st.write(hackathon_res.data)
    else:
        st.write("Not found")

    rec_learn_promt = f"""
    Answer: "Yes" or "No"  
    Reason: A brief justification from the resume content.

    Does the resume show evidence of the candidate staying current with industry trends through recent learning activities?
    {full_text}
    """

    rec_learn_res = recent_learning_agent.run_sync(rec_learn_promt)

    if rec_learn_res.data:
        st.subheader("Does the resume show evidence of the candidate staying current with industry trends through recent learning activities?")
        st.write(rec_learn_res.data)
    else:
        st.write("Not found")
    
    
    process_improvement_prompt = f"""
    Answer: "Yes" or "No"  
    Reason: A brief justification from the resume content.

    Are there examples of the candidate creating new processes, tools, or methodologies to improve efficiency?

    Resume:
    {full_text}
    """

    res_process_improvement = process_improvement_agent.run_sync(process_improvement_prompt)
    
    if res_process_improvement.data:
        st.subheader("Are there examples of the candidate creating new processes, tools, or methodologies to improve efficiency?")
        st.write(res_process_improvement.data)

    overcoming_challenges_prompt = f"""
    Answer: "Yes" or "No"  
    Reason: A brief justification from the resume content.

    Does the candidate mention overcoming significant obstacles or challenges in their career path?

    Resume:
    {full_text}
    """
    res_challenges = overcoming_challenges_agent.run_sync(overcoming_challenges_prompt)
    if res_challenges.data:
        st.subheader("Does the candidate mention overcoming significant obstacles or challenges in their career path?")
        st.write(res_challenges.data)

    # Leadership Progression
    leadership_progression_prompt = f"""
    Answer: "Yes" or "No"  
    Reason: A brief justification from the resume content.

    Has the candidate progressively taken on more leadership responsibilities throughout their career?

    Resume:
    {full_text}
    """
    res_leadership = leadership_progression_agent.run_sync(leadership_progression_prompt)
    if res_leadership.data:
        st.subheader("Has the candidate progressively taken on more leadership responsibilities throughout their career?")
        st.write(res_leadership.data)

    strategic_influence_prompt = f"""
    Answer with "Yes" or "No". If "Yes", provide examples.

    Is there evidence of the candidate influencing organizational decisions or strategic direction?

    Resume:
    {full_text}
    """
    res_strategic = strategic_influence_agent.run_sync(strategic_influence_prompt)
    if res_strategic.data:
        st.subheader("Is there evidence of the candidate influencing organizational decisions or strategic direction?")
        st.write(res_strategic.data)

    # Mentorship
    mentorship_prompt = f"""
    Answer with "Yes" or "No". If "Yes", provide details.

    Does the resume reflect experiences mentoring junior team members or leading knowledge-sharing initiatives?

    Resume:
    {full_text}
    """
    res_mentorship = mentorship_agent.run_sync(mentorship_prompt)
    if res_mentorship.data:
        st.subheader("Does the resume reflect experiences mentoring junior team members or leading knowledge-sharing initiatives?")
        st.write(res_mentorship.data)

    career_advancement_prompt = f"""
    Answer with "Yes" or "No". If "Yes", provide insights.

    Is there a clear progression in the complexity or scope of responsibilities over time that indicates deliberate career advancement?

    Resume:
    {full_text}
    """
    res_advancement = career_advancement_agent.run_sync(career_advancement_prompt)
    if res_advancement.data:
        st.subheader("Is there a clear progression in the complexity or scope of responsibilities over time that indicates deliberate career advancement?")
        st.write(res_advancement.data)
    
    # Promotions
    promotion_prompt = f"""
    Answer with "Yes" or "No". If "Yes", provide context.

    Does the resume show evidence of promotions or increased responsibilities within the same organization?

    Resume:
    {full_text}
    """
    res_promotions = promotions_agent.run_sync(promotion_prompt)
    if res_promotions.data:
        st.subheader("Does the resume show evidence of promotions or increased responsibilities within the same organization?")
        st.write(res_promotions.data)
    
    # Work Gaps
    work_gap_prompt = f"""
    Answer with "Yes" or "No". If "Yes", indicate the timeframes or areas of concern.

    Are there any significant gaps in the candidate's work history that require explanation?

    Resume:
    {full_text}
    """
    res_gaps = work_gaps_agent.run_sync(work_gap_prompt)
    if res_gaps.data:
        st.subheader("sAre there any significant gaps in the candidate's work history that require explanation?")
        st.write(res_gaps.data)
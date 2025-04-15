from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import pdfplumber

app = FastAPI()

# === Models ===
class Resume(BaseModel):
    name: Optional[str]
    email: Optional[str]
    phone_number: Optional[str]
    resume_text: Optional[str]

class Frequency(BaseModel):
    answer: str
    reason: Optional[str]

class Job_change(BaseModel):
    answer: str
    reason: Optional[str]

class YesNoAnswer(BaseModel):
    answer: str
    reason: Optional[str]

class Hackathon(BaseModel):
    answer: str
    reason: Optional[str]


# === AI Setup ===
model = OpenAIModel("gpt-4o", provider=OpenAIProvider())

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
frequency_agent = Agent(model, result_type=Frequency)
job_change_agent = Agent(model, result_type=Job_change)
hackathon_agent = Agent(model, result_type=Hackathon)


@app.post("/analyze-resume/")
async def analyze_resume(file: UploadFile = File(...)):
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        with pdfplumber.open(file.file) as pdf:
            full_text = "".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {e}")

    if not full_text.strip():
        raise HTTPException(status_code=400, detail="Empty or unreadable PDF")

    # Extract basic resume info
    base_prompt = f"""
    You are a resume analyser. Extract:
    - Name
    - Email
    - Phone number
    - Resume text

    Resume:
    {full_text}
    """
    parsed_resume = await agent.run(base_prompt)

    # Define all evaluation questions
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

    results = []
    total_score = 0
    max_score = 0
    
    for agent_instance, question, key in agents_info:
        
        prompt = f"""
        You are a resume analyst. Answer according to the following format:

        Answer: Yes or No
        Reason: A brief justification from the resume

        Question:
        {question}

        Resume:
        {full_text}
        """
        res = await agent_instance.run(prompt)
        answer = res.data.answer.strip().lower() if res.data else "no"
        score = 1.0 * weights[key] if answer == "yes" else 0.0
        total_score += score
        max_score += weights[key]

        results.append({
            "question": question,
            "answer": res.data.answer if res.data else None,
            "reason": res.data.reason if res.data else None,
            "score": score
        })

    
    # Frequency
    freq_prompt = f"""
    How frequently does the candidate pursue professional development opportunities compared to industry standards?
    Answer: How frequently
    Reason: A brief justification

    Resume:
    {full_text}
    """
    freq_res = await frequency_agent.run(freq_prompt)

    # Hackathons
    hackathon_prompt = f"""
    Has the candidate participated in hackathons, competitions, or challenges related to their field?
    Answer: How many (a number)
    Reason: A brief justification

    Resume:
    {full_text}
    """
    hackathon_res = await hackathon_agent.run(hackathon_prompt)

    # Job changes
    job_change_prompt = f"""
    Are there any frequent job changes? A job change can be a change in organization in less than 2 years.
    Answer: How frequently (a number)
    Reason: A brief justification

    Resume:
    {full_text}
    """
    job_res = await job_change_agent.run(job_change_prompt)
    
    return {
        "parsed_resume": parsed_resume.data.dict() if parsed_resume.data else {},
        "analysis_results": results,
        "frequency": freq_res.data.dict() if freq_res.data else {},
        "hackathon": hackathon_res.data.dict() if hackathon_res.data else {},
        "job_change": job_res.data.dict() if job_res.data else {},
        "score": {
            "total_score": total_score,
            "max_score": max_score
        }
    }

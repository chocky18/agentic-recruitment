import os
import re
import json
import logging
from typing import List

from fastapi import FastAPI, UploadFile, File, Body, WebSocket, WebSocketDisconnect
import uvicorn
import docx2txt
import pdfplumber

# Env + keys
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

import strawberry
from strawberry.asgi import GraphQL

from fastapi.middleware.cors import CORSMiddleware
import uuid
# ------------------------------------
# ENV + Logging
# ------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "candidates-db")

# ------------------------------------
# Clients
# ------------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure Pinecone index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,  # embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# ------------------------------------
# Helpers
# ------------------------------------
def read_resume(file_path: str):
    """Extract text from PDF or DOCX."""
    if file_path.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    elif file_path.endswith(".docx"):
        return docx2txt.process(file_path).strip()
    else:
        raise ValueError("Unsupported file type")


def chunk_text(text: str, chunk_size: int = 500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def get_embedding(text: str):
    resp = client.embeddings.create(model="text-embedding-3-large", input=text)
    return resp.data[0].embedding


def extract_metadata(text: str):
    """Metadata extractor with auto-translation to English for multi-lingual resumes."""
    prompt = f"""
    The following is a resume text which could be in any language.
    1. Detect the language. 
    2. If not English, translate it to English.
    3. Extract JSON metadata from the resume text.
    
    JSON keys:
      - name
      - designation
      - skills
      - experience_years
      - location
    
    Return only valid JSON.
    
    Resume text (first 1000 chars):
    {text[:1000]}
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception:
        return {
            "name": "Unknown",
            "designation": None,
            "skills": [],
            "experience_years": 0,
            "location": None
        }

def generate_jd(role, years, location, skills=None):
    """Generate JD JSON from GPT, with defaults if input is missing."""
    role = role or ""
    location = location or ""
    skills = skills or []

    prompt = f"Generate JD JSON for role '{role}', {years} years, location '{location}', skills={skills}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = resp.choices[0].message.content.strip()
    # Remove markdown formatting if any
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    
    try:
        return json.loads(raw)
    except Exception:
        # Fallback: ensure every key exists
        return {
            "role": role,
            "years_experience": years,
            "location": location,
            "skills": skills,
            "responsibilities": []
        }

def score_resume_vs_jd(resume_meta, jd):
    jd_skills = set([s.lower() for s in jd["skills"]])
    candidate_skills = set([s.lower() for s in resume_meta.get("skills", [])])

    overlap = jd_skills.intersection(candidate_skills)
    skills_score = (len(overlap) / len(jd_skills)) * 100 if jd_skills else 0
    exp_required = jd.get("years_experience", 0)
    exp_candidate = resume_meta.get("experience_years", 0)
    exp_score = min(exp_candidate / exp_required, 1.0) * 100 if exp_required else 0
    designation = resume_meta.get("designation", "").lower()
    designation_score = 100 if jd["role"].lower() in designation else 0
    final_score = (0.6 * skills_score) + (0.3 * exp_score) + (0.1 * designation_score)

    return {
        "skills_score": round(skills_score, 2),
        "experience_score": round(exp_score, 2),
        "designation_score": round(designation_score, 2),
        "final_score": round(final_score, 2),
    }

# ------------------------------------
# REST Endpoints (Base Agents)
# ------------------------------------
app = FastAPI(title="Recruitment Agentic Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://agentic-recruitment.vercel.app",
        "https://agentic-recruitment-on37fv0j2-naresh-tinnaluris-projects.vercel.app"],  # frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload_resumes")
async def upload_resumes(files: List[UploadFile] = File(...)):
    uploaded = []
    for file in files:
        file_path = f"./{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Read text
        text = read_resume(file_path)

        # Extract metadata
        metadata = extract_metadata(text)

        # Sanitize metadata
        metadata_sanitized = {
            "name": metadata.get("name") or "Unknown",
            "designation": metadata.get("designation") or "Unknown",
            "skills": metadata.get("skills") or [],
            "experience_years": metadata.get("experience_years") or 0,
            "location": metadata.get("location") or "Unknown"
        }

        # Chunk text and upsert
        for i, chunk in enumerate(chunk_text(text, 500)):
            emb = get_embedding(chunk)

            # Generate safe ASCII ID
            vector_id = str(uuid.uuid4())  # always unique & ASCII

            index.upsert(
                vectors=[{
                    "id": vector_id,
                    "values": emb,
                    "metadata": metadata_sanitized  # keep original name here
                }]
            )

        uploaded.append({"filename": file.filename, "metadata": metadata_sanitized})

    return {"uploaded_files": uploaded}


@app.post("/generate_jd")
def create_jd(role, years, location, skills=None):
    """Generate JD JSON from GPT."""
    prompt = f"Generate JD JSON for role {role}, {years} years, {location}, skills={skills or []}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        jd_dict = json.loads(raw)
        # Ensure all keys exist to avoid frontend errors
        jd_dict.setdefault("role", role or "")
        jd_dict.setdefault("years_experience", years or 0)
        jd_dict.setdefault("location", location or "")
        jd_dict.setdefault("skills", skills or [])
        jd_dict.setdefault("responsibilities", [])
        return jd_dict
    except Exception:
        return {"role": role or "", "years_experience": years or 0, "location": location or "",
                "skills": skills or [], "responsibilities": []}




@app.post("/match_candidates")
async def match_candidates(jd: dict):
    jd_text = f"{jd['role']} in {jd['location']} requiring {jd['years_experience']} years. Skills: {', '.join(jd['skills'])}"
    emb = get_embedding(jd_text)
    results = index.query(vector=emb, top_k=5, include_metadata=True)
    scored = []
    for match in results["matches"]:
        resume_meta = match["metadata"]
        score = score_resume_vs_jd(resume_meta, jd)
        if score["skills_score"] > 0:
            scored.append({"candidate": resume_meta, "score": score})
    # sort by final_score descending and return top 5
    top_candidates = sorted(scored, key=lambda x: x["score"]["final_score"], reverse=True)[:5]

    return top_candidates

# ------------------------------------
# WebSocket Chat (Chatbot) with GPT Intent Parsing + Auto Skill Extraction
# Fully compatible with React frontend
# ------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            user_msg = await websocket.receive_text()

            # Step 1: Parse intent from user text using GPT
            intent_prompt = f"""
            Extract the task and parameters from this user message:
            "{user_msg}"
            Return JSON with keys:
              task (one of: generate_jd, match_candidates)
              role
              years (integer, approximate if needed)
              location (string, optional)
              skills (list; if not specified, suggest relevant skills)
            """
            intent_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": intent_prompt}],
                temperature=0
            )

            raw_json = intent_resp.choices[0].message.content.strip()
            raw_json = re.sub(r"^```json\s*|\s*```$", "", raw_json, flags=re.MULTILINE)
            try:
                intent = json.loads(raw_json)
            except Exception:
                intent = {"task": "unknown"}

            # Step 2: Auto-fill skills if empty
            if not intent.get("skills"):
                skill_prompt = f"""
                Suggest top skills for role: {intent.get('role', '')}.
                Return as a JSON list like ["skill1", "skill2"]
                """
                skill_resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": skill_prompt}],
                    temperature=0
                )
                skills_raw = skill_resp.choices[0].message.content.strip()
                skills_raw = re.sub(r"^```json\s*|\s*```$", "", skills_raw, flags=re.MULTILINE)
                try:
                    intent["skills"] = json.loads(skills_raw)
                except Exception:
                    intent["skills"] = []

            # Step 3: Route to the correct function
            if intent.get("task") == "generate_jd":
                jd = create_jd(
                    role=intent.get("role", ""),
                    years=intent.get("years", 0),
                    location=intent.get("location", "Not specified"),
                    skills=intent.get("skills", [])
                )
                response = {"type": "jd", "data": jd}  # make sure jd is dict
                await websocket.send_json(response)


            elif intent.get("task") == "match_candidates":
                jd_query = {
                    "role": intent.get("role", ""),
                    "years_experience": intent.get("years", 0),
                    "location": intent.get("location", "Not specified"),
                    "skills": intent.get("skills", [])
                }
                scored = await match_candidates(jd_query)

                # Ensure top 5 by final_score
                top_candidates = sorted(scored, key=lambda x: x["score"]["final_score"], reverse=True)[:5]

                response = {"type": "candidates", "data": top_candidates}


                # Step 4: Send structured JSON to frontend
                await websocket.send_json(response)

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")


# ------------------------------------
# GraphQL Schema
# ------------------------------------
@strawberry.type
class Query:
    hello: str = "GraphQL is working 🚀"

@strawberry.type
class Mutation:
    @strawberry.mutation
    def generateJD(self, role: str, years: int, location: str, skills: List[str]) -> str:
        jd = generate_jd(role, years, location, skills)
        return json.dumps(jd)

schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_app = GraphQL(schema)
app.mount("/graphql", graphql_app)

# ------------------------------------
# Run
# ------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Render injects PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)


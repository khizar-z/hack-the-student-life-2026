"""
main.py - Panda FastAPI backend

Core experience:
  - Landing search
  - Browse all professors (alphabetical)
  - Subject filtering
  - Professor detail page
  - Reach-out coaching pointers
  - Paper key takeaways
"""

import io
import json
import re
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import boto3
import numpy as np
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# -- Config ------------------------------------------------------------
REGION = "us-west-2"
EMBED_MODEL = "amazon.titan-embed-text-v1"
LLM_MODELS = [
    "us.amazon.nova-lite-v1:0",
    "us.amazon.nova-pro-v1:0",
    "us.amazon.nova-micro-v1:0",
]
DATA_FILE = "professors_embedded.json"
DYNAMO_TABLE_NAME = "ResearchConnectQueries"

DEFAULT_SUBJECT = "General Computer Science"
ALL_SUBJECTS = "All Subjects"

SUBJECT_KEYWORDS: Dict[str, List[str]] = {
    "AI & Machine Learning": [
        "machine learning",
        "deep learning",
        "neural",
        "nlp",
        "natural language",
        "language model",
        "reinforcement",
        "transformer",
        "classification",
        "prediction",
    ],
    "Systems & Networking": [
        "distributed",
        "network",
        "operating system",
        "systems",
        "cloud",
        "compiler",
        "parallel",
        "architecture",
        "storage",
    ],
    "Security & Privacy": [
        "security",
        "privacy",
        "crypt",
        "malware",
        "adversarial",
        "vulnerability",
        "authentication",
    ],
    "Theory & Algorithms": [
        "algorithm",
        "graph",
        "optimization",
        "theorem",
        "proof",
        "complexity",
        "combinatorial",
    ],
    "HCI, Robotics & Interaction": [
        "human computer",
        "hci",
        "interaction",
        "robot",
        "assistive",
        "interface",
        "user study",
        "design",
    ],
    "Data Science & Computational Biology": [
        "data mining",
        "bio",
        "genome",
        "rna",
        "protein",
        "healthcare",
        "medical",
        "diagnosis",
        "disease",
    ],
    "Vision, Graphics & Media": [
        "computer vision",
        "image",
        "video",
        "graphics",
        "multimedia",
        "rendering",
    ],
    "Software Engineering": [
        "software engineering",
        "code",
        "testing",
        "program analysis",
        "debug",
        "developer",
        "maintenance",
    ],
}

# -- Globals -----------------------------------------------------------
app = FastAPI(title="Panda - UofT Research Discovery")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

professors: List[dict] = []
embeddings_matrix: Optional[np.ndarray] = None
table = None
bedrock = None
active_llm_model = None

summary_cache: Dict[str, str] = {}
description_cache: Dict[str, str] = {}
image_cache: Dict[str, Optional[str]] = {}


# -- Utility -----------------------------------------------------------
def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def truncate_text(text: str, max_chars: int) -> str:
    value = normalize_space(text)
    if len(value) <= max_chars:
        return value
    clipped = value[: max_chars - 3].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return f"{clipped}..."


def get_papers(prof: dict) -> List[dict]:
    papers = prof.get("papers", [])
    if isinstance(papers, list):
        return papers
    return []


def first_nonempty(*values: str) -> str:
    for value in values:
        cleaned = normalize_space(value)
        if cleaned:
            return cleaned
    return ""


def infer_subject(prof: dict) -> str:
    parts = [
        prof.get("name", ""),
        prof.get("research_areas", ""),
        prof.get("research_interests", ""),
    ]
    for paper in get_papers(prof)[:5]:
        parts.append(paper.get("title", ""))
        parts.append((paper.get("abstract", "") or "")[:300])

    blob = " ".join(parts).lower()
    blob = normalize_space(blob)
    if not blob:
        return DEFAULT_SUBJECT

    scores: Dict[str, int] = {}
    for subject, keywords in SUBJECT_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in blob:
                score += 2 if " " in keyword else 1
        if score:
            scores[subject] = score

    if not scores:
        return DEFAULT_SUBJECT

    return max(scores.items(), key=lambda item: item[1])[0]


def build_fallback_summary(prof: dict) -> str:
    focus = first_nonempty(prof.get("research_interests", ""), prof.get("research_areas", ""))
    if focus:
        return truncate_text(focus, 140)

    papers = get_papers(prof)
    if papers:
        title = papers[0].get("title", "").strip()
        if title:
            return truncate_text(f"Recent work includes: {title}", 140)

    return "Explore this professor's profile for current research directions and publication history."


def professor_prompt_context(prof: dict, max_papers: int = 3) -> str:
    chunks = []
    research_text = first_nonempty(prof.get("research_interests", ""), prof.get("research_areas", ""))
    if research_text:
        chunks.append(f"Research focus text: {research_text}")

    papers = get_papers(prof)[:max_papers]
    if papers:
        lines = []
        for paper in papers:
            title = normalize_space(paper.get("title", "Untitled paper"))
            abstract = truncate_text(paper.get("abstract", "") or "", 280)
            if abstract:
                lines.append(f"- Title: {title}\n  Abstract: {abstract}")
            else:
                lines.append(f"- Title: {title}")
        chunks.append("Papers:\n" + "\n".join(lines))

    if not chunks:
        chunks.append("No detailed research text was available.")

    return "\n".join(chunks)


def build_search_blob(prof: dict) -> str:
    segments = [
        prof.get("name", ""),
        prof.get("subject", ""),
        prof.get("research_areas", ""),
        prof.get("research_interests", ""),
    ]
    for paper in get_papers(prof)[:6]:
        segments.append(paper.get("title", ""))
        segments.append((paper.get("abstract", "") or "")[:260])
    return normalize_space(" ".join(segments).lower())


def fallback_keyword_similarity(query: str, prof: dict) -> float:
    tokens = [token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) > 2]
    if not tokens:
        return 0.0

    blob = prof.get("search_blob", "")
    if not blob:
        blob = build_search_blob(prof)

    token_hits = sum(1 for token in tokens if token in blob)
    phrase_bonus = 1 if query.lower() in blob else 0
    score = (token_hits + phrase_bonus) / max(len(tokens), 1)
    return min(score * 100.0, 99.0)


def get_professor_by_id(prof_id: str) -> dict:
    if not prof_id.isdigit():
        raise HTTPException(status_code=404, detail="Professor not found.")
    idx = int(prof_id)
    if idx < 0 or idx >= len(professors):
        raise HTTPException(status_code=404, detail="Professor not found.")
    return professors[idx]


def subject_matches(prof: dict, selected_subject: Optional[str]) -> bool:
    if not selected_subject or selected_subject == ALL_SUBJECTS:
        return True
    return prof.get("subject") == selected_subject


def build_card_payload(prof: dict, similarity: Optional[float] = None) -> dict:
    prof_id = prof["id"]
    summary = summary_cache.get(prof_id, prof.get("fallback_summary", ""))
    payload = {
        "id": prof_id,
        "name": prof.get("name", "Unknown"),
        "department": prof.get("department", "Computer Science"),
        "subject": prof.get("subject", DEFAULT_SUBJECT),
        "email": prof.get("email", ""),
        "profile_url": prof.get("profile_url", ""),
        "paper_count": len(get_papers(prof)),
        "summary": summary,
        "summary_source": "bedrock" if prof_id in summary_cache else "fallback",
    }
    if similarity is not None:
        payload["similarity"] = round(float(similarity), 1)
    return payload


def maybe_log_query(query_text: str, top_match: str) -> None:
    if not table:
        return
    try:
        table.put_item(
            Item={
                "query_id": str(uuid.uuid4()),
                "query_text": query_text,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "timestamp_unix": int(time.time()),
                "top_match": top_match,
            }
        )
    except Exception as exc:
        print(f"Failed to log query to DynamoDB: {exc}")


def get_embedding(text: str) -> np.ndarray:
    if len(text) > 20000:
        text = text[:20000]
    body = json.dumps({"inputText": text})
    resp = bedrock.invoke_model(
        modelId=EMBED_MODEL,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(resp["body"].read())
    vec = np.array(result["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def call_llm_internal(model_id: str, prompt: str, max_tokens: int) -> str:
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"max_new_tokens": max_tokens, "temperature": 0.5},
        }
    )
    resp = bedrock.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(resp["body"].read())
    return result["output"]["message"]["content"][0]["text"].strip()


def call_llm(prompt: str, max_tokens: int = 1024) -> str:
    if not active_llm_model:
        raise HTTPException(status_code=503, detail="No LLM model available.")
    try:
        return call_llm_internal(active_llm_model, prompt, max_tokens)
    except Exception as exc:
        print(f"[ERROR] call_llm failed with model={active_llm_model}")
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Bedrock AI error: {str(exc)}")


def call_llm_with_fallback(prompt: str, fallback: str, max_tokens: int = 256) -> str:
    try:
        generated = normalize_space(call_llm(prompt, max_tokens=max_tokens))
        if not generated:
            return fallback
        return generated
    except Exception:
        return fallback


def generate_short_summary(prof: dict) -> str:
    prof_id = prof["id"]
    if prof_id in summary_cache:
        return summary_cache[prof_id]

    fallback = prof.get("fallback_summary", build_fallback_summary(prof))
    context = professor_prompt_context(prof, max_papers=2)
    prompt = (
        "Write exactly one concise sentence (max 24 words) that helps a UofT student quickly "
        "understand this professor's research opportunity. Avoid hype and avoid markdown.\n\n"
        f"Professor: {prof.get('name', 'Unknown')}\n"
        f"{context}"
    )
    generated = call_llm_with_fallback(prompt, fallback=fallback, max_tokens=90)
    generated = generated.split("\n")[0].strip("- ").strip()
    generated = truncate_text(generated, 170)
    summary_cache[prof_id] = generated or fallback
    return summary_cache[prof_id]


def generate_professor_description(prof: dict) -> str:
    prof_id = prof["id"]
    if prof_id in description_cache:
        return description_cache[prof_id]

    fallback = (
        "This profile highlights ongoing research directions and recent papers. "
        "Open the paper list to review publication abstracts and key takeaways."
    )
    context = professor_prompt_context(prof, max_papers=3)
    prompt = (
        "You are writing a professor profile for a student research discovery app.\n"
        "Write 2 short paragraphs, 110-150 words total. Include likely research themes and "
        "mention up to two paper topics based only on provided context.\n"
        "Do not invent affiliations or achievements and do not use bullet points.\n\n"
        f"Professor: {prof.get('name', 'Unknown')}\n"
        f"{context}"
    )
    description = call_llm_with_fallback(prompt, fallback=fallback, max_tokens=320)
    description_cache[prof_id] = truncate_text(description, 900)
    return description_cache[prof_id]


def discover_profile_image(prof: dict) -> Optional[str]:
    prof_id = prof["id"]
    if prof_id in image_cache:
        return image_cache[prof_id]

    profile_url = normalize_space(prof.get("profile_url", ""))
    if not profile_url:
        image_cache[prof_id] = None
        return None

    parsed = urlparse(profile_url)
    if parsed.scheme not in {"http", "https"}:
        image_cache[prof_id] = None
        return None

    try:
        resp = requests.get(
            profile_url,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0 PandaDemoBot/1.0"},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        candidates = []
        meta_queries = [
            ("meta", {"property": "og:image"}),
            ("meta", {"name": "og:image"}),
            ("meta", {"name": "twitter:image"}),
            ("meta", {"property": "twitter:image"}),
        ]
        for tag_name, attrs in meta_queries:
            node = soup.find(tag_name, attrs=attrs)
            if node and node.get("content"):
                candidates.append(node["content"])

        if not candidates:
            for img in soup.find_all("img", src=True):
                src = img.get("src", "")
                alt = normalize_space(img.get("alt", "")).lower()
                if "logo" in alt:
                    continue
                if src:
                    candidates.append(src)
                if len(candidates) >= 3:
                    break

        resolved = None
        for candidate in candidates:
            if not candidate:
                continue
            full_url = urljoin(profile_url, candidate.strip())
            parsed_candidate = urlparse(full_url)
            if parsed_candidate.scheme in {"http", "https"}:
                resolved = full_url
                break

        image_cache[prof_id] = resolved
        return resolved
    except Exception:
        image_cache[prof_id] = None
        return None


def parse_limit(raw_limit: Optional[int], default: int = 40) -> int:
    if raw_limit is None:
        return default
    if raw_limit < 1:
        return 1
    return min(raw_limit, len(professors))


def list_subjects_with_counts() -> List[dict]:
    counts = Counter(prof.get("subject", DEFAULT_SUBJECT) for prof in professors)
    sorted_items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    payload = [{"name": ALL_SUBJECTS, "count": len(professors)}]
    payload.extend({"name": name, "count": count} for name, count in sorted_items)
    return payload


def browse_professors(subject: Optional[str], limit: int) -> Tuple[List[dict], int]:
    filtered = [p for p in professors if subject_matches(p, subject)]
    filtered.sort(key=lambda p: p.get("name", "").lower())
    total = len(filtered)
    sliced = filtered[:limit]
    return [build_card_payload(prof) for prof in sliced], total


# -- Startup -----------------------------------------------------------
@app.on_event("startup")
def load_data():
    global professors, embeddings_matrix, bedrock, active_llm_model, table
    print("Loading professor data...")
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        professors = json.load(file)

    for idx, prof in enumerate(professors):
        prof["id"] = str(idx)
        prof["subject"] = infer_subject(prof)
        prof["fallback_summary"] = build_fallback_summary(prof)
        prof["search_blob"] = build_search_blob(prof)

    vecs = [p["embedding"] for p in professors]
    embeddings_matrix = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_matrix = embeddings_matrix / norms

    bedrock = boto3.client("bedrock-runtime", region_name=REGION)
    print(f"Loaded {len(professors)} professors.")

    print("Initializing DynamoDB...")
    try:
        dynamodb = boto3.resource("dynamodb", region_name=REGION)
        existing_tables = list(dynamodb.tables.all())
        if not any(t.name == DYNAMO_TABLE_NAME for t in existing_tables):
            print(f"  Creating table {DYNAMO_TABLE_NAME}...")
            table = dynamodb.create_table(
                TableName=DYNAMO_TABLE_NAME,
                KeySchema=[{"AttributeName": "query_id", "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": "query_id", "AttributeType": "S"}],
                BillingMode="PAY_PER_REQUEST",
            )
            table.meta.client.get_waiter("table_exists").wait(TableName=DYNAMO_TABLE_NAME)
            print("  Table created.")
        else:
            table = dynamodb.Table(DYNAMO_TABLE_NAME)
            print("  Table already exists.")
    except Exception as exc:
        print(f"  Error initializing DynamoDB: {exc}")
        table = None

    for model_id in LLM_MODELS:
        try:
            print(f"  Probing LLM model: {model_id} ...")
            call_llm_internal(model_id, "Reply with exactly: ready", 8)
            active_llm_model = model_id
            print(f"  Using LLM model: {model_id}")
            break
        except Exception as exc:
            print(f"  Model {model_id} failed: {exc}")

    if not active_llm_model:
        print("  WARNING: No LLM model responded. AI generation endpoints will fallback or fail.")
    print("Ready!")


# -- Request Models ----------------------------------------------------
class SearchRequest(BaseModel):
    query: str = ""
    subject: Optional[str] = None
    limit: Optional[int] = 40


class DraftEmailRequest(BaseModel):
    professor_name: str
    professor_interests: str
    student_background: str
    paper_titles: Optional[List[str]] = []


class EmailPointersRequest(BaseModel):
    professor_name: str
    professor_interests: str
    paper_titles: List[str]
    student_background: str


class SummarizeRequest(BaseModel):
    title: str
    abstract: str


class TakeawaysRequest(BaseModel):
    title: str
    abstract: str


# -- Discovery Endpoints ----------------------------------------------
@app.get("/subjects")
def get_subjects():
    return {"subjects": list_subjects_with_counts()}


@app.get("/professors")
def list_professors(
    subject: Optional[str] = Query(default=None),
    sort: str = Query(default="alpha"),
    limit: int = Query(default=85, ge=1),
):
    safe_limit = parse_limit(limit, default=85)
    results, total = browse_professors(subject=subject, limit=safe_limit)

    if sort == "papers":
        results.sort(key=lambda item: item.get("paper_count", 0), reverse=True)

    return {"results": results, "total": total}


@app.post("/search")
def search(req: SearchRequest):
    query = normalize_space(req.query)
    subject = req.subject
    limit = parse_limit(req.limit, default=40)

    if not query:
        browse_results, total = browse_professors(subject=subject, limit=limit)
        return {"mode": "browse", "results": browse_results, "total": total}

    results = []
    try:
        query_vec = get_embedding(query)
        similarities = embeddings_matrix @ query_vec
        ranked_indices = np.argsort(similarities)[::-1]
        for idx in ranked_indices:
            prof = professors[int(idx)]
            if not subject_matches(prof, subject):
                continue
            results.append(build_card_payload(prof, similarity=float(similarities[idx]) * 100))
            if len(results) >= limit:
                break
    except Exception as exc:
        print(f"Embedding search failed, using keyword fallback: {exc}")
        scored = []
        for idx, prof in enumerate(professors):
            if not subject_matches(prof, subject):
                continue
            score = fallback_keyword_similarity(query, prof)
            scored.append((idx, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        for idx, score in scored[:limit]:
            results.append(build_card_payload(professors[idx], similarity=score))

    top_match = results[0]["name"] if results else "None"
    maybe_log_query(query_text=query, top_match=top_match)
    return {"mode": "search", "results": results, "total": len(results)}


@app.get("/professors/{prof_id}/summary")
def get_professor_summary(prof_id: str):
    prof = get_professor_by_id(prof_id)
    summary = generate_short_summary(prof)
    return {"id": prof_id, "summary": summary}


@app.get("/professors/{prof_id}")
def get_professor_detail(prof_id: str):
    prof = get_professor_by_id(prof_id)
    description = generate_professor_description(prof)
    image_url = discover_profile_image(prof)
    summary = summary_cache.get(prof_id, prof.get("fallback_summary", ""))
    papers = get_papers(prof)

    return {
        "id": prof["id"],
        "name": prof.get("name", "Unknown"),
        "department": prof.get("department", "Computer Science"),
        "subject": prof.get("subject", DEFAULT_SUBJECT),
        "email": prof.get("email", ""),
        "profile_url": prof.get("profile_url", ""),
        "image_url": image_url,
        "summary": summary,
        "description": description,
        "papers": papers,
    }


# -- Outreach & Paper Intelligence ------------------------------------
@app.post("/draft-email")
def draft_email(req: DraftEmailRequest):
    papers_context = ""
    if req.paper_titles:
        papers_context = f" Some relevant papers include: {', '.join(req.paper_titles[:3])}."

    prompt = (
        f"Write a concise and professional cold email from a University of Toronto student to "
        f"Professor {req.professor_name}. The professor's research context is: {req.professor_interests}.{papers_context} "
        f"The student's background is: {req.student_background}. Keep it under 170 words. "
        f"Include a useful subject line as the first line."
    )
    draft = call_llm(prompt, max_tokens=420)
    return {"draft": draft}


@app.post("/email-pointers")
def email_pointers(req: EmailPointersRequest):
    papers_str = "\n".join(f"- {title}" for title in req.paper_titles[:6]) if req.paper_titles else "No papers listed."
    prompt = (
        f"A UofT student wants to email Professor {req.professor_name} for research opportunities.\n\n"
        f"Professor research context: {req.professor_interests}\n\n"
        f"Professor paper titles:\n{papers_str}\n\n"
        f"Student background:\n{req.student_background}\n\n"
        f"Generate 5 practical bullet points with these labels:\n"
        f"- Overlap\n"
        f"- Paper to mention\n"
        f"- Unique angle\n"
        f"- Credibility signal\n"
        f"- Specific ask\n"
        f"Be specific and concrete."
    )
    tips = call_llm(prompt, max_tokens=520)
    return {"pointers": tips}


@app.post("/paper-takeaways")
def paper_takeaways(req: TakeawaysRequest):
    if not normalize_space(req.abstract):
        return {"takeaways": "- No abstract available for this paper.\n- Open the source link for full context.\n- Mention this paper title when reaching out."}

    prompt = (
        "You are helping undergraduate students understand research papers.\n"
        "Return exactly 3 bullet points with concrete key takeaways from this abstract.\n"
        "Then add one line starting with 'Why it matters:' in plain language.\n"
        "Avoid markdown headings.\n\n"
        f"Title: {req.title}\n"
        f"Abstract: {req.abstract}"
    )
    fallback = (
        "- This paper introduces a concrete method tied to the stated research problem.\n"
        "- It reports results that can inform follow-up research questions.\n"
        "- The abstract suggests practical implications for related work.\n"
        "Why it matters: You can reference this paper to show informed interest in the professor's work."
    )
    result = call_llm_with_fallback(prompt, fallback=fallback, max_tokens=320)
    return {"takeaways": result}


@app.post("/summarize-paper")
def summarize_paper(req: SummarizeRequest):
    if not normalize_space(req.abstract):
        return {"summary": "No abstract available for this paper."}
    prompt = (
        "Summarize the following paper in exactly 2 plain-English sentences for a university student.\n\n"
        f"Title: {req.title}\n"
        f"Abstract: {req.abstract}"
    )
    summary = call_llm(prompt, max_tokens=200)
    return {"summary": summary}


@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    content = await file.read()

    text = ""
    if file.filename and file.filename.lower().endswith(".pdf"):
        try:
            import pdfplumber

            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception:
            try:
                from PyPDF2 import PdfReader

                reader = PdfReader(io.BytesIO(content))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception:
                text = content.decode("utf-8", errors="ignore")
    else:
        text = content.decode("utf-8", errors="ignore")

    text = normalize_space(text)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from the uploaded file.")
    if len(text) > 15000:
        text = text[:15000]

    prompt = (
        "You are reading a student's resume. Return 5 concise bullet points covering:\n"
        "1) research interests\n2) technical skills\n3) academic background\n"
        "4) relevant projects/experience\n5) strongest evidence of fit for research outreach.\n"
        "Keep total output under 150 words and use only resume evidence.\n\n"
        f"Resume text:\n{text}"
    )
    summary = call_llm(prompt, max_tokens=420)
    return {"summary": summary}


# -- Analytics ---------------------------------------------------------
@app.get("/analytics")
def get_analytics():
    if not table:
        return {"queries": [], "error": "DynamoDB not initialized."}
    try:
        response = table.scan(Limit=100)
        items = response.get("Items", [])
        items.sort(key=lambda x: x.get("timestamp_unix", 0), reverse=True)
        return {"queries": items[:20]}
    except Exception as exc:
        print(f"Error fetching analytics: {exc}")
        return {"queries": [], "error": str(exc)}


# -- Frontend ----------------------------------------------------------
@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

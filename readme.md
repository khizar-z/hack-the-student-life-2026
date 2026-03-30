# Panda - UofT Research Discovery
<img width="806" height="459" alt="image" src="https://github.com/user-attachments/assets/0db0b93a-469b-4774-a2c2-5b4819d4b123" />

Panda is a hackathon web app that helps UofT students discover professors for research opportunities, understand paper context quickly, and get guidance for outreach.

This project targets **Hack The Student Life 2026 - Challenge Area 3**:
- Discovery of expertise
- Network visibility
- Personalized matching
- Community engagement

## What the app does
- Landing page search for research topics
- Browse page with subject filters and scrollable professor cards
- Professor detail page with profile info and paper list
- Paper popup with abstract + AI key takeaways
- Reach-out popup with resume-aware outreach pointers
- Dummy login button (no auth flow)

## Current stack
- Backend: FastAPI (`main.py`)
- Frontend: single-page HTML/CSS/JS (`index.html`)
- Data: `professors_embedded.json` (85 demo CS professors)
- Retrieval: NumPy cosine similarity over Titan embeddings
- Cloud: AWS Bedrock Runtime, DynamoDB

## AWS services used
- Amazon Bedrock Runtime
  - Titan Embeddings: `amazon.titan-embed-text-v1`
  - Nova LLMs (auto-probed): `us.amazon.nova-lite-v1:0`, `us.amazon.nova-pro-v1:0`, `us.amazon.nova-micro-v1:0`
- Amazon DynamoDB
  - Table: `ResearchConnectQueries`

## Repository files
- `main.py` - API + app runtime logic
- `index.html` - UI and client-side state/flow
- `scrape.py` - faculty + Semantic Scholar paper collection
- `embed.py` - embedding generation via Bedrock Titan
- `professors.json` - scraped/enriched profiles
- `professors_embedded.json` - profiles + embedding vectors

## Prerequisites
- Python 3.10+
- pip
- AWS credentials configured for your shell/user
- Bedrock model access enabled in your AWS account/region

## Quick start
Run from project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open:
- http://127.0.0.1:8000

## Regenerate dataset (optional)
If you want fresh faculty/paper data and vectors:

```bash
python scrape.py
python embed.py
```

Then restart the app:

```bash
uvicorn main:app --reload
```

Notes:
- `scrape.py` now attempts to fetch **all available papers** per matched author (paginated).
- `build_embed_text` intentionally caps papers included in embeddings for performance, while full paper lists remain in stored profile data.

## Main API endpoints

### Discovery
- `GET /subjects` - subject filters with counts
- `GET /professors` - browse list (supports subject/sort/limit)
- `POST /search` - semantic search (query + optional subject filter)
- `GET /professors/{id}/summary` - short card summary
- `GET /professors/{id}` - full professor detail

### Outreach and paper intelligence
- `POST /parse-resume` - parse uploaded resume into student summary
- `POST /email-pointers` - outreach pointers based on student + professor context
- `POST /draft-email` - generate a full cold email draft
- `POST /paper-takeaways` - key takeaways from abstract
- `POST /summarize-paper` - concise 2-sentence summary

### Analytics
- `GET /analytics` - recent search activity from DynamoDB

## Runtime behavior and fallbacks
- If AWS embeddings are available, search uses Titan + cosine similarity.
- If AWS embeddings fail (for example, missing credentials), search falls back to keyword scoring.
- If LLM generation fails, selected endpoints return safe fallback text.
- If profile image discovery fails, UI uses a generated avatar.

## Troubleshooting
- **`Unable to locate credentials`**
  - Configure AWS credentials/profile and retry.
- **AI outputs missing**
  - Confirm Bedrock model access and region (`us-west-2`).
- **No new paper count after scraper updates**
  - Re-run `python scrape.py` and `python embed.py`, then restart server.


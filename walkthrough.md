# Panda - Technical Walkthrough and Demo Guide (Hack The Student Life 2026)

## 1. What Panda Does
Panda is a web platform that helps UofT students discover professors for research opportunities, understand professor research quickly, and get personalized guidance on how to reach out.

Current demo scope:
- Dataset: 85 UofT CS professors (demo set)
- Core UX:
1. Landing search
2. Browse/search results with subject filters
3. Professor detail view with papers
4. Reach-out guidance popup (resume-aware)
5. Paper takeaways popup

This directly targets **Challenge Area 3**: strengthening academic and career connections across UofT by improving:
- Discovery of expertise
- Network visibility
- Personalized matching
- Community engagement

---

## 2. Judging Alignment (What We Built for the Rubric)
From the participant package, the judging categories are:
- Innovation and Relevance (10)
- Technical Complexity (15)
- Working Demo (10)
- Presentation (5)
- Feasibility and Effectiveness (10)

How Panda maps to those:
- Innovation/Relevance: combines semantic retrieval + generative guidance in one workflow from discovery to outreach.
- Technical Complexity: multi-stage pipeline (scraping, enrichment, embedding, retrieval, LLM generation, caching, analytics).
- Working Demo: complete end-to-end flow in one app with fail-safes when credentials or data are missing.
- Presentation: clear 3-screen flow mirrors the user journey.
- Feasibility/Effectiveness: practical AWS-first architecture, small enough to deploy quickly, useful for real student pain points.

---

## 3. System Architecture (End-to-End)

### Offline pipeline
1. `scrape.py`
- Scrapes UofT CS faculty directory
- Pulls name, email, profile URL, research fields
- Enriches with Semantic Scholar papers
- Builds `embed_text`
- Writes `professors.json`

2. `embed.py`
- Reads `professors.json`
- Calls Bedrock Titan Embeddings (`amazon.titan-embed-text-v1`)
- Stores vector per professor
- Writes `professors_embedded.json`

### Online app runtime
3. `main.py` (FastAPI)
- Loads professor data + vectors
- Normalizes embedding matrix for cosine similarity
- Probes available Bedrock LLM model (Nova)
- Serves API endpoints + frontend
- Optionally logs searches to DynamoDB

4. `index.html` (SPA frontend)
- 3-view app: Home -> Discover -> Profile
- Calls backend APIs for search, summaries, details, pointers, takeaways
- Shows modals for outreach and paper intelligence

---

## 4. Data Pipeline Details

### 4.1 Scraping and enrichment (`scrape.py`)
- Faculty source: `https://web.cs.toronto.edu/people/faculty-directory`
- Table parsing via BeautifulSoup
- Extracted fields:
  - `name`
  - `email`
  - `department`
  - `profile_url`
  - `research_areas`
  - `research_interests`
- Semantic Scholar author search enriches each professor with top papers:
  - `title`
  - `abstract`
- Builds `embed_text` by concatenating name + research text + papers

### 4.2 Embedding generation (`embed.py`)
- Model: Titan embed text v1
- One vector per professor
- Final artifact: `professors_embedded.json` with:
  - metadata
  - papers
  - `embedding` vector

---

## 5. Backend Technical Breakdown (`main.py`)

## 5.1 Startup sequence
On app startup:
1. Load `professors_embedded.json`
2. Assign internal `id` to each professor
3. Infer subject category from text keywords
4. Build fallback summary and search blob
5. Build normalized NumPy matrix for cosine similarity
6. Initialize Bedrock client
7. Initialize DynamoDB table handle (`ResearchConnectQueries`)
8. Probe available Nova model in priority order:
   - `us.amazon.nova-lite-v1:0`
   - `us.amazon.nova-pro-v1:0`
   - `us.amazon.nova-micro-v1:0`

## 5.2 Search and ranking logic
### Primary path (AWS available)
- Query -> Titan embedding
- Cosine similarity vs professor matrix
- Sort descending and return top N

### Fallback path (AWS unavailable)
- Keyword token scoring over prebuilt `search_blob`
- Ensures demo still functions if credentials are unavailable

### Why this matters
- You always get a working search experience
- With AWS enabled, matching is semantic, not just keyword-based

## 5.3 Subject filtering
- Subjects are inferred using weighted keyword dictionaries over each professor's text and paper metadata
- Supported labels include:
  - AI & Machine Learning
  - Systems & Networking
  - Security & Privacy
  - Theory & Algorithms
  - HCI, Robotics & Interaction
  - Data Science & Computational Biology
  - Vision, Graphics & Media
  - Software Engineering

## 5.4 AI-generated content and caching
### Per-card short summaries
- Endpoint: `GET /professors/{id}/summary`
- Generated once per professor (cached in memory)
- One concise sentence for browsing

### Professor detail descriptions
- Endpoint: `GET /professors/{id}`
- 2 short paragraphs generated from paper context
- Cached for reuse

### Paper takeaways
- Endpoint: `POST /paper-takeaways`
- Returns:
  - 3 key takeaways
  - 1 "Why it matters" line

### Reach-out guidance
- Endpoint: `POST /email-pointers`
- Combines:
  - professor research context
  - paper titles
  - student background / resume summary
- Returns practical bullets for cold outreach strategy

### Resume parsing
- Endpoint: `POST /parse-resume`
- Accepts PDF/text
- Extracts concise structured student summary via LLM

## 5.5 Profile image discovery
- On professor detail request:
  - fetch profile page
  - inspect `og:image` / `twitter:image` / first relevant `<img>`
  - cache result
- Frontend fallback: generated avatar if no image found

## 5.6 Analytics logging
- Endpoint: `GET /analytics`
- Search logs (when table available):
  - query id
  - query text
  - timestamp
  - top match
- Stored in DynamoDB for trend visibility

---

## 6. Frontend Technical Breakdown (`index.html`)

## 6.1 View model
Single-page app with three views:
1. Home (`homeView`)
- Hero + search
- Browse button
- Dummy login button

2. Discover (`discoverView`)
- Left sidebar: subject filter chips
- Top controls: inline search
- Main panel: scrollable opportunity cards

3. Profile (`profileView`)
- Left: professor info, image, description, reach-out action
- Right: scrollable paper list
- Two modals:
  - Reach-out guidance modal
  - Paper takeaways modal

## 6.2 State management
Client-side `state` object tracks:
- current view
- query
- selected subject
- loaded subjects
- results
- selected professor
- parsed resume summary
- in-flight summary requests

## 6.3 UI behavior flow
### Landing -> search flow
- User submits query from home
- `POST /search`
- Render cards + similarity badges
- Lazy hydrate summary text on top visible cards

### Browse flow
- User clicks Browse
- `GET /professors?sort=alpha&limit=85`
- Render alphabetical list

### Filter flow
- User picks subject chip
- Re-runs search or browse with subject constraint

### Card -> profile flow
- User clicks card
- `GET /professors/{id}`
- Render description + papers + image

### Reach out flow
- Optional resume upload (`POST /parse-resume`)
- Generate pointers (`POST /email-pointers`)

### Paper flow
- User clicks paper
- Show abstract
- Generate key takeaways (`POST /paper-takeaways`)

---

## 7. API Reference (Current)

### Discovery
- `GET /subjects`
  - returns subject names + counts
- `GET /professors?subject=&sort=&limit=`
  - browse mode (alpha by default)
- `POST /search`
  - body: `{ query, subject, limit }`
  - semantic ranking (with keyword fallback)
- `GET /professors/{id}/summary`
  - one-line card summary
- `GET /professors/{id}`
  - full profile detail

### Outreach and papers
- `POST /parse-resume`
  - multipart file upload
- `POST /email-pointers`
  - outreach strategy bullets
- `POST /draft-email`
  - full draft email (available if you choose to demo)
- `POST /paper-takeaways`
  - key takeaways + why it matters
- `POST /summarize-paper`
  - 2-sentence summary endpoint (legacy/extra)

### Analytics
- `GET /analytics`
  - recent query entries from DynamoDB

---

## 8. AWS Services Used
- Amazon Bedrock Titan Embeddings
  - semantic vector generation for professor retrieval
- Amazon Bedrock Nova LLM
  - summaries, descriptions, pointers, takeaways, resume parsing
- Amazon DynamoDB
  - query logging and analytics
- Boto3
  - service integrations in Python

---

## 13. Runbook (Local)

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Generate data (if needed)
```bash
python scrape.py
python embed.py
```

3. Run app
```bash
uvicorn main:app --reload
```

4. Open
- `http://127.0.0.1:8000`

---

## 14. Suggested Final 1-Line Pitch
"Panda is an AWS-powered UofT research discovery platform that helps students find the right professors faster and confidently turn that match into high-quality outreach."


"""
scrape.py — UofT Research Connect Data Pipeline

Scrapes the UofT CS department faculty directory and enriches
each professor's profile with their top papers from Semantic Scholar.
Outputs professors.json.
"""

import json
import time
import requests
from bs4 import BeautifulSoup

FACULTY_URL = "https://web.cs.toronto.edu/people/faculty-directory"
SEMANTIC_SCHOLAR_AUTHOR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/author/search"
SEMANTIC_SCHOLAR_AUTHOR_PAPERS_URL = "https://api.semanticscholar.org/graph/v1/author/{author_id}/papers"

def scrape_faculty():
    """Scrape the UofT CS faculty directory page."""
    print(f"[1/3] Fetching faculty page: {FACULTY_URL}")
    resp = requests.get(FACULTY_URL, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    table = soup.find("table", class_="blueTable")
    if not table:
        # Try finding any table on the page
        table = soup.find("table")
    if not table:
        print("ERROR: Could not find faculty table on the page.")
        return []

    rows = table.find_all("tr")
    professors = []

    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 3:
            continue

        # Column 1: Name (inside <a> tag)
        name_tag = cols[0].find("a")
        name = name_tag.get_text(strip=True) if name_tag else cols[0].get_text(strip=True)
        if not name:
            continue

        profile_url = ""
        if name_tag and name_tag.get("href"):
            href = name_tag["href"]
            if href.startswith("http"):
                profile_url = href
            elif href.startswith("/"):
                profile_url = "https://web.cs.toronto.edu" + href

        # Column 2: Email (inside <a> with mailto:)
        email_tag = cols[1].find("a", href=lambda h: h and h.startswith("mailto:"))
        email = ""
        if email_tag:
            email = email_tag["href"].replace("mailto:", "").strip()
        else:
            email = cols[1].get_text(strip=True)

        # Column 3: Research Areas + Research Interests
        info_cell = cols[2]
        info_text = info_cell.get_text(separator="\n", strip=True)

        research_areas = ""
        research_interests = ""

        for line in info_text.split("\n"):
            line = line.strip()
            if line.lower().startswith("research areas:"):
                research_areas = line.split(":", 1)[1].strip()
            elif line.lower().startswith("research interests:"):
                research_interests = line.split(":", 1)[1].strip()

        professors.append({
            "name": name,
            "email": email,
            "department": "Computer Science",
            "profile_url": profile_url,
            "research_areas": research_areas,
            "research_interests": research_interests,
            "papers": [],
            "embed_text": ""
        })

    print(f"    Found {len(professors)} professors.")
    return professors


def _request_with_rate_limit(url, params, retries=2, timeout=20):
    """Request helper with basic 429 retry handling."""
    for attempt in range(retries + 1):
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code != 429:
            return resp
        wait_seconds = 2 + attempt * 2
        print(f"      Rate limited, waiting {wait_seconds}s...")
        time.sleep(wait_seconds)
    return resp


def fetch_papers(professor_name):
    """Query Semantic Scholar for all available papers by a professor."""
    params = {
        "query": professor_name,
        "fields": "name,authorId,paperCount",
        "limit": 10
    }
    try:
        resp = _request_with_rate_limit(SEMANTIC_SCHOLAR_AUTHOR_SEARCH_URL, params=params, timeout=15)
        if resp.status_code != 200:
            return []

        data = resp.json()
        authors = data.get("data", [])
        if not authors:
            return []

        # Try to find the best-matching author
        best_author = None
        for author in authors:
            author_name = author.get("name", "").lower()
            prof_name = professor_name.lower()
            # Check if the professor name is a close match
            prof_parts = prof_name.split()
            if all(part in author_name for part in prof_parts):
                best_author = author
                break
        
        if not best_author:
            best_author = authors[0]

        author_id = best_author.get("authorId")
        if not author_id:
            return []

        papers = []
        offset = 0
        page_size = 100
        seen_titles = set()

        while True:
            page_resp = _request_with_rate_limit(
                SEMANTIC_SCHOLAR_AUTHOR_PAPERS_URL.format(author_id=author_id),
                params={
                    "fields": "title,abstract",
                    "limit": page_size,
                    "offset": offset
                },
                timeout=20
            )
            if page_resp.status_code != 200:
                break

            page_data = page_resp.json().get("data", [])
            if not page_data:
                break

            for paper in page_data:
                title = (paper.get("title") or "").strip()
                if not title:
                    continue
                title_key = title.lower()
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)
                papers.append({
                    "title": title,
                    "abstract": (paper.get("abstract") or "").strip()
                })

            if len(page_data) < page_size:
                break

            offset += page_size
            time.sleep(0.2)

        papers.sort(key=lambda item: item["title"].lower())
        result = []
        for paper in papers:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            result.append({
                "title": title,
                "abstract": abstract or ""
            })
        return result
    except Exception as e:
        print(f"      Error fetching papers: {e}")
        return []


def build_embed_text(prof):
    """Build embedding text from profile data (paper subset for efficiency)."""
    parts = [prof["name"]]
    if prof["research_areas"]:
        parts.append(f"Research Areas: {prof['research_areas']}")
    if prof["research_interests"]:
        parts.append(f"Research Interests: {prof['research_interests']}")
    # Keep embed payload bounded while keeping full papers list for UI.
    for paper in prof["papers"][:20]:
        parts.append(f"Paper: {paper['title']}")
        if paper["abstract"]:
            parts.append(f"Abstract: {paper['abstract']}")
    return "\n".join(parts)


def main():
    professors = scrape_faculty()
    if not professors:
        print("No professors found. Exiting.")
        return

    print(f"\n[2/3] Fetching papers from Semantic Scholar...")
    for i, prof in enumerate(professors):
        print(f"    [{i+1}/{len(professors)}] {prof['name']}...")
        prof["papers"] = fetch_papers(prof["name"])
        prof["embed_text"] = build_embed_text(prof)
        # Be polite to the API
        time.sleep(0.5)

    print(f"\n[3/3] Saving to professors.json...")
    with open("professors.json", "w", encoding="utf-8") as f:
        json.dump(professors, f, indent=2, ensure_ascii=False)

    print(f"Done! Saved {len(professors)} professors to professors.json")


if __name__ == "__main__":
    main()

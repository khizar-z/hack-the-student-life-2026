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
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/author/search"

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


def fetch_papers(professor_name, max_papers=3):
    """Query Semantic Scholar for a professor's papers."""
    params = {
        "query": professor_name,
        "fields": "name,papers.title,papers.abstract",
        "limit": 1
    }
    try:
        resp = requests.get(SEMANTIC_SCHOLAR_URL, params=params, timeout=15)
        if resp.status_code == 429:
            print(f"      Rate limited, waiting 3s...")
            time.sleep(3)
            resp = requests.get(SEMANTIC_SCHOLAR_URL, params=params, timeout=15)
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

        papers = best_author.get("papers", [])
        result = []
        count = 0
        for paper in papers:
            if count >= max_papers:
                break
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            if title:
                result.append({
                    "title": title,
                    "abstract": abstract or ""
                })
                count += 1
        return result
    except Exception as e:
        print(f"      Error fetching papers: {e}")
        return []


def build_embed_text(prof):
    """Build the embedding text from professor data."""
    parts = [prof["name"]]
    if prof["research_areas"]:
        parts.append(f"Research Areas: {prof['research_areas']}")
    if prof["research_interests"]:
        parts.append(f"Research Interests: {prof['research_interests']}")
    for paper in prof["papers"]:
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

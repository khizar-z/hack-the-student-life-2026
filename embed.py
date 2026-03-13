"""
embed.py — UofT Research Connect Embeddings

Reads professors.json, calls AWS Bedrock Titan Embeddings on each
professor's embed_text, and saves the results to professors_embedded.json.
"""

import json
import boto3

REGION = "us-west-2"
MODEL_ID = "amazon.titan-embed-text-v1"


def get_embedding(client, text):
    """Get embedding vector from AWS Bedrock Titan."""
    # Titan has a max input of ~8000 tokens, truncate if needed
    if len(text) > 20000:
        text = text[:20000]

    body = json.dumps({"inputText": text})
    response = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def main():
    print("[1/3] Loading professors.json...")
    with open("professors.json", "r", encoding="utf-8") as f:
        professors = json.load(f)

    print(f"    Loaded {len(professors)} professors.")

    print("[2/3] Generating embeddings via AWS Bedrock Titan...")
    client = boto3.client("bedrock-runtime", region_name=REGION)

    for i, prof in enumerate(professors):
        embed_text = prof.get("embed_text", prof["name"])
        if not embed_text.strip():
            embed_text = prof["name"]
        print(f"    [{i+1}/{len(professors)}] {prof['name']}...")
        prof["embedding"] = get_embedding(client, embed_text)

    print("[3/3] Saving to professors_embedded.json...")
    with open("professors_embedded.json", "w", encoding="utf-8") as f:
        json.dump(professors, f, indent=2, ensure_ascii=False)

    print(f"Done! Saved {len(professors)} professors with embeddings.")


if __name__ == "__main__":
    main()

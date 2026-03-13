from dotenv import load_dotenv
import os

print("Before dotenv:", "bedrock-key length:", len(os.environ.get("OPENAI_API_KEY", "")))

load_dotenv(override=True)
key = os.environ.get("OPENAI_API_KEY", "")
print("After dotenv:", "bedrock-key length:", len(key))
print("Ends with:", key[-20:])

from dotenv import load_dotenv
import os

load_dotenv()
key = os.environ.get("OPENAI_API_KEY")
print("Key length:", len(key) if key else None)
print("Key ends with:", key[-20:] if key else None)
if key:
    print("Does key have quotes?", key.startswith('"') or key.endswith('"'))


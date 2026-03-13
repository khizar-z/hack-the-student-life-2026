import os
from openai import OpenAI

# Check environment variables
print("OPENAI_API_KEY:", "***" if os.environ.get("OPENAI_API_KEY") else "Not Set")
print("OPENAI_BASE_URL:", os.environ.get("OPENAI_BASE_URL"))

try:
    client = OpenAI()
    response = client.chat.completions.create(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[{"role": "user", "content": "Say hello!"}],
        max_tokens=10
    )
    print("Success!")
    print(response.choices[0].message.content)
except Exception as e:
    print("Error:", e)

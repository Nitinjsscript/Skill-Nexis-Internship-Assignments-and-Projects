import os
from google import genai

# Client picks up GEMINI_API_KEY automatically from environment
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say 'Environment setup successful!'",
)

print(response.text)
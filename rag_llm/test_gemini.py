import os
from dotenv import load_dotenv
from google import genai

# I load a local, ignored .env file while preserving variables already set by the process.
load_dotenv()

# The SDK reads GOOGLE_API_KEY from the environment; this check sends a live request.
client = genai.Client()

# A short fixed prompt verifies that model creation and response generation both succeed.
chat = client.chats.create(model="gemini-3.5-flash-lite")

response = chat.send_message("Say 'Hello! Gemini is fully working!' in 5 words or less.")

print("\n--- GEMINI RESPONSE ---")
print(response.text)
print("-----------------------\n")

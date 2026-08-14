import os
from dotenv import load_dotenv
from google import genai

# Load API key from .env
load_dotenv()

# Initialize Client
client = genai.Client()

#  available free model
chat = client.chats.create(model="gemini-3.5-flash-lite")

# test message
response = chat.send_message("Say 'Hello! Gemini is fully working!' in 5 words or less.")

print("\n--- GEMINI RESPONSE ---")
print(response.text)
print("-----------------------\n")

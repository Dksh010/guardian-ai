import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()

chat = client.chats.create(model="gemini-3.5-flash-lite")

response = chat.send_message("Say 'Hello! Gemini is fully working!' in 5 words or less.")

print("\n--- GEMINI RESPONSE ---")
print(response.text)
print("-----------------------\n")

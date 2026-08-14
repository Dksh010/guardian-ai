# engine.py

import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

from vector_db import initialize_vector_store, query_top_rules
from chat_buffer import ChatBuffer
from prompts import SYSTEM_INSTRUCTION, SafetyAuditResult, build_audit_prompt

# Load environment variables from .env
load_dotenv()

class GuardianEngine:
    def __init__(self, model_name="gemini-3.5-flash-lite"):
        print("Initializing Guardian AI Engine...")
        
        # 1. Initialize Gemini Client
        self.client = genai.Client()
        self.model_name = model_name

        # 2. Initialize ChromaDB Vector Store
        self.vector_store = initialize_vector_store()

        # 3. Initialize Chat Buffer (retains last 5 messages)
        self.buffer = ChatBuffer(max_history=5)
        
        print("Guardian AI Engine ready!\n")

    def audit_message(self, sender: str, text: str) -> dict:
        """
        Main Pipeline:
        1. Query ChromaDB for top-2 relevant safety policies.
        2. Retrieve sliding window context from the buffer.
        3. Append target message to the buffer.
        4. Send combined context block to Gemini for structured JSON audit.
        """
        # Step A: Retrieve Top-2 relevant rules for broader semantic coverage
        matched_rules = query_top_rules(self.vector_store, text, n_results=2)
        
        # Format candidate rules into a unified prompt block
        retrieved_rules_text = "\n\n".join([r["content"] for r in matched_rules])
        primary_match = matched_rules[0]  # Closest vector match metadata

        # Step B: Get sliding window context BEFORE pushing current message
        chat_context = self.buffer.get_formatted_context()

        # Step C: Append current message to sliding window history
        self.buffer.add_message(sender, text)

        # Step D: Construct multi-context audit prompt
        prompt = build_audit_prompt(
            chat_context=chat_context,
            target_message=f"{sender}: {text}",
            retrieved_rule=retrieved_rules_text
        )

        # Step E: Configure Structured JSON Schema & Temperature
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=SafetyAuditResult,
            temperature=0.1  # Low temperature for deterministic evaluation
        )

        # Step F: Issue audit request via Gemini Interactions API
        chat = self.client.chats.create(
            model=self.model_name,
            config=config
        )
        response = chat.send_message(prompt)

        # Step G: Parse strict Pydantic-enforced JSON output
        audit_data = json.loads(response.text)
        
        # Attach primary vector match metadata for UI debugging/display
        audit_data["retrieved_rule_id"] = primary_match["rule_id"]
        audit_data["retrieved_rule_category"] = primary_match["category"]

        return audit_data

    def reset_chat(self):
        """Resets the sliding window chat buffer."""
        self.buffer.clear()


if __name__ == "__main__":
    # Quick execution sanity test
    engine = GuardianEngine()

    print("--- TESTING ENHANCED ENGINE AUDIT ---")
    test_result = engine.audit_message("Player_2", "shut THE FUCK UP everyone!")
    print(json.dumps(test_result, indent=2))
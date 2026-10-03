import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

from vector_db import initialize_vector_store, query_top_rules
from chat_buffer import ChatBuffer
from prompts import SYSTEM_INSTRUCTION, SafetyAuditResult, build_audit_prompt

load_dotenv()

class GuardianEngine:
    """I coordinate policy retrieval, rolling context, and structured Gemini audits."""
    def __init__(self, model_name="gemini-3.5-flash-lite"):
        """I initialize the Gemini client and in-memory policy/context state."""
        print("Initializing Guardian AI Engine...")
        
        # The Google SDK reads GOOGLE_API_KEY from the process environment or loaded .env.
        self.client = genai.Client()
        # Callers can override the default model while keeping the same response schema.
        self.model_name = model_name

        # The store and buffer are per-engine instances so each lobby has isolated context.
        self.vector_store = initialize_vector_store()

        self.buffer = ChatBuffer(max_history=5)
        
        print("Guardian AI Engine ready!\n")

    def audit_message(self, sender: str, text: str) -> dict:
        """I retrieve relevant rules and request a schema-constrained audit for one message."""
        matched_rules = query_top_rules(self.vector_store, text, n_results=2)
        
        # I pass both retrieved candidates to the prompt but attach the first as display metadata.
        retrieved_rules_text = "\n\n".join([r["content"] for r in matched_rules])
        # query_top_rules guarantees at least a matching policy or its clean-play sentinel.
        primary_match = matched_rules[0]

        # I snapshot earlier messages before appending the target so it appears only in its
        # dedicated prompt field rather than being duplicated in the preceding transcript.
        chat_context = self.buffer.get_formatted_context()

        self.buffer.add_message(sender, text)

        prompt = build_audit_prompt(
            chat_context=chat_context,
            target_message=f"{sender}: {text}",
            retrieved_rule=retrieved_rules_text
        )

        # JSON mode and the Pydantic schema constrain the response to the dashboard's fields.
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=SafetyAuditResult,
            temperature=0.1
        )

        chat = self.client.chats.create(
            model=self.model_name,
            config=config
        )
        response = chat.send_message(prompt)

        # The engine expects valid JSON; parse errors remain visible to the caller.
        audit_data = json.loads(response.text)
        
        audit_data["retrieved_rule_id"] = primary_match["rule_id"]
        audit_data["retrieved_rule_category"] = primary_match["category"]

        return audit_data

    def reset_chat(self):
        """I clear prior messages so the next audit starts a fresh lobby context."""
        self.buffer.clear()


if __name__ == "__main__":
    engine = GuardianEngine()

    print("--- TESTING ENHANCED ENGINE AUDIT ---")
    test_result = engine.audit_message("Player_2", "shut THE FUCK UP everyone!")
    print(json.dumps(test_result, indent=2))
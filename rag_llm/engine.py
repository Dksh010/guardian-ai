import json
import os
import re
import warnings
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
    from .vector_db import initialize_vector_store, query_top_rules
    from .chat_buffer import ChatBuffer
    from .prompts import SYSTEM_INSTRUCTION, SafetyAuditResult, build_audit_prompt
    from .safety_policies import SAFETY_RULES
except ImportError:
    from vector_db import initialize_vector_store, query_top_rules
    from chat_buffer import ChatBuffer
    from prompts import SYSTEM_INSTRUCTION, SafetyAuditResult, build_audit_prompt
    from safety_policies import SAFETY_RULES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _classify_without_gemini(text, primary_rule):
    compact = re.sub(r"[^a-z0-9]", "", text.lower())
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    normalized = normalized.translate(
        str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})
    )

    if (
        "kys" in compact
        or re.search(r"\b(kill yourself|kill urself|end yourself|go die|get cancer and die)\b", normalized)
        or re.search(r"\b(subhuman|racial slur|hate speech)\b", normalized)
    ):
        label = 2
        severity = "CRITICAL"
        action = "Permanent Permaban & Escalate"
        reasoning = "The message contains severe abuse or self-harm encouragement."
    elif re.search(r"\b(diff|gap|gg ez|trash|uninstall|noob|useless)\b", normalized):
        label = 1
        severity = "MEDIUM"
        action = "Lobby Mute"
        reasoning = "The message contains a gameplay insult or mild competitive taunt."
    else:
        label = int(primary_rule.get("label", 0))
        if label == 2:
            severity, action = "HIGH", "Temporary Ban"
            reasoning = "The retrieved community policy marks this message as a severe violation."
        elif label == 1:
            severity, action = "LOW", "Lobby Mute"
            reasoning = "The retrieved community policy marks this message as mild toxicity."
        else:
            severity, action = "NONE", "No Action"
            reasoning = "The message appears to be clean play under the retrieved community policy."

    rule_id = primary_rule.get("rule_id", "NONE")
    category = primary_rule.get("category", "Standard Clean Conversation")
    target_group = (
        "Kids" if re.search(r"\b(child|kid|minor|school)\b", normalized)
        else "Women / Gender Minorities" if re.search(r"\b(woman|women|girl|gender)\b", normalized)
        else "General Lobby" if label
        else "None"
    )
    return {
        "predicted_label": label,
        "confidence": 0.70,
        "applied_rule_id": rule_id,
        "retrieved_rule_id": rule_id,
        "retrieved_rule_category": category,
        "reasoning": reasoning,
        "is_flagged": label > 0,
        "severity": severity,
        "violated_rule": f"{rule_id}: {category}" if label else "None",
        "target_group": target_group,
        "summary_reasoning": reasoning,
        "recommended_action": action,
    }


class GuardianEngine:
    def __init__(
        self,
        model_name="gemini-3.5-flash-lite",
        vector_store=None,
        initialize_store=True,
    ):
        self.model_name = model_name
        self.buffer = ChatBuffer(max_history=5)
        self.client = None
        self.vector_store = vector_store

        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if api_key:
            try:
                self.client = genai.Client(api_key=api_key)
            except Exception as exc:
                warnings.warn(
                    f"Gemini client setup failed ({type(exc).__name__}: {exc}); "
                    "local policy classification will be used.",
                    RuntimeWarning,
                )
        else:
            print("GOOGLE_API_KEY is not set. Gemini is disabled; local policy fallback is active.")

        if self.vector_store is None and initialize_store:
            try:
                self.vector_store = initialize_vector_store()
            except Exception as exc:
                warnings.warn(
                    f"ChromaDB setup failed ({type(exc).__name__}: {exc}); "
                    "safety rules will be matched locally.",
                    RuntimeWarning,
                )

    def audit_message(self, sender: str, text: str) -> dict:
        matched_rules = []
        if self.vector_store is not None:
            try:
                matched_rules = query_top_rules(self.vector_store, text, n_results=2)
                if matched_rules and matched_rules[0]["rule_id"] == "NONE":
                    matched_rules = []
            except Exception as exc:
                warnings.warn(
                    f"ChromaDB lookup failed ({type(exc).__name__}: {exc}); "
                    "using local policy matching.",
                    RuntimeWarning,
                )

        compact = re.sub(r"[^a-z0-9]", "", text.lower())
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        normalized = normalized.translate(
            str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})
        )
        if "kys" in compact:
            priority_rule_ids = ["RULE_110", "RULE_103"]
        elif re.search(r"kill yourself|kill urself|end yourself|go die", normalized):
            priority_rule_ids = ["RULE_103"]
        elif re.search(r"get cancer|subhuman|hate speech", normalized):
            priority_rule_ids = ["RULE_107", "RULE_103"]
        elif re.search(r"school|discord|snapchat|minor", normalized):
            priority_rule_ids = ["RULE_102"]
        elif re.search(r"woman|women|girl|kitchen", normalized):
            priority_rule_ids = ["RULE_101"]
        else:
            priority_rule_ids = []

        if priority_rule_ids and matched_rules:
            rule_by_id = {rule["rule_id"]: rule for rule in SAFETY_RULES}
            prioritized = [
                {
                    "content": rule_by_id[rule_id]["content"],
                    "rule_id": rule_id,
                    "category": rule_by_id[rule_id]["category"],
                    "label": rule_by_id[rule_id].get("label", 0),
                }
                for rule_id in priority_rule_ids
            ]
            prioritized_ids = set(priority_rule_ids)
            matched_rules = prioritized + [
                rule for rule in matched_rules if rule["rule_id"] not in prioritized_ids
            ]

        if not matched_rules:
            if "kys" in compact:
                preferred_rule_ids = ["RULE_110", "RULE_103"]
            elif re.search(r"kill yourself|kill urself|end yourself|go die", normalized):
                preferred_rule_ids = ["RULE_103"]
            elif re.search(r"get cancer|subhuman|hate speech", normalized):
                preferred_rule_ids = ["RULE_107", "RULE_103"]
            elif re.search(r"school|discord|snapchat|minor", text.lower()):
                preferred_rule_ids = ["RULE_102"]
            elif re.search(r"woman|women|girl|kitchen", text.lower()):
                preferred_rule_ids = ["RULE_101"]
            elif re.search(r"\bgg\s+ez\b", text.lower()):
                preferred_rule_ids = ["RULE_112", "RULE_105"]
            elif re.search(r"\bdiff\b|\bgap\b", text.lower()):
                preferred_rule_ids = ["RULE_111", "RULE_105"]
            elif re.search(r"\bff\s+at\s+15\b", text.lower()):
                preferred_rule_ids = ["RULE_113", "RULE_105"]
            elif re.search(r"trash|uninstall|noob", text.lower()):
                preferred_rule_ids = ["RULE_105", "RULE_106"]
            else:
                preferred_rule_ids = ["RULE_105"]
            rule_by_id = {rule["rule_id"]: rule for rule in SAFETY_RULES}
            matched_rules = [
                {
                    "content": rule_by_id[rule_id]["content"],
                    "rule_id": rule_id,
                    "category": rule_by_id[rule_id]["category"],
                    "label": rule_by_id[rule_id].get("label", 0),
                }
                for rule_id in preferred_rule_ids
            ]
        primary_match = matched_rules[0]
        fallback_result = _classify_without_gemini(text, primary_match)

        if self.client is None:
            self.buffer.add_message(sender, text)
            return fallback_result

        retrieved_rules_text = "\n\n".join(rule["content"] for rule in matched_rules)
        chat_context = self.buffer.get_formatted_context()
        prompt = build_audit_prompt(
            chat_context=chat_context,
            target_message=f"{sender}: {text}",
            retrieved_rule=retrieved_rules_text,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=SafetyAuditResult,
                    temperature=0.1,
                ),
            )
            audit_data = SafetyAuditResult.model_validate_json(response.text).model_dump()
            audit_data["applied_rule_id"] = primary_match["rule_id"]
            audit_data["retrieved_rule_id"] = primary_match["rule_id"]
            audit_data["retrieved_rule_category"] = primary_match["category"]
            self.buffer.add_message(sender, text)
            return audit_data
        except Exception as exc:
            warnings.warn(
                f"Gemini audit failed ({type(exc).__name__}: {exc}); "
                "using local policy classification for this message.",
                RuntimeWarning,
            )
            self.buffer.add_message(sender, text)
            return fallback_result

    def reset_chat(self):
        self.buffer.clear()


if __name__ == "__main__":
    engine = GuardianEngine()

    print("--- TESTING ENHANCED ENGINE AUDIT ---")
    test_result = engine.audit_message("Player_2", "shut THE FUCK UP everyone!")
    print(json.dumps(test_result, indent=2))

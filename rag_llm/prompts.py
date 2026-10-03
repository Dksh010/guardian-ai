from typing import Literal
from pydantic import BaseModel, Field

class SafetyAuditResult(BaseModel):
    """I define the typed fields and allowed values expected from a moderation audit."""
    is_flagged: bool = Field(
        description="True if the target message violates the community policy; False if clean banter or allowed behavior."
    )
    severity: Literal["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        description="Severity level of the violation."
    )
    violated_rule: str = Field(
        description="The specific rule ID and title violated (e.g., 'RULE_102: Child Safety'), or 'None' if clean."
    )
    target_group: Literal["Kids", "Women / Gender Minorities", "General Lobby", "None"] = Field(
        description="The demographic or player group targeted by the violation."
    )
    summary_reasoning: str = Field(
        description="1-2 concise sentences explaining why the message was or was not flagged, referencing chat context."
    )
    recommended_action: Literal["No Action", "Lobby Mute", "Temporary Ban", "Permanent Permaban & Escalate"] = Field(
        description="The automated moderation action recommended."
    )

# This instruction supplies the stable policy-reading principles shared by each Gemini request.
SYSTEM_INSTRUCTION = """
You are "Guardian AI", an expert Trust & Safety Auditor for online multiplayer games.
Your task is to analyze live player chat logs against established community guidelines and produce an objective safety audit.

KEY AUDIT PRINCIPLES:
1. MULTI-TURN CONTEXT SENSITIVITY: Evaluate the target message within the context of recent chat history. A seemingly innocent query (e.g., 'where do you live?') becomes CRITICAL when following a minor's message.
2. LEETSPEAK & FILTER EVASION: Actively decode symbols, number substitutions, or phonetic misspellings (e.g., 'k1ll y0urs3lf', '3a1 p00p', 'b!7ch'). Obfuscation increases intent severity.
3. BANTER VS. HARASSMENT:
   - CLEAN BANTER: Competitive trash-talk focused purely on skill ('too easy', 'bad shot', 'my team is sleeping') is PERMISSIBLE (Rule 105).
   - HARASSMENT: Personal attacks, identity insults, persistent multi-turn targeting, or predatory queries are SEVERE VIOLATIONS.
4. OBJECTIVITY: Maintain neutral, evidence-based reasoning.
"""

def build_audit_prompt(chat_context: str, target_message: str, retrieved_rule: str) -> str:
    """I combine retrieved policy, prior chat, and the current target into one audit prompt."""
    return f"""
=== RETRIEVED COMMUNITY POLICY ===
{retrieved_rule}

=== RECENT CHAT TRANSCRIPT (SLIDING WINDOW) ===
{chat_context}

=== TARGET MESSAGE TO AUDIT ===
Target Message: "{target_message}"

Analyze the target message against the retrieved community policy and chat history, then generate the safety audit.
"""
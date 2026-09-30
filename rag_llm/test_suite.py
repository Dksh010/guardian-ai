import json
from rag_llm.engine import GuardianEngine

def run_test_suite():
    engine = GuardianEngine()

    scenarios = [
        {
            "title": "Scenario 1: Clean Banter",
            "messages": [
                ("PlayerOne", "Nice shot mate!"),
                ("GamerBoy", "Too easy, diff gap in mid lane.")
            ]
        },
        {
            "title": "Scenario 2: Gender Harassment",
            "messages": [
                ("GamerGirl99", "I'll cover point B guys."),
                ("ToxikUser", "Is that a girl on mic? Go back to the kitchen and make me a sandwich.")
            ]
        },
        {
            "title": "Scenario 3: Leetspeak Filter Evasion",
            "messages": [
                ("SaltyPlayer", "you trash zero skill"),
                ("SaltyPlayer", "g0 k1ll ur53lf un1st4ll l1f3")
            ]
        },
        {
            "title": "Scenario 4: Multi-Turn Child Safety Violation",
            "messages": [
                ("MiniGamer", "im only 10 years old this game is hard"),
                ("PredatorX", "cool what school do you go to? add me on snapchat right now")
            ]
        }
    ]

    for scenario in scenarios:
        print(f"\n==================================================")
        print(f"RUNNING: {scenario['title']}")
        print(f"==================================================")
        engine.reset_chat()

        for sender, text in scenario["messages"]:
            print(f"\nIncoming Message -> {sender}: '{text}'")
            result = engine.audit_message(sender, text)
            
            flag_status = "🚨 FLAGGED" if result["is_flagged"] else "✅ CLEAN"
            print(f"Audit Status : {flag_status}")
            print(f"Severity     : {result['severity']}")
            print(f"Rule Cited   : {result['violated_rule']}")
            print(f"Reasoning    : {result['summary_reasoning']}")
            print(f"Action       : {result['recommended_action']}")

if __name__ == "__main__":
    run_test_suite()
class ChatBuffer:
    def __init__(self, max_history=5):
        """
        Maintains a sliding window of the last N chat messages.
        """
        self.max_history = max_history
        self.history = []

    def add_message(self, sender: str, text: str):
        """
        Appends a new message to the buffer and keeps only the latest max_history entries.
        """
        self.history.append({"sender": sender, "text": text})
        if len(self.history) > self.max_history:
            self.history.pop(0)  # Remove oldest message

    def get_formatted_context(self) -> str: 
        """
        Formats the recent chat history into a readable transcript for Gemini.
        """
        if not self.history:
            return "No prior chat context."
        
        formatted = []
        for idx, msg in enumerate(self.history, start=1):
            formatted.append(f"[{idx}] {msg['sender']}: {msg['text']}")
        return "\n".join(formatted)

    def clear(self):
        """
        Clears the chat transcript (e.g., when switching game lobbies).
        """
        self.history = []


if __name__ == "__main__":
    # Quick sanity test for the chat buffer
    buffer = ChatBuffer(max_history=3)
    buffer.add_message("KidGamer_11", "Hi I'm 11 years old")
    buffer.add_message("DarkLordX", "cool what school do you go to?")
    buffer.add_message("KidGamer_11", "Lincoln Middle School")
    buffer.add_message("DarkLordX", "add me on discord right now")

    print("--- SLIDING WINDOW BUFFER TEST (Max 3) ---")
    print(buffer.get_formatted_context())
    print("------------------------------------------\n")

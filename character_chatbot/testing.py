with open("character_chatbot/hiccup_lines.txt", "r") as f:
    raw = f.read()
    print(f"Number of newline characters: {raw.count(chr(10))}")  # chr(10) = "\n"


with open("character_chatbot/raw_lines.txt", "r") as f:
    lines = f.readlines()

hiccup_lines_count = 0
hiccup_lines = []
for line in lines:
    if line[:7] == "Hiccup:" :
        hiccup_line = line[8:]
        print(hiccup_line)
        hiccup_lines.append(hiccup_line)
        hiccup_lines_count += 1

hiccup_lines = [line for line in hiccup_lines if len(line) > 10]
with open("hiccup_lines.txt", "w") as f: 
    for line in hiccup_lines:
        f.write(line)

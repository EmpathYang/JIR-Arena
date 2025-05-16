import json
import re
import os
from utils import CURRENT_DIR

# Path to transcripts folder
transcripts_folder = os.path.join(CURRENT_DIR, "data/transcripts")
output_folder = os.path.join(CURRENT_DIR, "formatted")

# Iterate over each JSON file in the folder
for filename in os.listdir(transcripts_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(transcripts_folder, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle the "message" wrapper
        transcript_data = data.get("message", {}).get("transcript", [])

        sentences = []
        sentence_id = 1

        for d in transcript_data:
            chunks = d.get("chunks", [])

            current_sentence = ""
            start_time = None
            end_time = None

            for chunk in chunks:
                text = chunk["text"]

                if start_time is None:
                    start_time = chunk["start"]

                current_sentence += text + " "
                end_time = chunk["end"]

                if re.search(r"[.?!]$", text):
                    sentence = {
                        "id": str(sentence_id),
                        "sentence": current_sentence.strip(),
                        "start": start_time,
                        "end": end_time,
                        "data_type": "transcript"
                    }
                    sentences.append(sentence)

                    # Reset
                    sentence_id += 1
                    current_sentence = ""
                    start_time = None
                    end_time = None

        output = {"sentences": sentences}

        # Create output folder for this file
        final_output_folder = os.path.join(output_folder, filename.replace(".json", ""))
        os.makedirs(final_output_folder, exist_ok=True)

        # Save output
        output_path = os.path.join(final_output_folder, "formatted_transcript.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

print("All transcripts processed and saved.")

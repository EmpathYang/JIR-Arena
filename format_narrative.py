# import json
# import os
# from utils import CURRENT_DIR, PROMPT_DIR, extract_json

# def load_json(file_path):
#     with open(file_path, "r") as f:
#         return json.load(f)


# video_narrative = load_json(os.path.join(CURRENT_DIR, "video_narratives"))

# for idx, item in enumerate(video_narrative, start=1):
#     item['id'] = str(idx)
#     item['sentence'] = item['narrative']
#     item['type'] = "video"
#     del item['narrative']

# wrapped_output = {
#         "sentences": video_narrative
#     }

# with open("formatted_narrative.json", "w") as f:
#     json.dump(wrapped_output, f, indent=4)

import json
import os
from utils import CURRENT_DIR

# Folder containing video narrative JSON files
video_narratives_folder = os.path.join(CURRENT_DIR, "data/video_narratives")
output_folder = os.path.join(CURRENT_DIR, "formatted")

# Process each JSON file in the folder
for filename in os.listdir(video_narratives_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(video_narratives_folder, filename)

        # Load and transform
        with open(file_path, "r", encoding="utf-8") as f:
            video_narrative = json.load(f)

        for idx, item in enumerate(video_narrative, start=1):
            item["id"] = str(idx)
            item["sentence"] = item["narrative"]
            item["data_type"] = "video"
            del item["narrative"]

        wrapped_output = {"sentences": video_narrative}

        # Create output folder
        final_output_folder = os.path.join(output_folder, filename.replace(".json", ""))
        os.makedirs(final_output_folder, exist_ok=True)

        # Save formatted file
        output_path = os.path.join(final_output_folder, "formatted_narrative.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(wrapped_output, f, indent=4, ensure_ascii=False)

print("✅ All video narratives formatted and saved.")

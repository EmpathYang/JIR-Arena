import os
import json

def aggregate_judge_output(input_folder, output_file):
    aggregated_output = {
        "judge_output": []
    }

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            filepath = os.path.join(input_folder, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                    for entry in data.get("judge_output", []):
                        filtered_entry = {
                            "need": entry.get("question"),
                            "data_type": entry.get("data_type"),
                            "start_time": entry.get("start_time")
                        }
                        aggregated_output["judge_output"].append(filtered_entry)
                except json.JSONDecodeError:
                    print(f"Error decoding {filename}, skipping.")

    output_path = os.path.join(input_folder, output_file)
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(aggregated_output, outfile, indent=2)

# Example usage:
input_folder = "human_labeling/ghZRzOb_bZo"  # Replace with your actual folder path
output_file = "labels.json"
aggregate_judge_output(input_folder, output_file)

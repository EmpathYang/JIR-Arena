import os
import json
import argparse
import re
from collections import defaultdict
from copy import deepcopy
from collections import Counter

def load_combined_entries_by_index(folder_path):
    pattern = re.compile(r"judge_time_\d+")
    file_pattern = re.compile(r"(info|video)_judge_.*_updated_output\.json")

    grouped_data_by_type = defaultdict(list)

    # Collect files by type (info/video)
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path) and pattern.fullmatch(subfolder):
            for filename in sorted(os.listdir(subfolder_path)):
                if file_pattern.fullmatch(filename):
                    file_type = filename.split("_")[0]  # "info" or "video"
                    file_path = os.path.join(subfolder_path, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f).get("judge_output", [])
                            grouped_data_by_type[file_type].append(data)
                    except json.JSONDecodeError:
                        print(f"Could not parse JSON in file: {file_path}")

    final_outputs = {}

    for data_type, list_of_outputs in grouped_data_by_type.items():
        merged = {}
        
        for output in list_of_outputs:
            for entry in output:
                key = (entry.get("sentence_id"), entry.get("need"), entry.get("type"))
                if key not in merged:
                    base = deepcopy(entry)
                    base["end_times"] = []
                    base.pop("end_time", None)
                    merged[key] = base
                
                if entry.get("end_time"):
                    merged[key]["end_times"].append(entry["end_time"])
        
        final_outputs[data_type] = list(merged.values())

    return final_outputs

def populate_end_time(final_outputs):
    for data_type, entries in final_outputs.items():
        for entry in entries:
            end_times = entry.get("end_times", [])
            
            if not end_times:
                entry["end_time"] = None  # If there are no end_times, set it as None
                continue

            # Step 1: Ensure that each item in end_times is a hashable type
            # If they are dictionaries, extract a key or value that makes sense
            hashable_end_times = []
            for et in end_times:
                if isinstance(et, dict):
                    # If it's a dict, extract its key or value (modify as needed)
                    # Example: take the first key, or use a specific key from the dict
                    hashable_end_times.append((et.get('value'),et.get('end_sentence_id')))  # Convert the dict to string for hashing
                else:
                    hashable_end_times.append(et)
            
            # Step 2: Count occurrences of each end_time
            end_time_counts = Counter([et[0] for et in hashable_end_times])
            # print(end_time_counts)
            # Step 3: Get the most common end_time
            most_common_end_time, most_common_count = end_time_counts.most_common(1)[0]

            # Step 4: If the most common count is 1 (no clear majority), take the max end_time
            if most_common_count == 1:
                # print("this is max", max(hashable_end_times))
                max_tuple = max(hashable_end_times, key=lambda x: x[0])
                entry["end_time"] = max_tuple[0]
                entry["end_sentence_id"] = max_tuple[1]  # Ensure we take the maximum in case of dicts
            else:
                for value, sid in hashable_end_times:
                    if value == most_common_end_time:
                        entry["end_time"] = value
                        entry["end_sentence_id"] = sid
                        break

    return final_outputs

def save_entries(entries, output_path):
    wrapped_output = {
        "needs": entries
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(wrapped_output, f, indent=2)
    print(f"Saved {len(entries)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Deduplicate JSON entries based on semantic similarity and time overlap.")
    parser.add_argument("--input_folder", help="Folder containing JSON files")
    parser.add_argument("--output_file", help="Path to save deduplicated output")

    args = parser.parse_args()
    print(f"Loading entries from: {args.input_folder}")
    entries = load_combined_entries_by_index(args.input_folder)
    final_outputs = populate_end_time(entries)
    combined_entries = final_outputs.get("info", []) + final_outputs.get("video", [])

    print(f"{len(combined_entries)} entries remain after aggregation.")
    save_entries(combined_entries, args.output_file)


if __name__ == "__main__":
    main()

# python3 deduplicator.py --output deduplicated.json --input_folder output/procastination/aggregated --threshold 0.85

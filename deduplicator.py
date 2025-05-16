import os
import json
import argparse
from sentence_transformers import SentenceTransformer, util
import re
from collections import defaultdict
from copy import deepcopy
from collections import Counter

def load_entries_from_folder(folder_path):
    all_entries = []
    pattern = re.compile(r"need_\d+")

    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path) and pattern.fullmatch(subfolder):
            # print(subfolder_path)
            for filename in os.listdir(subfolder_path):
                if filename.endswith("aggregated.json"):
                    file_path = os.path.join(subfolder_path, filename)
                    # print(file_path)
                    if os.path.exists(file_path):
                        # print("TRUE")
                        with open(file_path, "r", encoding="utf-8") as f:
                            try:
                                data = json.load(f)
                                data = data.get('needs', [])
                                if isinstance(data, list):
                                    all_entries.extend(data)
                            except json.JSONDecodeError:
                                print(f"Could not parse JSON in file: {file_path}")
    return all_entries

def deduplicate_entries(entries, similarity_threshold=0.8):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    needs = [entry["need"] for entry in entries]
    embeddings = model.encode(needs, convert_to_tensor=True)

    merged = [False] * len(entries)
    result = []

    for i in range(len(entries)):
        if merged[i]:
            continue

        merged_entry = entries[i].copy()
        merged[i] = True

        for j in range(i + 1, len(entries)):
            if merged[j]:
                continue

            e1 = merged_entry
            e2 = entries[j]

            # Check overlapping time windows
            if not (float(e1["end_time"]) < float(e2["start_time"]) or float(e2["end_time"]) < float(e1["start_time"])):
                sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                if sim > similarity_threshold:
                    # Merge entries: min start time, max end time
                    if(e1['data_type'] != e2['data_type']):
                        if(e1['data_type'] == 'transcript'):
                            merged_entry = e1
                        else:
                            merged_entry = e2
                    else:
                        merged_entry["sentence_id"] = min(int(e1["sentence_id"]), int(e2["sentence_id"]))
                        merged_entry["start_time"] = min(float(e1["start_time"]), float(e2["start_time"]))
                        merged_entry["end_time"] = max(float(e1["end_time"]), float(e2["end_time"]))
                        merged_entry["end_sentence_id"] = max(int(e1["end_sentence_id"]), int(e2["end_sentence_id"]))
                        merged_entry["end_times"] = e1["end_times"] + e2["end_times"]
                    merged[j] = True

        result.append(merged_entry)

    return result

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
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold for deduplication")

    args = parser.parse_args()
    print(f"Loading entries from: {args.input_folder}")
    entries = load_entries_from_folder(args.input_folder)

    
    print(f"Deduplicating {len(entries)} entries with threshold: {args.threshold}")
    deduplicated = deduplicate_entries(entries, args.threshold)

    print(f"{len(deduplicated)} entries remain after aggregation.")
    save_entries(deduplicated, args.output_file)


if __name__ == "__main__":
    main()

# python3 deduplicator.py --output deduplicated.json --input_folder output/procastination/aggregated --threshold 0.85

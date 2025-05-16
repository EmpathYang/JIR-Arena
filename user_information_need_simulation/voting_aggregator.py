import json
import os
import argparse
import copy
from collections import defaultdict

def load_judge_outputs(root_folder, score_threshold=5):
    entry_votes = defaultdict(lambda: {"entries": [], "positive": 0, "total": 0})

    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)

        # Check if it is a judge folder
        if os.path.isdir(subdir_path) and subdir.startswith("judge_score_"):
            for filename in ["judge_deduplicated.json"]:
                file_path = os.path.join(subdir_path, filename)
                if os.path.isfile(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for entry in data.get("judge_output", []):
                            entry_key = f"{entry['sentence_id']}||{entry['need']}"
                            entry_votes[entry_key]["total"] += 1

                            if entry.get("likelihood_score", 0) > score_threshold:
                                entry_votes[entry_key]["positive"] += 1

                            if entry not in entry_votes[entry_key]["entries"]:
                                entry_votes[entry_key]["entries"].append(entry)

    return entry_votes

def aggregate_likelihood_scores(data):
    new_item = copy.deepcopy(data[0])
    new_item.pop("likelihood_score", None)
    new_item.pop("likelihood_score_reason", None)
    new_item.pop("likelihood_score_model_id", None)
    new_item["likelihood_scores"] = []
    for item in data:
        # Create new aggregated field
        new_item["likelihood_scores"].append({
            "score": item["likelihood_score"],
            "reason": item["likelihood_score_reason"],
            "model_id": item["likelihood_score_model_id"]
        })
    return new_item

def apply_voting_logic(entry_votes, voting_threshold):
    final_output = []
    for record in entry_votes.values():
        if record["positive"] >= voting_threshold:
            best_entry = aggregate_likelihood_scores(record["entries"])
            final_output.append(best_entry)
    return final_output


def save_output(output_data, output_path):
    with open(output_path, "w") as out_file:
        json.dump({"judge_output": output_data}, out_file, indent=2)


def main(args):
    entry_votes = load_judge_outputs(args.folder_path, args.score_threshold)
    final_output = apply_voting_logic(entry_votes, args.voting_threshold)
    output_file = os.path.join(args.folder_path, "final_output.json")
    save_output(final_output, output_file)
    print(f"Saved aggregated output to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate judge outputs based on voting.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing .json judge outputs.")
    parser.add_argument("--score_threshold", type=int, default=5, help="Relevance score threshold to consider a positive vote.")
    parser.add_argument("--voting_threshold", type=int, default=2, help="Minimum number of positive votes required.")

    args = parser.parse_args()
    main(args)

# python3 voting_aggregator.py --folder_path output/procastination/judges --score_threshold 5 --voting_threshold 2

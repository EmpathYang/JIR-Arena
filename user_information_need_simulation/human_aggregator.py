import json
import argparse
import os
from sentence_transformers import SentenceTransformer, util

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(text1, text2):
    """Compute cosine similarity between two sentences."""
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

def merge_needs(human_labels, llm_labels, similarity_threshold):
    merged_output = llm_labels.copy()
    merged_judge_output = llm_labels.get("judge_output", [])

    all_keys = {key for item in merged_judge_output for key in item.keys()}

    llm_needs_text = [
        (index, item.get("need", ""))
        for index, item in enumerate(merged_judge_output)
        if item.get("need")
    ]

    total_removed = 0  # Track number of removed items

    for human_need_obj in human_labels.get("judge_output", []):
        human_need = human_need_obj.get("need", "")
        if not human_need:
            continue

        # Find similar needs
        similar_indices = [
            index for index, llm_need in llm_needs_text
            if compute_similarity(human_need, llm_need) >= similarity_threshold
        ]
        total_removed += len(similar_indices)

        # Remove them from the output
        merged_judge_output = [
            item for i, item in enumerate(merged_judge_output) if i not in similar_indices
        ]

        enriched_entry = {key: None for key in all_keys}
        enriched_entry.update(human_need_obj)
        merged_judge_output.append(enriched_entry)

        llm_needs_text = [
            (i, item.get("need", ""))
            for i, item in enumerate(merged_judge_output)
            if item.get("need")
        ]

    merged_output["judge_output"] = merged_judge_output
    return merged_output, total_removed

def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Merge human-labeled needs into LLM output based on semantic similarity.")
    parser.add_argument("--human", required=True, help="Path to human labeling JSON file")
    parser.add_argument("--llm", required=True, help="Path to LLM labeling JSON file (final_output.json)")
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold (default: 0.8)")
    return parser.parse_args()

def main():
    args = parse_args()

    human_labels = load_json(args.human)
    llm_labels = load_json(args.llm)

    merged_result, removed_count = merge_needs(human_labels, llm_labels, args.threshold)

    output_dir = os.path.dirname(args.llm)
    output_path = os.path.join(output_dir, "final_output_with_human.json")
    save_json(merged_result, output_path)

    print(f"✅ Final output written to '{output_path}'.")
    print(f"🗑️  Removed {removed_count} redundant LLM-labeled needs.")

if __name__ == "__main__":
    main()

#python3 human_aggregator.py  --human human_labeling/wvXDB9dMdEo/labels.json --llm output/wvXDB9dMdEo/final_output.json --threshold 0.75

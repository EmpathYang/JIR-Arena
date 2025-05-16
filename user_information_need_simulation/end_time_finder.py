import json
import argparse
from typing import Dict, List, Any
from utils import load_json


def save_json(data: Any, filepath: str) -> None:
    """Save JSON to a file."""
    wrapped_output = {
        "judge_output": data
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(wrapped_output, f, indent=2)


def populate_end_time_values(judge_data: List[Dict], sentence_map: Dict[str, str]) -> List[Dict]:
    """Populate 'end_time.value' in judge_data using sentence_map."""
    for entry in judge_data:
        end_time = entry.get("end_time", {})
        end_sentence_id = str(end_time.get("end_sentence_id"))
        start_sentence_id = str(entry.get("sentence_id"))
        entry["start_time"] = sentence_map[start_sentence_id]['start']
        if end_sentence_id and end_sentence_id in sentence_map:
            entry["end_time"]["value"] = sentence_map[end_sentence_id]['end']
    return judge_data

def build_sentence_map(data: Any) -> Dict[str, str]:
    """
    Convert a list of sentence dictionaries into a mapping from ID to sentence.
    Assumes each item has an 'id' and a 'sentence'.
    """
    if isinstance(data, list) and all(isinstance(d, dict) and 'id' in d and 'sentence' in d for d in data):
        return {str(item['id']): item for item in data}
    else:
        raise ValueError("Unsupported sentence data format.")

def main():
    parser = argparse.ArgumentParser(description="Populate end_time.value in judge_output.json using sentences.json.")
    parser.add_argument("--judge_output", required=True, help="Path to the judge_output JSON file.")
    parser.add_argument("--sentences", required=True, help="Path to the sentences JSON file.")
    parser.add_argument("--output", required=True, help="Path to save the updated judge_output JSON file.")

    args = parser.parse_args()

    judge_output = load_json(args.judge_output).get("judge_output",[])
    sentence_data = load_json(args.sentences).get("sentences",[])
    sentence_map = build_sentence_map(sentence_data)

    updated_output = populate_end_time_values(judge_output, sentence_map)
    save_json(updated_output, args.output)

    print(f"Updated file saved to {args.output}")


if __name__ == "__main__":
    main()
# python3 end_time_finder.py --judge_output output/procastination/judge_output.json --sentences example/procrastination/formatted_transcript.json --output updated_output.json

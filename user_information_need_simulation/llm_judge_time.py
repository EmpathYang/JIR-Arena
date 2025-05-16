import os
import json
import time
import argparse
from llms.gpt import call_gpt
from llms.models import call_model
from llms.uiuc_chat_llms import call_uiuc_chat_llms
from llms.utils import INFORMATION_NEED_SYSTEM_PROMPT
from utils import CURRENT_DIR, PROMPT_DIR, NEEDS_DIR, extract_json, load_json, load_prompt
from tqdm import tqdm
from typing import Dict, List, Any
from user_information_need_simulation.end_time_finder import build_sentence_map
from llm_judge import invoke_llm, generate_summary, find_end_time


def get_end_time_values(end_sentence_id, sentence_map: Dict[str, str]) -> List[Dict]:
    """Populate 'end_time.value' in judge_data using sentence_map."""
    end_sentence_id = str(end_sentence_id)
    if end_sentence_id and end_sentence_id in sentence_map:
        end_time_value = sentence_map[end_sentence_id]['end']
    return end_time_value

def process_transcript(args):
    all_needs_data = load_json(os.path.join(CURRENT_DIR, args.needs_file)).get("needs", [])
    transcript_needs = [n for n in all_needs_data if n.get("data_type") == "transcript"]
    video_needs = [n for n in all_needs_data if n.get("data_type") == "video"]
    model_ids = args.model_id
    results = []

    # Helper function to run processing for each group
    def run_processing(needs_data, transcript_file):
        transcript_data = load_json(os.path.join(CURRENT_DIR, transcript_file)).get("sentences", [])
        sentence_map = build_sentence_map(transcript_data)
        summary_prompt_template = load_prompt(os.path.join(PROMPT_DIR, "summary_prompt.txt"))
        llm_judge_prompt_template = load_prompt(os.path.join(PROMPT_DIR, "relevance_score_judge_1.txt"))
        find_end_time_prompt_template = load_prompt(os.path.join(PROMPT_DIR, "find_end_time.txt"))

        local_results = []

        for i in tqdm(range(0, len(transcript_data), args.chunk_size)):
            chunk = transcript_data[i:i + args.chunk_size]
            chunk_ids = [str(s['id']) for s in chunk]
             
            try:
                cur_needs = [n['information_need'] for n in needs_data if str(n['id']) in chunk_ids]
            except:
                cur_needs = [n for n in needs_data if str(n['sentence_id']) in chunk_ids]

            if not cur_needs:
                continue

            prev_sentences = transcript_data[max(0, i - args.n_last):i]
            history_sentences = transcript_data[:max(0, i - args.n_last)]
            next_sentences = transcript_data[i + args.chunk_size: i + args.chunk_size + args.n_next]

            for cur_need in cur_needs:
                # print(cur_need)
                cur_models = [entry["model_id"] for entry in cur_need["end_times"]]
                remaining_models = [model for model in model_ids if model not in cur_models]

                for model in remaining_models:
                    summary = generate_summary(history_sentences, summary_prompt_template, model)

                    end_time_info = {}

                    temp_cur_need = cur_need.copy()
                    temp_cur_need.pop("end_times")
                    temp_cur_need.pop("end_time")
                    temp_cur_need.pop("end_sentence_id")
                    end_time_info = find_end_time(summary, prev_sentences, chunk, next_sentences, temp_cur_need, find_end_time_prompt_template, model)
                    try:
                        end_time = end_time_info.get('end_time')[0]
                    except:
                        end_time = end_time_info
                    end_time['model_id'] = model
                    end_time['value'] = get_end_time_values(end_time['end_sentence_id'], sentence_map)
                    combined_item = cur_need.copy()
                    combined_item['end_times'].append(end_time)
                    combined_item["end_time"] = max(float(combined_item["end_time"]), float(end_time["value"]))
                    combined_item["end_sentence_id"] = max(int(combined_item["end_sentence_id"]), int(end_time["end_sentence_id"]))
                        
                    local_results.append(combined_item)
        return local_results

    # Run for transcript needs
    if transcript_needs:
        print("Processing Transcript Needs")
        results += run_processing(transcript_needs, args.transcript_file)

    # Run for video needs
    if video_needs:
        print("Processing Video Needs")
        results += run_processing(video_needs, args.narrative_file)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process transcript for information needs and end times.")
    parser.add_argument('--transcript_file', type=str, required=True, help='Transcript JSON file path (relative to CURRENT_DIR)')
    parser.add_argument('--narrative_file', type=str, required=True, help='Narrative JSON file path (relative to CURRENT_DIR)')
    parser.add_argument('--needs_file', type=str, required=True, help='Needs JSON file path (relative to CURRENT_DIR)')
    parser.add_argument('--output_file', type=str, default="final_output.json", help='Output file name')
    parser.add_argument('--model_id', nargs='+', required=True, help="List of model IDs")
    parser.add_argument('--chunk_size', type=int, default=1, help='Number of sentences per chunk')
    parser.add_argument('--n_last', type=int, default=5, help='Number of previous sentences for context')
    parser.add_argument('--n_next', type=int, default=5, help='Number of next sentences for context')
    
    args = parser.parse_args()
    final_output = process_transcript(args)

    if(len(final_output) > 0):
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump({"needs": final_output}, f, indent=2, ensure_ascii=False)

        print(f"Saved output to {args.output_file}")

#python3 llm_judge.py --transcript_file example/procrastination/formatted_transcript.json --narrative_file example/procrastination/formatted_narrative.json --needs_file output/procastination/info_need_transcript.json --output_file output/procastination/judge_output.json --model_id qwen2.5:14b-instruct-fp16 --chunk_size 1 --n_last 5 --n_next 5 --mode score

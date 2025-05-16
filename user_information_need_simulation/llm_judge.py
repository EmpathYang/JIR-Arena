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


def invoke_llm(prompt, model_id, system_prompt=INFORMATION_NEED_SYSTEM_PROMPT, max_retries=10, retry_delay=3):
    for attempt in range(max_retries):
        try:
            response = call_model(prompt, model_id=model_id, system_prompt=system_prompt)
            json_str = extract_json(response)
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"[Attempt {attempt + 1}] JSON decoding failed. Retrying... : {response}")
            time.sleep(retry_delay)
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    print("Max retries reached. Skipping this chunk.")
    return {}


def generate_summary(history_sentences, summary_prompt_template, model_id):
    if len(history_sentences) == 0:
        return ""
    history_text = " ".join([s["sentence"] for s in history_sentences])
    prompt = summary_prompt_template.replace("{{Sentences}}", history_text)
    try:
        response = invoke_llm(prompt, model_id)
        return response.get("summary", "")
    except Exception as e:
        print(f"Error generating summary: {e}")
        return ""


def find_end_time(summary, prev_sentences, chunk, next_sentences, info_needs, find_end_time_prompt_template, model_id):
    try:
        prev_text = ". ".join([s["sentence"] for s in prev_sentences])
        next_text = ". ".join([s["sentence"] for s in next_sentences])
        prompt = (
            find_end_time_prompt_template
                .replace("{{Summary}}", str(summary))
                .replace("{{Prev_Sentences}}", prev_text)
                .replace("{{Next_Sentences}}", json.dumps(next_sentences))
                .replace("{{Information_Need}}", json.dumps(info_needs))
                .replace("{{Transcript_Chunk}}", json.dumps(chunk))
        )
        response = invoke_llm(prompt, model_id)
        return response
    except Exception as e:
        print("Error finding end time:", e)
        return {}


def process_transcript(args):
    all_needs_data = load_json(os.path.join(CURRENT_DIR, args.needs_file)).get("needs", [])
    transcript_needs = [n for n in all_needs_data if n.get("data_type") == "transcript"]
    video_needs = [n for n in all_needs_data if n.get("data_type") == "video"]

    results = []

    # Helper function to run processing for each group
    def run_processing(needs_data, transcript_file):
        transcript_data = load_json(os.path.join(CURRENT_DIR, transcript_file)).get("sentences", [])
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

            summary = generate_summary(history_sentences, summary_prompt_template, args.model_id)

            judged_output = {}
            end_time_info = {}

            if args.mode in ["score", "both"]:
                prev_text = ". ".join([s['sentence'] for s in prev_sentences])
                judge_prompt = (
                    llm_judge_prompt_template
                        .replace("{{Transcript_Chunk}}", json.dumps(chunk))
                        .replace("{{Background_Context}}", str(summary))
                        .replace("{{Information_Need}}", json.dumps(cur_needs))
                        .replace("{{Prev_Sentences}}", prev_text)
                )
                judged_output = invoke_llm(judge_prompt, args.model_id)

            if args.mode in ["end_time", "both"]:
                end_time_info = find_end_time(summary, prev_sentences, chunk, next_sentences, cur_needs, find_end_time_prompt_template, args.model_id)

            if args.mode == "score":
                for info_need, score in zip(cur_needs, judged_output.get('score', [])):
                    combined_item = info_need.copy()
                    combined_item['likelihood_score'] = float(score['relevance_score'])
                    combined_item['likelihood_score_reason'] = score['relevance_score_reason']
                    combined_item['likelihood_score_model_id'] = args.model_id
                    local_results.append(combined_item)
            elif args.mode == "end_time":
                for info_need, end_time in zip(cur_needs, end_time_info.get('end_time', [])):
                    end_time['model_id'] = args.model_id
                    combined_item = info_need.copy()
                    combined_item['end_time'] = end_time
                    local_results.append(combined_item)
            elif args.mode == "both":
                for info_need, end_time, score in zip(cur_needs, end_time_info.get('end_time', []), judged_output.get('score', [])):
                    combined_item = info_need.copy()
                    end_time['model_id'] = args.model_id
                    combined_item['end_time'] = end_time
                    combined_item['likelihood_score'] = float(score['relevance_score'])
                    combined_item['likelihood_score_reason'] = score['relevance_score_reason']
                    combined_item['likelihood_score_model_id'] = args.model_id
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
    parser.add_argument('--model_id', type=str, default="qwen2.5:14b-instruct-fp16", help='Model ID to use')
    parser.add_argument('--chunk_size', type=int, default=1, help='Number of sentences per chunk')
    parser.add_argument('--n_last', type=int, default=5, help='Number of previous sentences for context')
    parser.add_argument('--n_next', type=int, default=5, help='Number of next sentences for context')
    parser.add_argument('--mode', choices=["score", "end_time", "both"], default="both", help='Mode to execute: score, end_time, or both')

    args = parser.parse_args()
    final_output = process_transcript(args)

    if(len(final_output) > 0):
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump({"judge_output": final_output}, f, indent=2, ensure_ascii=False)

        print(f"Saved output to {args.output_file}")

#python3 llm_judge.py --transcript_file example/procrastination/formatted_transcript.json --narrative_file example/procrastination/formatted_narrative.json --needs_file output/procastination/info_need_transcript.json --output_file output/procastination/judge_output.json --model_id qwen2.5:14b-instruct-fp16 --chunk_size 1 --n_last 5 --n_next 5 --mode score

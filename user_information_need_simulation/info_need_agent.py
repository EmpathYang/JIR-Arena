import os
import json
import time
import logging
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from llms.gpt import call_gpt
from llms.uiuc_chat_llms import call_uiuc_chat_llms
from llms.models import call_model
from utils import CURRENT_DIR, PROMPT_DIR, extract_json, load_json, load_text, load_prompt
from llms.utils import INFORMATION_NEED_SYSTEM_PROMPT


# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM Information Need Extractor")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file.")
    parser.add_argument("--transcript_path", type=str, required=True, help="Path to the transcript JSON file.")
    parser.add_argument("--chunk_size", type=int, default=5, help="Number of sentences per chunk.")
    parser.add_argument('--model_id', type=str, default="qwen2.5:14b-instruct-fp16", help='Model ID to use')
    return parser.parse_args()


def invoke_llm(prompt, model_id, max_retries = 10, retry_delay = 3):
    attempt = 0
    while attempt < max_retries:
        try:
            response = call_model(prompt, model_id=model_id, system_prompt=INFORMATION_NEED_SYSTEM_PROMPT)
            return json.loads(extract_json(response))
        except json.JSONDecodeError:
            attempt += 1
            print(f"JSON decode error, retrying {attempt}/{max_retries}...")
            time.sleep(retry_delay)
        except Exception as e:
            print(f"Unexpected error during LLM call: {e}")
            break
    print("Max retries reached. Skipping this chunk.")
    return {}


def build_prompt(template, replacements):
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def process_transcript_chunk(chunk, context, templates, model_id,
                              max_retries = 5, retry_delay = 3):
    chunk_text = " ".join(item["sentence"] for item in chunk)

    info_need1_prompt = build_prompt(
        templates["info_need1"],
        {"{{Transcript_Chunk}}": str(chunk), "{{Background_Context}}": context}
    )
    info_need1_data = invoke_llm(info_need1_prompt, model_id=model_id)

    info_need2_prompt = build_prompt(
        templates["info_need2"],
        {
            "{{Transcript_Chunk}}": str(chunk),
            "{{Background_Context}}": context,
            "{{Information_Need}}": str(info_need1_data)
        }
    )
    info_need2_data = invoke_llm(info_need2_prompt, model_id=model_id)

    summary_prompt = build_prompt(templates["summary"], {"{{Sentences}}": chunk_text})
    try:
        summary_response = extract_json(call_model(summary_prompt, model_id=model_id))
        summary_data = json.loads(summary_response)
        summary = summary_data.get("summary", "")
    except Exception:
        summary = ""

    processed_data = []
    if info_need2_data:
        for data in info_need2_data.get("information_needs", []):
            for item in chunk:
                if str(item["id"]) == str(data["sentence_id"]):
                    data["data_type"] = item["data_type"]
                    data["model_id"] = model_id
                    processed_data.append({
                        "id": str(item["id"]),
                        "sentence": item["sentence"],
                        "data_type": item["data_type"],
                        "information_need": data
                    })

    return summary, processed_data


def main():
    args = parse_arguments()

    transcript = load_json(os.path.join(CURRENT_DIR, args.transcript_path)).get("sentences", [])
    templates = {
        "info_need1": load_prompt("info_need1.txt"),
        "info_need2": load_prompt("info_need2.txt"),
        "summary": load_prompt("summary_prompt.txt"),
        # "llm_judge": load_prompt("llm_judge.txt")
    }

    context = ""
    final_output = []
    output_file_path = os.path.join(CURRENT_DIR, args.output_path)
    entries_since_last_save = 0

    for i in tqdm(range(0, len(transcript), args.chunk_size)):
        chunk = transcript[i:i + args.chunk_size]
        summary, processed_chunk_data = process_transcript_chunk(chunk, context, templates, model_id=args.model_id)
        context = summary
        final_output.extend(processed_chunk_data)
        entries_since_last_save += len(processed_chunk_data)

        # Save progress every 10 items added
        if entries_since_last_save >=10:
            print("Saving Progress....")
            wrapped_output = {
                "needs": final_output
            }
            with open(output_file_path, "w") as f:
                json.dump(wrapped_output, f, indent=4)
            entries_since_last_save = 0


    # Wrap final_output under "needs"
    wrapped_output = {
        "needs": final_output
    }

    with open(output_file_path, "w") as f:
        json.dump(wrapped_output, f, indent=4)


if __name__ == "__main__":
    main()

 #python3 info_need_agent.py --output_path output/procastination/info_need_transcript.json --transcript_path example/procrastination/transcript_sentences.json --chunk_size 5 --model_id qwen2.5:14b-instruct-fp16
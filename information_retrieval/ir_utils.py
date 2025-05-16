import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from IR_CONSTANTS import JSONL_DATA_DIR
from utils import METAINFO_DIR, load_jsonl, load_template
from llms.gpt import call_gpt
from llms.deepseek_r1 import call_deepseek_r1
from llms.claude import call_claude

def load_id_text_dict_from_dir(directory):
    data_dict = {}

    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    # Parse each line as a JSON object
                    json_obj = json.loads(line.strip())
                    # Assume the json_obj has 'id' and 'contents' keys
                    data_dict[json_obj['id']] = json_obj['contents']
    return data_dict

def load_id_text_dict_from_knowledge_base(knowledge_base):
    return load_id_text_dict_from_dir(os.path.join(JSONL_DATA_DIR, knowledge_base))

def dump_json_to_jsonl_file(json_obj, jsonl_filepath):
    with open(jsonl_filepath, 'a') as jsonl_file:
        json.dump(json_obj, jsonl_file)
        jsonl_file.write('\n')

def get_video_type_by_youtube_id(youtube_id):
    for filename in os.listdir(METAINFO_DIR):
        if filename.endswith(".jsonl"):
            jsonls = load_jsonl(os.path.join(METAINFO_DIR, filename))
            for jsonl in jsonls:
                if youtube_id == jsonl["youtube_id"]:
                    return filename[:-len(".jsonl")]


def find_unprocessed_videos(root_dir, unwanted_names):
    valid_dirs = []
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        
        if not os.path.isdir(subdir_path):
            continue  

        if any(unwanted in subdir for unwanted in unwanted_names):
            continue

        has_final_jsonl = any(fname.endswith('jir_references_relevance_score.jsonl') for fname in os.listdir(subdir_path))
        if not has_final_jsonl:
            valid_dirs.append(subdir)
    return valid_dirs

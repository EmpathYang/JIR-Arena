import os
import json
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(CURRENT_DIR, "prompts")
METAINFO_DIR = os.path.join(CURRENT_DIR, "data/scene_metainfo")


def load_jsonl(filepath):
    data = []
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def load_template(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()
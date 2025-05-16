import os
import re
import json
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(CURRENT_DIR, "prompts")
NEEDS_DIR = os.path.join(CURRENT_DIR, "needs")
METAINFO_DIR = os.path.join(CURRENT_DIR, "data/scene_metainfo")

# Load JSON data
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_text(file_path):
    with open(file_path, "r") as f:
        return f.read()

def load_prompt(prompt_name):
    return load_text(os.path.join(PROMPT_DIR, prompt_name))

def build_prompt(template, replacements):
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template

def chunk_document(document:str, k:int, overlap_pct=0.1):
    # Tokenize the document into words (or tokens)
    tokens = document.split(" ")
    
    # Calculate the overlap size (overlap_pct of k)
    overlap_size = int(k * overlap_pct)
    
    # Create the chunks with overlap
    chunks = []
    for i in range(0, len(tokens), k - overlap_size):
        chunk = tokens[i:i + k]
        if len(chunk) < k:
            # If the chunk has less than k tokens, we add the remainder
            chunk = tokens[i:]
        chunks.append(" ".join(chunk))
    
    return chunks

def extract_json(text):
    start = text.find('{')
    end = text.rfind('}')

    if start != -1 and end != -1 and start < end:
        json_str = text[start:end+1]
        return json_str
    else:
        print("Could not find valid JSON boundaries.")
        return None

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

import json
import re
import os
from tqdm import tqdm
from functools import partial
from IR_CONSTANTS import get_video_knowledge_bases, QUALITY_CHECK_KNOWLEDGE_BASE_TO_CUSTOMED_PROMPT_MAP, YOUTUBE_ID_TO_PAPER_ID_MAP, PRE_RETRIEVAL_SOURCE_CHECK_PROMPT
from ir_utils import dump_json_to_jsonl_file, load_jsonl, load_template, call_gpt, call_deepseek_r1, get_video_type_by_youtube_id

def extract_json_from_response(response: str, attempts=1) -> dict:
    """
    Extracts JSON content from a string that may include markdown-style code block delimiters.
    """
    def transform_backslashes(input_string):
        """
        Transform a string by doubling all backslashes (\ → \$ except those that form
        special escape sequences like \n, \t, \r, etc.
        
        Args:
            input_string (str): The input string to transform
            
        Returns:
            str: The transformed string with proper backslash handling
        """
        if not input_string:
            return ""
        
        # List of common escape sequences to preserve
        escape_sequences = ['\\n', '\\t', '\\r', '\\b', '\\f', '\\v', '\\"', "\\'", '\\a', '\\\\']
        
        i = 0
        result = []
        while i < len(input_string):
            # Check if current character is a backslash
            if input_string[i] == '\\':
                # Check if this backslash starts a recognized escape sequence
                if i + 1 < len(input_string) and f"\\{input_string[i+1]}" in escape_sequences:
                    # This is an escape sequence, keep it as is
                    result.append(input_string[i:i+2])
                    i += 2
                else:
                    # This is a lone backslash, double it
                    result.append('\\\\')
                    i += 1
            else:
                # Not a backslash, just append the character
                result.append(input_string[i])
                i += 1
        
        return ''.join(result)
    cleaned = re.sub(r'^```json\s*', '', response.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'```$', '', cleaned.strip())

    fixed_str = cleaned.rstrip()

    if fixed_str.endswith('}'):
        try:
            return json.loads(fixed_str)
        except json.JSONDecodeError:
            pass
    if fixed_str.endswith('"'):
        fixed_str = fixed_str[:-1]

    if not fixed_str.endswith('}'):
        fixed_str += '..."}'

    try:
        return json.loads(fixed_str)
    except:
        try:
            _fixed_str = fixed_str.replace("\c", "\\c")
            return json.loads(_fixed_str)
        except:
            try:
                _fixed_str = transform_backslashes(fixed_str)
                return json.loads(_fixed_str)
            except json.JSONDecodeError as e:
                if attempts >= 3:
                    print(fixed_str)
                    json_filepath = input("Input manually created json obj dir: ")
                    if json_filepath.strip() == "":
                        json_filepath = "/home/key4/trash.json"
                    return json.load(open(json_filepath, "r"))
                else:
                    print("❌ Failed to parse JSON:", e)
                    raise
        
def extract_yes_no(text):
    match = re.search(r'1\.\s*\**(Yes|No)\**', text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    else:
        return "Yes"        

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def split_text_by_tokens(text, tokenizer, max_tokens):
    input_ids = tokenizer.encode(text)
    chunks = []

    for i in range(0, len(input_ids), max_tokens):
        chunk_ids = input_ids[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)

    return chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--youtube_id_list', type=str, nargs='+', required=True)
    parser.add_argument('--jir_references_dir', type=str, required=True)
    parser.add_argument('--output_jir_references_quality_check_dir', type=str, required=True)
    parser.add_argument('--template_filepath', type=str, required=True)
    parser.add_argument('--id_content_dict_filepath', type=str, required=True)

    args = parser.parse_args()

    YOUTUBE_ID_LIST = args.youtube_id_list
    JIR_REFERENCES_DIR = args.jir_references_dir
    OUTPUT_JIR_REFERENCES_QUALITY_CHECK_DIR = args.output_jir_references_quality_check_dir    
    
    template = load_template(args.template_filepath)
    id_content_dicts = json.load(open(args.id_content_dict_filepath, "r"))
    call_models = {"gpt-4o": partial(call_gpt, model_id="gpt-4o"), "DeepSeek-V3-0324": partial(call_deepseek_r1, model_id="DeepSeek-V3-0324-2")}
    
    for youtube_id in YOUTUBE_ID_LIST:
        jirs_filepath = os.path.join(JIR_REFERENCES_DIR, youtube_id, "jir_references_raw.jsonl")
        if not os.path.exists(jirs_filepath):
            continue
        video_type = get_video_type_by_youtube_id(youtube_id)
        video_id = YOUTUBE_ID_TO_PAPER_ID_MAP[youtube_id] if video_type == "paper" else youtube_id
        knowledge_bases = get_video_knowledge_bases(video_type, video_id)
        jirs_quality_check_filepath = os.path.join(OUTPUT_JIR_REFERENCES_QUALITY_CHECK_DIR, youtube_id, "jir_references_quality_check.jsonl")
        
        jirs_list = load_jsonl(jirs_filepath)
        existing_jirs_list = load_jsonl(jirs_quality_check_filepath)
        existing_list = [' '.join(jir.get(k, '').strip() for k in ["need", "question", "reason"] if jir.get(k, '')) for jir in existing_jirs_list]
        for i, jir in enumerate(jirs_list):
            if ' '.join(jir.get(k, '').strip() for k in ["need", "question", "reason"] if jir.get(k, '')) in existing_list:
                continue
            
            if jir.get("reason") and jir.get("need") and jir.get("question"):
                query = f'''Reason: {jir['reason']}\nAudience Information Need: {jir['need']}\nQuery: {jir['question']}'''
            elif not jir.get("reason") and jir.get("need") and not jir.get("question"):
                query = jir["need"]
            jir["ir_quality_check"] = {}
            for model_id in call_models.keys():
                jir["ir_quality_check"][model_id] = {}          
                for kb in knowledge_bases:
                    pre_retrieval_source_check_prompt = PRE_RETRIEVAL_SOURCE_CHECK_PROMPT[kb.split("/")[0]].replace("{{query}}", query)
                    pre_retrieval_source_check_response = call_models[model_id](pre_retrieval_source_check_prompt) 
                    
                    pre_retrieval_source_check = True if extract_yes_no(pre_retrieval_source_check_response)=="No" else False
                    jir["ir_quality_check"][model_id][kb] = {
                        "pre_retrieval_source_check": pre_retrieval_source_check_response
                    }
                    if pre_retrieval_source_check:
                        continue
                    customed_instruction = QUALITY_CHECK_KNOWLEDGE_BASE_TO_CUSTOMED_PROMPT_MAP[kb.split("/")[0]]
                    for ir in tqdm(jir["content"]["references"][kb], f"{youtube_id} {i+1} {model_id} {kb}"):
                        skip_doc = False
                        id = ir[0]
                        text = id_content_dicts[kb][id]
                        token_num = 2048 if kb == "wikipedia" else 1024
                        text_splits = split_text_by_tokens(text, tokenizer, token_num)
                        for text_split in text_splits:
                            prompt = template.replace("{{customed_instruction}}", customed_instruction).replace("{{query}}", query).replace("{{reference_material}}", text_split)
                            get_valid_response = False
                            attempts = 0
                            while not get_valid_response:
                                if attempts >= 5:
                                    exit()
                                response = call_models[model_id](prompt)
                                attempts += 1
                                try: 
                                    if response == "":
                                        get_valid_response = True
                                        continue
                                    json_obj = extract_json_from_response(response, attempts)
                                except:
                                    continue
                                if "answerable" in json_obj.keys() and "supporting_content" in json_obj.keys():
                                    get_valid_response = True
                                else:
                                    continue
                                if json_obj["answerable"]:
                                    if id in jir["ir_quality_check"][model_id][kb].keys():
                                        jir["ir_quality_check"][model_id][kb][id].append(json_obj["supporting_content"])
                                    else:
                                        jir["ir_quality_check"][model_id][kb][id] = [json_obj["supporting_content"]]
                                    skip_doc = True
                                    break
                            if skip_doc:
                                break

            dump_json_to_jsonl_file(jir, jirs_quality_check_filepath)
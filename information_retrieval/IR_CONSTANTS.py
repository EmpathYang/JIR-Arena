import os
import json
RAW_DATA_DIR = ""
JSONL_DATA_DIR = ""
SPARSE_INDEXING_DIR = ""
DENSE_INDEXING_DIR = ""
PAPER_META_INFO_FILEPATH = ""
LECTURE_META_INFO_FILEPATH = ""

PAPER_BASIC_KB = ["wikipedia", "arxiv"]
LECTURE_BASIC_KB = ["wikipedia"]

PAPER_ID_TO_ADDITIONAL_KNOWLEDGE_BASES_MAP = {
    json_obj['paper_id']: [f"paper/{json_obj['paper_id']}"]
    if os.path.exists(os.path.join(SPARSE_INDEXING_DIR, "paper", str(json_obj['paper_id']))) 
    else [] 
    for json_obj in [json.loads(line) for line in open(PAPER_META_INFO_FILEPATH, "r", encoding="utf-8")]
}

LECTURE_ID_TO_ADDITIONAL_KNOWLEDGE_BASES_MAP = {
    "ghZRzOb_bZo": ["textbook/textdatabook"],
    "u-mPCGWlUVo": ["textbook/textdatabook"],
    "ejwr_Wxe-7Y": ["textbook/textdatabook"],
    "SoZStBaLbws": ["textbook/textdatabook"],
}

BENCHMARK_ID_TO_ADDITIONAL_KNOWLEDGE_BASES_MAP = {
    **PAPER_ID_TO_ADDITIONAL_KNOWLEDGE_BASES_MAP,
    **LECTURE_ID_TO_ADDITIONAL_KNOWLEDGE_BASES_MAP
}

VIDEO_TYPE_TO_BASIC_KB_MAP = {
    "paper": PAPER_BASIC_KB,
    "lecture": LECTURE_BASIC_KB
}

VIDEO_TYPE_ID_TO_ADDITIONAL_KNOWLEDGE_BASES_MAP = {
    "paper": PAPER_ID_TO_ADDITIONAL_KNOWLEDGE_BASES_MAP,
    "lecture": LECTURE_ID_TO_ADDITIONAL_KNOWLEDGE_BASES_MAP,
}

def get_video_knowledge_bases(video_type, video_id):
    return VIDEO_TYPE_TO_BASIC_KB_MAP[video_type]+VIDEO_TYPE_ID_TO_ADDITIONAL_KNOWLEDGE_BASES_MAP[video_type].get(video_id, [])


QUALITY_CHECK_KNOWLEDGE_BASE_TO_CUSTOMED_PROMPT_MAP = {
    "wikipedia": "The reference material comes from a Wikipedia article. This article is not the source where the query comes from, so if the query is asking about the specific content, methods, or claims of the paper/study being presented, this reference is insufficient. In such cases, the correct response is \"answerable\": false.",
    "arxiv": "The reference material includes the title and abstract of an arXiv paper. This is not the source where the query comes from, so if the query is asking about the specific content, methods, or claims of the paper/study being presented, this reference is insufficient. In such cases, the correct response is \"answerable\": false.",
    "paper": "The reference material is extracted from the body of the research paper being presented. It may have been converted from a PDF and could contain distortions such as missing figures with only captions, incomplete tables, or malformed equations. Ignore these potential issues and assume the text reflects the intended content of the original paper. Your task is to evaluate whether the given query can be answered based on the provided excerpt, even partially, as if it were a faithful representation of the original document.",
    "textbook": "The reference material comes from a relevant textbook specialized on the topic being discussed."
}

YOUTUBE_ID_TO_PAPER_ID_MAP = {
    data['youtube_id']: data['paper_id'] for data in (json.loads(line.strip()) for line in open(PAPER_META_INFO_FILEPATH)) if 'youtube_id' in data and 'paper_id' in data
}

PRE_RETRIEVAL_SOURCE_CHECK_PROMPT = {
    "wikipedia": '''Do you believe that the query below could be at least partially answered using content from Wikipedia pages?

{{query}}

In your response, include:
1. Yes or No
2. A brief explanation''',

    "arxiv": '''Do you believe that the query below could be at least partially answered using content from arXiv papers (**excluding the original study's paper/report or its primary data/code**)?

{{query}}

In your response, include:
1. Yes or No
2. A brief explanation''',

    "paper": '''Do you believe that the query below could be at least partially answered using content from the original study's paper/report or its primary data?

{{query}}

In your response, include:
1. Yes or No
2. A brief explanation''',

    "textbook": '''Do you believe that the query below could be at least partially answered using content from a relevant textbook specialized on the topic being discussed?

{{query}}

In your response, include:
1. Yes or No
2. A brief explanation'''
}
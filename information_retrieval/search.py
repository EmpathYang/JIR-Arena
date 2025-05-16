import os
import json
from pyserini.search.faiss import FaissSearcher
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.hybrid import HybridSearcher

from IR_CONSTANTS import SPARSE_INDEXING_DIR, DENSE_INDEXING_DIR, VIDEO_TYPE_TO_BASIC_KB_MAP, VIDEO_TYPE_ID_TO_ADDITIONAL_KNOWLEDGE_BASES_MAP, get_video_knowledge_bases, YOUTUBE_ID_TO_PAPER_ID_MAP
from ir_utils import dump_json_to_jsonl_file, get_video_type_by_youtube_id

KNOWLEDGE_BASE_TO_SPARSE_SEARCHER_MAP = {
    "wikipedia": LuceneSearcher(os.path.join(SPARSE_INDEXING_DIR, "wikipedia")),
    "arxiv": LuceneSearcher(os.path.join(SPARSE_INDEXING_DIR, "arxiv"))
}

KNOWLEDGE_BASE_TO_DENSE_SEARCHER_MAP = {
    "wikipedia": FaissSearcher(os.path.join(DENSE_INDEXING_DIR, "wikipedia"), 'castorini/tct_colbert-v2-hnp-msmarco'),
    "arxiv": FaissSearcher(os.path.join(DENSE_INDEXING_DIR, "arxiv"), 'castorini/tct_colbert-v2-hnp-msmarco')
}

KNOWLEDGE_BASE_TO_HYBRID_SEARCHER_MAP = {
    "wikipedia": HybridSearcher(KNOWLEDGE_BASE_TO_DENSE_SEARCHER_MAP["wikipedia"], KNOWLEDGE_BASE_TO_SPARSE_SEARCHER_MAP["wikipedia"]),
    "arxiv": HybridSearcher(KNOWLEDGE_BASE_TO_DENSE_SEARCHER_MAP["arxiv"], KNOWLEDGE_BASE_TO_SPARSE_SEARCHER_MAP["arxiv"])
}

def get_queries_metainfo(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get("judge_output", [])


def get_hybrid_searchers(sparse_indexing_path, dense_indexing_path):
    sparse_searcher = LuceneSearcher(sparse_indexing_path)
    dense_searcher = FaissSearcher(
        dense_indexing_path,
        'castorini/tct_colbert-v2-hnp-msmarco'
    )

    searcher = HybridSearcher(dense_searcher, sparse_searcher)
    return searcher

def get_video_hybrid_searchers(video_type, video_id):
    hybrid_searchers = []

    hybrid_searchers.extend([KNOWLEDGE_BASE_TO_HYBRID_SEARCHER_MAP[kb] for kb in VIDEO_TYPE_TO_BASIC_KB_MAP[video_type]])
    additional_knowledge_bases = VIDEO_TYPE_ID_TO_ADDITIONAL_KNOWLEDGE_BASES_MAP[video_type].get(video_id, [])
    for additional_knowledge_base in additional_knowledge_bases:
        sparse_indexing_path = os.path.join(SPARSE_INDEXING_DIR, additional_knowledge_base)
        dense_indexing_path = os.path.join(DENSE_INDEXING_DIR, additional_knowledge_base)
        hybrid_searchers.append(get_hybrid_searchers(sparse_indexing_path, dense_indexing_path))
    
    return hybrid_searchers

def hybrid_search(searcher, queries):

    hits = []
    for query in queries:
        hits.append(searcher.search(query))
    
    return hits

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--youtube_id_list', type=str, nargs='+', required=True)
    parser.add_argument('--metainfo_dir', type=str, required=True)
    parser.add_argument('--output_jir_references_dir', type=str, required=True)
    args = parser.parse_args()

    YOUTUBE_ID_LIST = args.youtube_id_list
    METAINFO_DIR = args.metainfo_dir
    OUTPUT_JIR_REFERENCES_DIR = args.output_jir_references_dir

    for youtube_id in YOUTUBE_ID_LIST:
        print(youtube_id)
        metainfo_filepath = os.path.join(METAINFO_DIR, youtube_id, "final_output.json")
        if not os.path.exists(metainfo_filepath):
            continue
        output_jir_references_filepath = os.path.join(OUTPUT_JIR_REFERENCES_DIR, youtube_id, "jir_references_raw.jsonl")
        video_type = get_video_type_by_youtube_id(youtube_id)
        video_id = YOUTUBE_ID_TO_PAPER_ID_MAP[youtube_id] if video_type == "paper" else youtube_id
        knowledge_bases = get_video_knowledge_bases(video_type, video_id)

        queries_metainfo = get_queries_metainfo(metainfo_filepath)
        queries = [query_metainfo["question"] if query_metainfo["question"] else query_metainfo["need"] for query_metainfo in queries_metainfo]
        video_hybrid_searchers = get_video_hybrid_searchers(video_type, video_id)
        video_hits_list = []
        for video_hybrid_searcher in video_hybrid_searchers:
            video_hits_list.append(hybrid_search(video_hybrid_searcher, queries))

        with open(output_jir_references_filepath, 'w') as jsonl_file:
            pass
        for i, query_hits_list in enumerate(zip(*video_hits_list)):
            query_metainfo = queries_metainfo[i]
            query_metainfo["content"] = {"references": {}}
            for query_hits in query_hits_list:
                knowledge_base = [kb for kb in knowledge_bases if kb in query_hits[0].docid][0]
                query_metainfo["content"]["references"][knowledge_base] = [(hits.docid, hits.score) for hits in query_hits]
            dump_json_to_jsonl_file(query_metainfo, output_jir_references_filepath)
    
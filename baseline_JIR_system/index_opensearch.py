import requests
import json
import re
import time
import glob
import os


# set up all data paths
data_path = "/shared/jir-arena/jsonl/"
#data_sources = ["arxiv/arxiv-metadata-oai-snapshot-Apr21.jsonl", "wikipedia/kilt_knowledgesource-Apr21.jsonl"]
# add papers
#papers = glob.glob(data_path + "paper/*")
#for file in papers:
#    data_sources.append("paper/" + os.path.basename(file) + "/all.jsonl")
## add textbook
data_sources = []
textbooks = glob.glob(data_path + "textbook/*")
for file in textbooks:
    data_sources.append("textbook/" + os.path.basename(file) + "/all.jsonl")
data_sources = [data_path + x for x in data_sources]

# set up index mapping
# match to docker container running opensearch
domain = "http://localhost:9200/"
index_name = "baseline_index"
resp = requests.head(domain + index_name)
if resp.status_code != 200:
    print("Index not found, adding index.")
    with open("opensearch_mappings.json", "r") as f:
        mapping = json.load(f)
    r = requests.put(domain + index_name, json=mapping)
    print(r.text)
else:
    print("Index is already setup: ", index_name)


"""
Formats

Arxiv line: {"id": "arxiv-0704.0002", "contents": "## Title\nSparsity-certifying Graph Decompositions\n\n## Abstract\nWe describe a new algorithm, the $(k,\\ell)$-pebble game with colors, ...}

	- TOTAL: 2710806

Wiki line: {"id": "wikipedia-290", "contents": "A\nA (named , plural \"As\", \"A's\", \"a\"s, \"a's\" or \"aes\") is the first letter and the first vowel of the modern English alphabet and the ISO basic Latin alphabe....}

	- TOTAL: 5903530

Paper line: {"id": "paper/20/liquid-in-practice.jsonl/1", "contents": "liquid democracy, Wisdom of Crowds, epistemic democracy, representation"}

	- find . -maxdepth 1 -type d -exec sh -c 'echo "$(wc -l < "$1/all.jsonl") $1"' _ {} \; | awk '{total += $1} END {print "Total lines:", total}'
	- TOTAL: 1452
    
    
Textbook line: {"id": "paper-textdatabook/front.jsonl/0", "contents": "information retrieval, text mining, text analysis, search engines, data mining, natural language processing\n\\tableofcontents\n\\listoffigures\n\\input{preface}"}

	- TOTAL: 655

    
SUM: 8,616,443
Reported: 8,616,420 (http://timan108.cs.illinois.edu:9200/baseline_index/_count)

-23 for blank line endings?

"""


batch_size = 100

for data_source in data_sources:
    print("Adding ", data_source)
    

    with open(data_source, "r") as f:

        start_time = time.time()
        bulk_insert = []
        for i,line in enumerate(f):
            json_line = json.loads(line)

            id = json_line.get("id", None)
            contents = json_line.get("contents", None)
            if not id or not contents:
                continue

            # remove all non-alphanumeric
            contents = re.sub("[^a-z0-9 ]", " ", contents.lower())

            inst = { "update": { "_index": index_name, "_id": id} }
            doc = { "doc": { "contents": contents}, "doc_as_upsert": True }

            bulk_insert.append(inst)
            bulk_insert.append(doc)

            if i % batch_size == 1:
                bulk_insert = "\n".join([json.dumps(x) for x in bulk_insert]) + "\n"
                resp = requests.post(domain + index_name + "/_bulk", data=bulk_insert, headers={'Content-Type': 'application/x-ndjson'})
                bulk_insert = []
            
            if i % 100000 == 1:
                print("Inserted 100,000 documents in ", time.time()- start_time)
                print("Total: ", i)
                start_time = time.time()

        # handle any remaining
        if len(bulk_insert):
            bulk_insert = "\n".join([json.dumps(x) for x in bulk_insert]) + "\n"
            resp = requests.post(domain + index_name + "/_bulk", data=bulk_insert, headers={'Content-Type': 'application/x-ndjson'})
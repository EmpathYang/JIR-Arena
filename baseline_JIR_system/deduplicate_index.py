import requests
import os
import glob
import json
import urllib.parse

OPENSEARCH_URL = 'http://localhost:9200'
INDEX_NAME = 'baseline_index'

# change to download all of the IDs in the index
if False:
    
    SCROLL_DURATION = '1m'
    PAGE_SIZE = 1000

    headers = {
        "Content-Type": "application/json"
    }

    # Step 1: Initial scroll request
    initial_query = {
        "_source": False,
        "query": {
            "match_all": {}
        },
        "size": PAGE_SIZE
    }

    response = requests.get(
        f"{OPENSEARCH_URL}/{INDEX_NAME}/_search?scroll={SCROLL_DURATION}",
        json=initial_query,
        headers=headers
    )
    data = response.json()

    scroll_id = data['_scroll_id']
    hits = data['hits']['hits']

    # Collect IDs
    all_ids = [hit['_id'] for hit in hits]

    # Step 2: Continue scrolling until no more hits
    while hits:
        scroll_query = {
            "scroll": SCROLL_DURATION,
            "scroll_id": scroll_id
        }
        scroll_response = requests.post(
            f"{OPENSEARCH_URL}/_search/scroll",
            json=scroll_query,
            headers=headers
        )
        scroll_data = scroll_response.json()
        hits = scroll_data['hits']['hits']
        scroll_id = scroll_data.get('_scroll_id', scroll_id)
        all_ids.extend(hit['_id'] for hit in hits)
        
    print(f"Total document IDs: {len(all_ids)}")

    with open("ids_in_index.txt", "w") as f:
        for id in all_ids:
            f.write(id + '\n')


if False:
    all_ids_in_data = []
    # set up all data paths
    data_path = "/shared/jir-arena/jsonl/"
    #data_sources = ["arxiv/arxiv-metadata-oai-snapshot-Apr21.jsonl", "wikipedia/kilt_knowledgesource-Apr21.jsonl"]
    # add papers
    #papers = glob.glob(data_path + "paper/*")
    #for file in papers:
    #    data_sources.append("paper/" + os.path.basename(file) + "/all.jsonl")
    # add textbook
    data_sources = []
    textbooks = glob.glob(data_path + "textbook/*")
    for file in textbooks:
        data_sources.append("textbook/" + os.path.basename(file) + "/all.jsonl")
    data_sources = [data_path + x for x in data_sources]


    for data_source in data_sources:        

        with open(data_source, "r") as f:

            bulk_insert = []
            for i,line in enumerate(f):
                json_line = json.loads(line)

                id = json_line.get("id", None)
                all_ids_in_data.append(id)


    with open("ids_in_data.txt", "w") as f:
        for id in all_ids_in_data:
            f.write(id + '\n')


all_ids_in_index = {}
all_ids_in_data = {}
with open("ids_in_data.txt", "r") as f:
    for line in f:
        line = line.strip()
        all_ids_in_data[line] = True
with open("ids_in_index.txt", "r") as f:
    for line in f:
        line = line.strip()
        if "arxiv" in line:
            continue
        if "wikipedia" in line:
            continue
        if "paper/" in line:
            continue
        all_ids_in_index[line] = True

print(len(all_ids_in_data))
print(len(all_ids_in_index))

for id in all_ids_in_index:
    if id not in all_ids_in_data:
        # all old ones are "paper-"
        print(id)
        encoded_id = urllib.parse.quote(id, safe='')
        url = f"{OPENSEARCH_URL}/{INDEX_NAME}/_doc/{encoded_id}"
        response = requests.delete(url)
        if response.status_code == 200:
            print("Document deleted successfully.")
        elif response.status_code == 404:
            print("Document not found.")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
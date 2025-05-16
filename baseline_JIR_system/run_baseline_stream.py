import requests
import glob
import json
import os
import re
import argparse
from functools import partial
from llms.claude import call_claude
from llms.gpt import call_gpt
from llms.deepseek_r1 import call_deepseek_r1

call_models = {"gpt-4o": partial(call_gpt, model_id="gpt-4o"), "DeepSeek-V3-0324": partial(call_deepseek_r1, model_id="DeepSeek-V3-0324"), "claude-3-7": partial(call_claude, model_id="arn:aws:bedrock:us-east-2:717279722391:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0")}

parser = argparse.ArgumentParser()

# if empty, assumes values are in environment (via Docker)
parser.add_argument("--gemini_key", required=False, help="gemini key")
parser.add_argument("--ref_only", required=False, help="ref only")
parser.add_argument("--model", required=False, help="model type")


chunk_lookback = 5

args = parser.parse_args()


headers = {
    "Content-Type": "application/json",
}



def call_llm(prompt):
     
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generation_config": {
            "temperature": 0.0
        }
    }

    
    response = requests.post("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + args.gemini_key, headers=headers, json=data)

    text = response.text
    return text

def search_api_arxiv(query, num_search_results=10):
    """
    Basic keyword serarch
    """
    domain = "http://timan108.cs.illinois.edu:9200"

    index_name = "/baseline_index"

    q = {
        "query": {
            "bool": {
            "must": [
                {
                "match": {
                    "contents": query
                }
                }
            ]
            }
        },
        "from": 0,
        "size": num_search_results,
    }

    resp = requests.get(domain + index_name + "/_search", json=q)
    resp_json = resp.json()
    references = []
    for i,hit in enumerate(resp_json["hits"]["hits"]):
        id = hit["_id"]
        references.append(id)
    return references


to_do = [os.path.basename(x) for x in glob.glob("output/*")]

path_to_transcripts = "data/transcripts/"
presentation_paths = glob.glob(path_to_transcripts + "*")


for model_id in call_models:
    if args.model not in model_id: continue
    for j,presentation_path in enumerate(presentation_paths):
        print(f"Model: {model_id}")
        presentation_name = os.path.basename(presentation_path)

        print(presentation_name)

        # for the small subset, changed here to only run for small subset
        # only do ones in completed
        if presentation_name.split(".")[0] not in to_do:
            continue

        print("Starting " + presentation_name, j)

        if not args.ref_only:

            presentation_transcript = json.load(open(presentation_path))
            if "message" in presentation_transcript:
                presentation_transcript = presentation_transcript["message"]
            presentation_transcript = presentation_transcript["transcript"]
            

            needs = []

            for i in range(0, len(presentation_transcript), chunk_lookback):
            
                
                transcript_for_prompt = ""

                for chunk in presentation_transcript[i:i+chunk_lookback]:

                    for word in chunk["chunks"]:
                        transcript_for_prompt += str(word["end"]) + " " + word["text"] + "\n"
                

                print("\tGenerating " + str(i) + " out of " + str(len(presentation_transcript)))


                prompt = """Given the transcript formatted by lines of timestamps/words, predict the questions that a listener might have. Respond in the following JSON format:

                {
                    "needs": [
                        
                        {
                            "start_time": <str, the earliest timestamp that the specific question may appear>,
                            "end_time": <str, the latest timestamp that the specific question may appear>,
                            "question": <str, the description of the question>
                        }
                    ]
                
                }
                
                Here is the transcript:
                
                <begin_transcript>
                
                <REPLACE_WITH_TRANSCRIPT>
                
                <end_transcript>
                
                As a reminder, given the above transcript with timestamps, predict the questions that a listener might have. Respond in the following JSON format:

                {
                    "needs": [
                        
                        {
                            "start_time": <str, the earliest timestamp that the specific question may appear>,
                            "end_time": <str, the latest timestamp that the specific question may appear>,
                            "question": <str, the description of the question>
                        }
                    ]
                
                }

                Make sure to follow these rules
                - Include all context in a question so that it can be answered without the transcript
                - Only include a single question/need per needs block.
                - Simulate a curious listener interested in the presented topic - questions should be reasonably likely to be asked by an attendee of this presentation.
                
                Respond only in the above JSON format, with nothing else.
            """

                prompt = prompt.replace("<REPLACE_WITH_TRANSCRIPT>", transcript_for_prompt)

                resp_json = call_models[model_id](prompt)


                """ FOR GEMINI
                llm_response = call_llm(prompt)

                try:
                    response_data = json.loads(llm_response)
                    resp_json = response_data["candidates"][0]["content"]["parts"][0]["text"]
                except Exception as e:
                    print(f"Error decoding JSON: {e}")
                    print(f"Raw response text: {llm_response}")
                    continue
                
                """
                
                try:
                    resp_json = json.loads(resp_json)
                except Exception as e:
                
                    resp_json = resp_json[8:-4]
                    resp_json = re.sub("\n", "", resp_json)
                    resp_json = re.sub("\t", "", resp_json)

                    try:
                        resp_json = json.loads(resp_json)
                    except Exception as e:
                        print("Error", e)
                        print(resp_json)
                        exit()
                        continue

                print("\tNeed generation completed")

                for need in resp_json["needs"]:
                    need["start_time"] = float(need["start_time"])
                    need["end_time"] = float(need["end_time"])
                    need["references"] = []

                needs += resp_json["needs"]


            resp_json = {"needs": needs}

            json.dump(resp_json, open("data/baseline_stream_runs_" + model_id + "/" + presentation_name, "w"))



        resp_json = json.load(open("data/baseline_stream_runs_" + model_id + "/" + presentation_name, "r"))
        for need in resp_json["needs"]:

            query = need["question"]
            formatted_query = re.sub("\?", query, query.lower())

            references = search_api_arxiv(formatted_query)

            need["references"] = references

        print("\tReferences found")

        json.dump(resp_json, open("data/baseline_stream_runs_" + model_id + "/" + presentation_name, "w"))    

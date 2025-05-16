import openai
from openai import OpenAI, AzureOpenAI
import time
import numpy as np
from PIL import Image
import base64
import io
import requests
import os

from llms.utils import DEFAULT_SYSTEM_PROMPT

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", None)
headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {OPENAI_API_KEY}"
}

def call_gpt(prompt, model_id="gpt-4o", system_prompt=DEFAULT_SYSTEM_PROMPT):
    client = OpenAI() if not AZURE_ENDPOINT else AzureOpenAI(azure_endpoint = AZURE_ENDPOINT, api_key=OPENAI_API_KEY, api_version="2024-02-15-preview")
    num_attempts = 0
    while True:
        if num_attempts >= 10:
            raise ValueError("OpenAI request failed.")
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.95,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            
            return response.choices[0].message.content.strip()
        except openai.AuthenticationError as e:
            print(e)
            return None
        except openai.RateLimitError as e:
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1
        except Exception as e:
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1

def arrange_message_for_gpt(item_list):
    def image_path_to_bytes(file_path):
        with open(file_path, "rb") as image_file:
            image_bytes = image_file.read()
        return image_bytes
    combined_item_list = []
    previous_item_is_text = False
    text_buffer = ""
    for item in item_list:
        if item[0] == "image":
            if len(text_buffer) > 0:
                combined_item_list.append(("text", text_buffer))
                text_buffer = ""
            combined_item_list.append(item)
            previous_item_is_text = False
        else:
            if previous_item_is_text:
                text_buffer += item[1]
            else:
                text_buffer = item[1]
            previous_item_is_text = True
    if item_list[-1][0] != "image" and len(text_buffer) > 0:
        combined_item_list.append(("text", text_buffer))
    content = []
    for item in combined_item_list:
        item_type = item[0]
        if item_type == "text":
            content.append({
                "type": "text",
                "text": item[1]
            })
        elif item_type == "image":
            if isinstance(item[1], str):
                image_bytes = image_path_to_bytes(item[1])
                image_data = base64.b64encode(image_bytes).decode("utf-8")
            elif isinstance(item[1], np.ndarray):
                image = Image.fromarray(item[1]).convert("RGB")
                width, height = image.size
                image = image.resize((int(0.5*width), int(0.5*height)), Image.LANCZOS)
                image_bytes = io.BytesIO()
                image.save(image_bytes, format='JPEG')
                image_bytes = image_bytes.getvalue()
                image_data = base64.b64encode(image_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                },
            })
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    return messages

def call_gpt_with_messages(messages, model_id="gpt-4o", system_prompt=DEFAULT_SYSTEM_PROMPT):
    client = OpenAI() if not AZURE_ENDPOINT else AzureOpenAI(azure_endpoint = AZURE_ENDPOINT, api_key=OPENAI_API_KEY, api_version="2024-02-15-preview")
    num_attempts = 0
    while True:
        if num_attempts >= 10:
            raise ValueError("OpenAI request failed.")
        try:
            if any("image" in c["type"] for m in messages for c in m["content"]):
                payload = {
                "model": "gpt-4-turbo",
                "messages": messages,
                }

                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                return response.json()["choices"][0]["message"].get("content", "").strip()
            else:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages if messages[0]["role"] == "system" else [{"role": "system", "content": system_prompt}] + messages,
                    temperature=0.5,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None
                )
                return response.choices[0].message.content.strip()
        except openai.AuthenticationError as e:
            print(e)
            return None
        except openai.RateLimitError as e:
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1
        except Exception as e:
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1

if __name__ == "__main__":
    print(call_gpt("hello",model_id="gpt-4o"))    
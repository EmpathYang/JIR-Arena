import requests
import os
import time
from llms.utils import DEFAULT_SYSTEM_PROMPT, INFORMATION_NEED_SYSTEM_PROMPT

from dotenv import load_dotenv
load_dotenv()

UIUC_CHAT_API_KEY = os.environ.get("UIUC_CHAT_API_KEY", None)

def call_uiuc_chat_llms(prompt, model_id="qwen2.5:14b-instruct-fp16", system_prompt=DEFAULT_SYSTEM_PROMPT):
    num_attempts = 0
    while True:
        if num_attempts >= 10:
            raise ValueError("UIUC-chat request failed.")
        try:
            data = {
            "model": model_id, # llama3.1:8b-instruct-fp16, deepseek-r1:14b-qwen-distill-fp16, qwen2.5:14b-instruct-fp16, qwen2.5:7b-instruct-fp16, Qwen/Qwen2.5-VL-72B-Instruct
            "api_key": UIUC_CHAT_API_KEY,
            "messages": [
                {
                "role": "system",
                "content": system_prompt
                },
                {
                "role": "user",
                "content": prompt
                }
            ],
            "course_name": "JIRArena",
            }

            response = requests.post(
            "https://uiuc.chat/api/chat-api/chat",
            json=data
            ).json()
            
            return response["message"].strip()
        except Exception as e:
            print("Error getting LLM response:", e)
            print("Waiting... for 5s...")
            time.sleep(5)
            num_attempts += 1

if __name__ == "__main__":
    print(call_uiuc_chat_llms("Hi"))
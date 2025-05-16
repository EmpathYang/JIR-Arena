import os
import time
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

from llms.utils import DEFAULT_SYSTEM_PROMPT
from dotenv import load_dotenv
load_dotenv()

endpoint = os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT", None)
api_key = os.environ.get("AZURE_AI_FOUNDRY_API_KEY", None)

def call_deepseek_r1(prompt, model_id="DeepSeek-R1", system_prompt=DEFAULT_SYSTEM_PROMPT):

    num_attempts = 0
    while True:
        if num_attempts >= 10:
            raise ValueError("OpenAI request failed.")
        try:
            client = ChatCompletionsClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(api_key),
            )

            response = client.complete(
                messages=[
                    SystemMessage(content=system_prompt),
                    UserMessage(content=prompt)
                ],
                max_tokens=2048,
                model=model_id
            )

            return response.choices[0].message.content
        except Exception as e:
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1

if __name__ == "__main__":
    print(call_deepseek_r1("Hi"))
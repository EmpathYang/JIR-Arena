import boto3
import json
import numpy as np
from PIL import Image
import base64
import io
import time

from llms.utils import DEFAULT_SYSTEM_PROMPT

def call_claude(prompt, model_id="arn:aws:bedrock:us-east-2:717279722391:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0", system_prompt=DEFAULT_SYSTEM_PROMPT, region_name="us-east-2"):
    # Create a Bedrock Runtime client in the AWS Region of your choice.
    client = boto3.client("bedrock-runtime", region_name=region_name)

    # Format the request payload using the model's native structure.
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.95,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    num_attempts = 0
    while True:
        if num_attempts >= 10:
            raise ValueError("OpenAI request failed.")
        try:
            # Invoke the model with the request.
            response = client.invoke_model(modelId=model_id, body=request)
            # Decode the response body.
            model_response = json.loads(response["body"].read())

            # Extract and print the response text.
            response_text = model_response["content"][0]["text"]
            return response_text

        except Exception as e:
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1

    

def arrange_message_for_claude(item_list):
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
                media_type = "image/png" # "image/jpeg"
                image_bytes = image_path_to_bytes(item[1])
                image_data = base64.b64encode(image_bytes).decode("utf-8")
            elif isinstance(item[1], np.ndarray):
                media_type = "image/jpeg"
                image = Image.fromarray(item[1]).convert("RGB")
                width, height = image.size
                image = image.resize((int(0.5*width), int(0.5*height)), Image.LANCZOS)
                image_bytes = io.BytesIO()
                image.save(image_bytes, format='JPEG')
                image_bytes = image_bytes.getvalue()
                image_data = base64.b64encode(image_bytes).decode("utf-8")
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data,
                },
            })
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    return messages

def call_claude_with_messages(messages, model_id="arn:aws:bedrock:us-east-2:717279722391:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0", system_prompt=DEFAULT_SYSTEM_PROMPT):
    # Create a Bedrock Runtime client in the AWS Region of your choice.
    client = boto3.client("bedrock-runtime", region_name="us-east-2")

    # Format the request payload using the model's native structure.
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.95,
        "system": system_prompt,
        "messages": messages,
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    num_attempts = 0
    while True:
        if num_attempts >= 10:
            raise ValueError("OpenAI request failed.")
        try:
            # Invoke the model with the request.
            response = client.invoke_model(modelId=model_id, body=request)
            # Decode the response body.
            model_response = json.loads(response["body"].read())

            # Extract and print the response text.
            response_text = model_response["content"][0]["text"]
            return response_text

        except Exception as e:
            print(e)
            print("Sleeping for 10s...")
            time.sleep(10)
            num_attempts += 1

    

if __name__ == "__main__":
    print(call_claude('Hi'))
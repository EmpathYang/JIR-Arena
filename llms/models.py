from llms.claude import call_claude
from llms.deepseek_r1 import call_deepseek_r1
from llms.gpt import call_gpt
from llms.uiuc_chat_llms import call_uiuc_chat_llms
from llms.utils import DEFAULT_SYSTEM_PROMPT

def call_model(prompt, model_id, system_prompt = DEFAULT_SYSTEM_PROMPT):
    if model_id in ["llama3.1:8b-instruct-fp16", "deepseek-r1:14b-qwen-distill-fp16", "qwen2.5:14b-instruct-fp16", "qwen2.5:7b-instruct-fp16", "Qwen/Qwen2.5-VL-72B-Instruct"]:
        return call_uiuc_chat_llms(prompt, model_id, system_prompt)
    
    elif model_id in ["DeepSeek-V3-0324", "DeepSeek-R1"]:
        return call_deepseek_r1(prompt, model_id, system_prompt)
    
    elif model_id in ["gpt-4o"]:
        return call_gpt(prompt, model_id, system_prompt)
    
    elif model_id in ["claude"]:
        return call_claude(prompt, system_prompt)

if __name__ == "__main__":
    print(call_model("hello", model_id="Qwen/Qwen2.5-VL-72B-Instruct")) 
    print(call_model("hello", model_id="DeepSeek-V3-0324")) 
    print(call_model("hello", model_id="gpt-4o")) 
    print(call_model("hello", model_id="claude")) 

    
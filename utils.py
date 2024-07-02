# Test with Maria conversation
# Test with the served Finetune checkpoints
from openai import OpenAI

def formattting_query_prompt_func_with_sys(prompt, sys_prompt,
                                           tokenizer,
                                           completion = "####Dummy-Answer"):
    """ 
    Formatting response according to specific llama3 chat template
    """
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion}
    ]
    format_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    query_prompt = format_prompt.split(completion)[0]
    return query_prompt

def formatting_query_prompt(message_history, 
                             sys_prompt,
                             tokenizer,
                             completion = "####Dummy-Answer"):
    """ 
    Formatting response according to specific llama3 chat template
    """
    messages = [{"role":"system", "content":sys_prompt}]
    messages.extend(message_history)
    messages.extend([{"role": "assistant", "content": completion}])
    format_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    query_prompt = format_prompt.split(completion)[0]
    return query_prompt

def get_response_from_finetune_checkpoint(format_prompt, do_print=True):
    """
    - Using vLLM to serve the fine-tuned Llama3 checkpoint
    - v6 full precision checkpoint is adopted here
    """
    # Serving bit of the client
    client = OpenAI(api_key="EMPTY", base_url="http://43.218.77.178:8000/v1")    
    # model_name = "Ksgk-fy/ecoach_philippine_v6_product_merge"
    model_name = "Ksgk-fy/genius_merge_v1"
    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # Streaming bit of the client
    stream = client.completions.create(
                model=model_name,
                prompt=format_prompt,
                max_tokens=512,
                temperature=0.0,
                stop=["<|eot_id|>"],
                stream=True,
                extra_body={
                    "repetition_penalty": 1.1,
                    "length_penalty": 1.0,
                    "min_tokens": 0,
                },
            )

    if do_print:
        print("Maria: ", end="")
    response_text = ""
    for response in stream:
        txt = response.choices[0].text
        if txt == "\n":
            continue
        if do_print:
            print(txt, end="")
        response_text += txt
    response_text += "\n"
    if do_print:
        print("")
    return response_text


from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer
from os import getenv
from huggingface_hub import login
login(getenv("HF_TOKEN"))
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_fast=False)


# SmartAss OnBoard
def parse_lex_transcript(soup):
    transcript_segments = soup.find_all('div', class_='ts-segment')
    transcript = ""
    curr_speaker = ""
    curr_response = ""

    for segment in transcript_segments:
        speaker = segment.find('span', class_='ts-name').get_text(strip=True)
        timestamp = segment.find('span', class_='ts-timestamp').get_text(strip=True)
        text = segment.find('span', class_='ts-text').get_text(strip=True)

        if curr_speaker == "":
            curr_speaker = speaker
            curr_response = text 
        elif curr_speaker != speaker:
            transcript += f"\n\n{curr_speaker}: {curr_response}"
            curr_speaker = speaker
            curr_response = text
        elif curr_speaker == speaker:
            curr_response = curr_response.strip() + f" {text}"  

    # Save the trnascript into txt files 
    with open("smart-ass/sm1.txt", "w") as file:
        file.write(transcript.strip())
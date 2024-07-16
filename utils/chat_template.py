from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

output = "../out/"

tokenizer = AutoTokenizer.from_pretrained(output)
model = AutoModelForCausalLM.from_pretrained(output, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map='cuda')

tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
tokenizer.add_tokens(['<|im_start|>', '<|im_end|>'], special_tokens=True)
tokenizer.pad_token = tokenizer.eos_token
emb = model.resize_token_embeddings(len(tokenizer))
seqlen = emb.weight.shape[1] # 1024

output_tk = "../out_tk/"
tokenizer.save_pretrained(output_tk)
model.save_pretrained(output_tk)

ds = load_dataset("teknium/OpenHermes-2.5")

def sharegpt_to_chatml(example):
    chatml_conversations = []
    for conv in example["conversations"]:
        if conv["from"] == "human":
            role = "user"
        elif conv["from"] == "system":
            role = "system"
        elif conv["from"] == "gpt":
            role = "assistant"
        else:
            role = "user"
        chatml_format = {"role": role, "content": conv["value"]}
        chatml_conversations.append(chatml_format)
    formatted = tokenizer.apply_chat_template(chatml_conversations, tokenize=False, add_generation_prompt=False)
    return {"text": formatted}

ds = ds.map(sharegpt_to_chatml)

ds.push_to_hub("jrahn/OpenHermes-2.5_chatml")


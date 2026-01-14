import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time 

SEED = 42  # TODO
torch.manual_seed(SEED)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=50,
)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
        

def generate_once(seed):
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

for s in [1, 2, 3, 4, 5]:
    print("SEED", s)
    print(generate_once(s))
    print("-" * 40)

def generate_once_penalty(seed):
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty = 2.0
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

for s in [1]:
    print("SEED WITH PENALTY", s)
    print(generate_once_penalty(s))
    print("-" * 40)

print("NO PENALTY seed=1")
print(generate_once(1))
print("WITH PENALTY seed=1")
print(generate_once_penalty(1))


def generate_once_tempb(seed):
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=0.1,
        top_k=50,
        top_p=0.95
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

for s in [1]:
    print("SEED WITH TEMP=0.1", s)
    print(generate_once_tempb(s))
    print("-" * 40)


def generate_once_temph(seed):
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=2.0,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

for s in [1]:
    print("SEED WITH TEMP=2.0", s)
    print(generate_once_temph(s))
    print("-" * 40)

for nb in [5, 10, 20]:
    start = time.time()
    out_b = model.generate(
        **inputs,
        max_length=50,
        num_beams=nb,
        early_stopping=True
    )
    elapsed = time.time() - start
    txt_beam = tokenizer.decode(out_b[0], skip_special_tokens=True)
    print(f"num_beams={nb} time_sec={elapsed}")
    print(txt_beam)
    print("-" * 40)



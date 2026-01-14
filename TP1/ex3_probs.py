from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import math

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

phrase = "Artificial intelligence is fascinating."
inputs = tokenizer(phrase, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # (1, seq_len, vocab)

# TODO: convertir en probabilités (softmax)
probs = torch.softmax(logits, dim=-1)

# On affiche P(token_t | tokens_<t) pour t>=1
input_ids = inputs["input_ids"][0]
for t in range(1, len(input_ids)):
    tok_id = input_ids[t].item()
    p = probs[0, t-1, tok_id].item()
    tok_txt = tokenizer.decode([tok_id])
    print(t, repr(tok_txt), f"{p:.3e}")

log_probs = torch.log_softmax(logits, dim=-1)
input_ids = inputs["input_ids"][0]

total_logp = 0.0
n = 0

for t in range(1, len(input_ids)):
    tok_id = input_ids[t].item()
    lp = log_probs[0, t-1, tok_id].item()
    total_logp += lp
    n += 1

avg_neg_logp = - (total_logp / n)
ppl = math.exp(avg_neg_logp)

print("total_logp:", total_logp)
print("avg_neg_logp:", avg_neg_logp)
print("perplexity:", ppl)

phrase_a = "Artificial intelligence is fascinating."
phrase_b = "Artificial fascinating intelligence is."

for s in [phrase_a, phrase_b]:
    inp = tokenizer(s, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
        lg = out.logits

    lp_all = torch.log_softmax(lg, dim=-1)
    ids = inp["input_ids"][0]

    tot = 0.0
    nn = 0
    for t in range(1, len(ids)):
        tid = ids[t].item()
        tot += lp_all[0, t-1, tid].item()
        nn += 1

    avg_nll = - (tot / nn)
    ppl_s = math.exp(avg_nll)

    print("\nPhrase:", repr(s))
    print("total_logp:", tot)
    print("perplexity:", ppl_s)

phrase_fr = "L'intelligence artificielle est fascinante."

inp_fr = tokenizer(phrase_fr, return_tensors="pt")
with torch.no_grad():
    out_fr = model(**inp_fr)
    lg_fr = out_fr.logits

lp_fr = torch.log_softmax(lg_fr, dim=-1)
ids_fr = inp_fr["input_ids"][0]

tot_fr = 0.0
nn_fr = 0
for t in range(1, len(ids_fr)):
    tid = ids_fr[t].item()
    tot_fr += lp_fr[0, t-1, tid].item()
    nn_fr += 1

avg_nll_fr = - (tot_fr / nn_fr)
ppl_fr = math.exp(avg_nll_fr)

print("\nPhrase FR:", repr(phrase_fr))
print("total_logp:", tot_fr)
print("perplexity:", ppl_fr)

prefix = "Artificial intelligence is"
inp = tokenizer(prefix, return_tensors="pt")

with torch.no_grad():
    out = model(**inp)
    logits2 = out.logits  # (1, seq_len, vocab)

# TODO: récupérer la distribution pour le prochain token (dernier pas de temps)
last_logits = logits2[0, -1, :]
last_probs = torch.softmax(last_logits, dim=-1)

topk = 10
vals, idx = torch.topk(last_probs, k=topk)

for p, tid in zip(vals.tolist(), idx.tolist()):
    print(repr(tokenizer.decode([tid])), f"{p:.3e}")
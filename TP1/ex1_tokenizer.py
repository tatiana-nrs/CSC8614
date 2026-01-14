from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
phrase = "Artificial intelligence is metamorphosing the world!"

# TODO: tokeniser la phrase
tokens = tokenizer.tokenize(phrase)
print(tokens)


# TODO: obtenir les IDs
token_ids = tokenizer.encode(phrase, add_special_tokens=False)
print("Token IDs:", token_ids)

print("Détails par token:")

for tid in token_ids:
    # TODO: décoder un seul token id
    txt = tokenizer.decode([tid])
    print(tid, repr(txt))


phrase2 = "GPT models use BPE tokenization to process unusual words like antidisestablishmentarianism."

tokens2 = tokenizer.encode(phrase2)
print("Détails par token:")
for tid in tokens2:
    txt = tokenizer.decode([tid])
    print(tid, repr(txt))

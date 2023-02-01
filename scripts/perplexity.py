from statistics import fmean
from typing import Type

import spacy
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
device = "cpu"


def check_probabilities(in_text):
    print(f"Perplexity of whole text: {perplexity(in_text)}")

    nlp = spacy.load('en_core_web_sm')
    tokens = nlp(in_text)
    perplexity_per_sentence = []
    for sent in tokens.sents:
        ppl = perplexity(sent.__str__())
        perplexity_per_sentence.append(ppl)
        print(f"Sentence: {sent}\nPerplexity: {ppl}")

    print(f"PPL: {perplexity_per_sentence}")
    print(f"Average perplexity: {fmean(perplexity_per_sentence)}")


def perplexity(in_text) -> Type[float]:
    max_length = model.config.n_positions
    stride = 512
    encodings = tokenizer(in_text, return_tensors='pt')
    seq_len = tokenizer(in_text, return_tensors='pt').data['input_ids'].size(1)

    nlls = []
    prev_end_loc = 0
    end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

    return ppl.item()

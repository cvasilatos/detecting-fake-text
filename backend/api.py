import numpy as np
import spacy
import torch
import time

from transformers import GPT2Tokenizer, GPTNeoForCausalLM, GPT2LMHeadModel, GPTNeoXPreTrainedModel, \
    GPTNeoXTokenizerFast, GPTNeoXForCausalLM

from backend.abstract_language_checker import AbstractLanguageChecker
from backend.class_register import register_api

import MoreThanSentiments as mts
import pandas as pd

from statistics import fmean


@register_api(name='gpt2')
@register_api(name='gpt2-medium')
@register_api(name='EleutherAI/gpt-neo-125M')
@register_api(name='EleutherAI/gpt-neo-1.3B')
@register_api(name='EleutherAI/gpt-neo-2.7B')
@register_api(name='EleutherAI/gpt-neox-20b')
class LM(AbstractLanguageChecker):
    def __init__(self, model_name_or_path):
        super(LM, self).__init__()
        self.enc = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        self.start_token = self.enc(self.enc.bos_token, return_tensors='pt').data['input_ids'][0]
        print(f"Loaded {model_name_or_path} model!")

    def check_probabilities(self, in_text, topk=40):
        token_ids = torch.concat([self.start_token, self.enc(in_text, return_tensors='pt').data['input_ids'][0]])

        all_probs = torch.softmax(self.model(token_ids.to(self.device)).logits[:-1].detach().squeeze(), dim=1)

        real_topk = self.calculate_real_top_k(all_probs, token_ids[1:])
        bpe_strings = self.calculate_bpe_strings(token_ids)
        pred_topk = self.calculate_pred_top_k(token_ids, all_probs, topk)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with open('/Users/cv43/sources/detecting-fake-text/txt/the_text.txt', 'w') as f:
            f.write(in_text)

        df = mts.read_txt_files('/Users/cv43/sources/detecting-fake-text/txt/')
        df['sent_tok'] = df.text.apply(mts.sent_tok)

        df['cleaned_data'] = pd.Series()
        for i in range(len(df['sent_tok'])):
            df['cleaned_data'][i] = [mts.clean_data(x, lower=True,
                                                    punctuations=True,
                                                    number=False,
                                                    unicode=True,
                                                    stop_words='True') for x in df['sent_tok'][i]]

        df['cleaned_data'] = df.text.apply(mts.clean_data, args=(True, True, False, True, False))
        df['Boilerplate'] = mts.Boilerplate(df['cleaned_data'], n=4, min_doc=0.2, get_ngram=False)
        df['Redundancy'] = mts.Redundancy(df.cleaned_data, n=10)
        df['Specificity'] = mts.Specificity(df.text)
        df['Relative_prevalence'] = mts.Relative_prevalence(df.text)

        with pd.option_context('display.max_rows', None,
                               'display.max_columns', 300,
                               'display.precision', 10,
                               ):
            print(df)

        with open('./result.csv', 'w') as f:
            df_str = df.to_string()
            f.write(df_str)

        print(f"Perplexity of whole text: {self.perplexity(in_text)}")

        nlp = spacy.load('en_core_web_sm')
        tokens = nlp(in_text)
        perplexity_per_sentence = []
        for sent in tokens.sents:
            perplexity = self.perplexity(sent.__str__())
            perplexity_per_sentence.append(perplexity)
            print(f"Sentence: {sent}\nPerplexity: {perplexity}")

        print(f"PPL: {perplexity_per_sentence}")
        print(f"Average perplexity: {fmean(perplexity_per_sentence)}")

        return {'bpe_strings': bpe_strings, 'real_topk': real_topk, 'pred_topk': pred_topk}

    def perplexity(self, in_text) -> float:
        max_length = self.model.config.n_positions
        stride = 512
        encodings = self.enc(in_text, return_tensors='pt')
        seq_len = self.enc(in_text, return_tensors='pt').data['input_ids'].size(1)

        nlls = []
        prev_end_loc = 0
        end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over input tokens.
                # Multiply it with trg_len to get the summation instead of average.
                # We will take average over all the tokens to get the true average
                # in the last step of this example.
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

        return ppl.data

    @staticmethod
    def calculate_real_top_k(all_probs, y):
        sorted_preds = torch.argsort(all_probs, dim=1, descending=True).cpu()
        real_topk_pos = list([int(np.where(sorted_preds[i] == y[i].item())[0][0]) for i in range(y.shape[0])])
        real_topk_probs_init = all_probs[np.arange(0, y.shape[0], 1), y].data.cpu().numpy().tolist()
        real_topk_probs = list(map(lambda x: round(x, 5), real_topk_probs_init))
        return list(zip(real_topk_pos, real_topk_probs))

    def calculate_bpe_strings(self, token_ids):
        return [self.postprocess(s) for s in self.enc.convert_ids_to_tokens(token_ids[:])]

    def calculate_pred_top_k(self, token_ids, all_probs, topk):
        topk_prob_values, topk_prob_inds = torch.topk(all_probs, k=topk, dim=1)
        pred_topk_init = [list(zip(self.enc.convert_ids_to_tokens(topk_prob_inds[i]),
                                   topk_prob_values[i].data.cpu().numpy().tolist()
                                   )) for i in range(token_ids[1:].shape[0])]
        return [[(self.postprocess(t[0]), t[1]) for t in pred] for pred in pred_topk_init]

    def postprocess(self, token):
        with_space = False
        with_break = False
        if token.startswith('Ġ'):
            with_space = True
            token = token[1:]
        elif token.startswith('â'):
            token = ' '
        elif token.startswith('Ċ'):
            token = ' '
            with_break = True

        token = '-' if token.startswith('â') else token
        token = '“' if token.startswith('ľ') else token
        token = '”' if token.startswith('Ŀ') else token
        token = "'" if token.startswith('Ļ') else token

        if with_space:
            token = '\u0120' + token
        if with_break:
            token = '\u010A' + token

        return token

    @staticmethod
    def top_k_logits(logits, k):
        """
        Filters logits to only the top k choices
        from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py
        """

        if k == 0:
            return logits

        values, _ = torch.topk(logits, k)
        min_values = values[:, -1]

        return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def main():
    raw_text = """In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously 
    unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns 
    spoke perfect English. 

    The scientist named the population, after their distinctive horn, Ovid’s Unicorn. These four-horned, silver-white 
    unicorns were previously unknown to science. 

    Now, after almost two centuries, the mystery of what sparked this odd phenomenon is finally solved.

    Dr. Jorge Pérez, an evolutionary biologist from the University of La Paz, and several companions, were exploring 
    the Andes Mountains when they found a small valley, with no other animals or humans. Pérez noticed that the 
    valley had what appeared to be a natural fountain, surrounded by two peaks of rock and silver snow. 

    Pérez and the others then ventured further into the valley. “By the time we reached the top of one peak, 
    the water looked blue, with some crystals on top,” said Pérez. 

    Pérez and his friends were astonished to see the unicorn herd. These creatures could be seen from the air without 
    having to move too much to see them – they were so close they could touch their horns. 

    While examining these bizarre creatures the scientists discovered that the creatures also spoke some fairly 
    regular English. Pérez stated, “We can see, for example, that they have a common ‘language,’ something like a 
    dialect or dialectic.” 

    Dr. Pérez believes that the unicorns may have originated in Argentina, where the animals were believed to be 
    descendants of a lost race of people who lived there before the arrival of humans in those parts of South America. 

    While their origins are still unclear, some believe that perhaps the creatures were created when a human and a 
    unicorn met each other in a time before human civilization. According to Pérez, “In South America, such incidents 
    seem to be quite common.” 

    However, Pérez also pointed out that it is likely that the only way of knowing for sure if unicorns are indeed 
    the descendants of a lost alien race is through DNA. “But they seem to be able to communicate in English quite 
    well, which I believe is a sign of evolution, or at least a change in social organization,” said the scientist. """

    raw_text = """In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously 
    unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns 
    spoke perfect English. """

    raw_text = """Quantum computing is a type of computing that uses quantum-mechanical phenomena, 
    such as superposition and entanglement, to perform operations on data. Quantum computers are different from 
    classical computers, which use bits to store and process information. Quantum computers use quantum bits, 
    or qubits, which can represent a 0, a 1, or both at the same time. This allows them to perform certain types of 
    calculations much faster than classical computers. One of the key features of quantum computers is their ability 
    to perform a process called quantum parallelism, which allows them to perform many calculations at the same time. 
    This makes them particularly well-suited for certain types of problems, such as factorizing large numbers or 
    searching large databases, which are difficult for classical computers to solve efficiently. 

    Quantum computers are still in the early stages of development, and it is not yet clear what their full potential 
    will be. However, they have the potential to revolutionize fields such as drug discovery, machine learning, 
    and financial modeling, among others. """

    raw_text = """However, we find that the current PPL cannot fairly evaluate the text quality (i.e., fluency) when 
    meeting the following scenarios. (i) The texts to be evaluated have different lengths (Meister and Cotterell, 
    2021b). In fact, text quality is not strictly related to length. However, we find that the PPL is sensitive to 
    text length, e.g., the PPL of short text is larger than long text. (ii) The texts to be evaluated have some 
    repeated span(s). Of course, sometimes creators use repeated text span(s) to express emphasis et al.. However, 
    PPL cannot distinguish between the right emphasis and abnormal repetition, and always foolishly assigns lower 
    scores to text that is not fluent but has repeated spans. (iii) The texts to be evaluated are sensitive to 
    punctuation marks. For example, we have two texts, the former ends with punctuation, and the latter deletes the 
    last punctuation. In theory, the qualified metric should compute the same or similar value. However, there is a 
    significant difference between the PPL values of those two texts. """
    '''
    Tests for GPT-2
    '''
    lm = LM('gpt2')
    start = time.time()
    raw_text = raw_text.replace("\n", "").strip()
    raw_text = " ".join(raw_text.split())
    payload = lm.check_probabilities(raw_text, topk=5)
    end = time.time()
    print(f"{end - start} Seconds for a check with GPT-2")


if __name__ == "__main__":
    main()

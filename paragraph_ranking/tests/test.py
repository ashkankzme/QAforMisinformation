import sys
sys.path.insert(1, '../')
from test_article2 import text, q, exp
from rank_gold_standard import biased_textrank
from utils import get_sentences


texts = get_sentences(text)
ranks, texts = biased_textrank(texts, exp)
sorted_rankings = sorted(zip(ranks, texts), key=lambda pair: pair[0])
top_paragraphs = [p for _, p in sorted_rankings[-10: ]] # top 10 pieces

for rank, p in zip(ranks, texts):
    if p in top_paragraphs:
        print('{}'.format(p))

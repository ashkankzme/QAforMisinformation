from test_article2 import text, q, exp
from rank_gold_standard import biased_textrank

ranks, paragraphs = biased_textrank(text, q, exp)
sorted_rankings = sorted(zip(ranks, paragraphs), key=lambda pair: pair[0])
top_paragraphs = [p for _, p in sorted_rankings[len(sorted_rankings)-12: ]] # top 10 pieces

for rank, p in zip(ranks, paragraphs):
    if p in top_paragraphs:
        print('{}'.format(p))

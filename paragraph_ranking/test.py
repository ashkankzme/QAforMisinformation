from test_article import text, q, exp
from rank_gold_standard import biased_textrank

ranks, paragraphs = biased_textrank(text, q, exp)

for rank, p in zip(ranks, paragraphs):
    print('{} : {}'.format(rank*10000, p))

from test_article import text, q, exp
from rank import rank_paragraphs

ranks, paragraphs = rank_paragraphs(text, q, exp)

for rank, p in zip(ranks, paragraphs):
    print(rank + ': ' + p)

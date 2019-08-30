from test_article import text, q, exp
from rank import rank_paragraphs

ranks, paragraphs = rank_paragraphs(text, q, exp)

print(zip(ranks, paragraphs))
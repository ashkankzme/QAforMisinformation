import torch
from reformer_pytorch import ReformerLM

ARTICLE_SEQ_LEN = 2048
EXPLANATION_SEQ_LEN = 128

encoder = ReformerLM(
    num_tokens=30000,  # as big as BERT vocabulary
    emb_dim=128,
    dim=1024,
    depth=12,
    heads=8,
    max_seq_len=ARTICLE_SEQ_LEN,
    fixed_position_emb=True,
    return_embeddings=True  # return output of last attention layer
).cuda()

decoder = ReformerLM(
    num_tokens=30000,  # as big as BERT vocabulary
    emb_dim=128,
    dim=1024,
    depth=12,
    heads=8,
    max_seq_len=EXPLANATION_SEQ_LEN,
    fixed_position_emb=True,
    causal=True
).cuda()

x = torch.randint(0, 20000, (1, ARTICLE_SEQ_LEN)).long().cuda()
yi = torch.randint(0, 20000, (1, EXPLANATION_SEQ_LEN)).long().cuda()

enc_keys = encoder(x)  # (1, 4096, 1024)
yo = decoder(yi, keys=enc_keys)  # (1, 4096, 20000)

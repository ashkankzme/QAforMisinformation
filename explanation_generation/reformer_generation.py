import torch
import torch.nn as nn
from transformers import BertTokenizer
from reformer_pytorch import ReformerLM, Autopadder

ARTICLE_SEQ_LEN = 2048
EXPLANATION_SEQ_LEN = 128
SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'


class ReformerForExplanationGeneration(nn.Module):
    def __init__(self, article_seq_len=ARTICLE_SEQ_LEN, explanation_seq_len=EXPLANATION_SEQ_LEN):
        super().__init__()

        self.encoder = ReformerLM(
            num_tokens=30000,  # as big as BERT vocabulary
            emb_dim=128,
            dim=1024,
            depth=12,
            heads=8,
            max_seq_len=article_seq_len,
            fixed_position_emb=True,
            return_embeddings=True  # return output of last attention layer
        ).cuda()

        self.encoder = Autopadder(self.encoder)

        self.decoder = ReformerLM(
            num_tokens=30000,  # as big as BERT vocabulary
            emb_dim=128,
            dim=1024,
            depth=12,
            heads=8,
            max_seq_len=explanation_seq_len,
            fixed_position_emb=True,
            causal=True
        ).cuda()

        self.decoder = Autopadder(self.decoder)

        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    def forward(self, article, question):
        input_text = CLS_TOKEN + ' ' + article + ' ' + SEP_TOKEN + ' ' + question + ' ' + SEP_TOKEN
        tokenized_input = self._tokenizer.tokenize(input_text)
        indexed_tokenized_input = torch.tensor(self._tokenizer.convert_tokens_to_ids(tokenized_input))

        encoder_out = self.encoder(indexed_tokenized_input)
        decoder_out = self.decoder()




x = torch.randint(0, 20000, (1, ARTICLE_SEQ_LEN)).long().cuda()
yi = torch.randint(0, 20000, (1, EXPLANATION_SEQ_LEN)).long().cuda()

enc_keys = encoder(x)  # (1, 4096, 1024)
yo = decoder(yi, keys=enc_keys)  # (1, 4096, 20000)

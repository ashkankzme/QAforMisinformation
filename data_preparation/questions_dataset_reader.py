from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token, WordTokenizer, Tokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import TextField, LabelField
from typing import Iterator, List, Dict, Optional
import json


@DatasetReader.register('questions')
class QuestionsDatasetReader(DatasetReader):
    """
    DatasetReader for Question Asnwering Task
    """

    def __init__(self, tokenizer: Optional[Tokenizer] = None,
                 token_indexers: Dict[str, TokenIndexer] = None, lazy=False) -> None:
        super().__init__(lazy)
        self.tokenizer = tokenizer or WordTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], label: int = None) -> Instance:
        fields = {
            'tokens': TextField(tokens, self.token_indexers),
            'label': LabelField(label=label, skip_indexing=True)
        }

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as question_file:
            articles = json.load(question_file)

        for article in articles:
            yield self.text_to_instance([x for x in self.tokenizer.tokenize(article['article'])],
                                        article['answer'])
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import TextField, LabelField
from typing import Iterator, List, Dict, Callable
import json


class QuestionsDatasetReader(DatasetReader):
    """
    DatasetReader for Question Asnwering Task
    """

    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None, lazy=False) -> None:
        super().__init__(lazy)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], label: int = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}

        label_field = LabelField(label=label)
        fields["label"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as question_file:
            articles = json.load(question_file)

        for article in articles:
            yield self.text_to_instance([Token(x) for x in self.tokenizer(article['article'])],
                                        article['answer'])

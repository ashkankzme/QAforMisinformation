/** You could basically use this config to train your own BERT classifier,
    with the following changes:

    1. change the `bert_model` variable to "bert-base-uncased" (or whichever you prefer)
    2. swap in your own DatasetReader. It should generate instances
       that look like {"tokens": TextField(...), "label": LabelField(...)}.

       You don't need to add the "[CLS]" or "[SEP]" tokens to your instances,
       that's handled automatically by the token indexer.
    3. replace train_data_path and validation_data_path with real paths
    4. any other parameters you want to change (e.g. dropout)
 */


local bert_model = "bert-base-uncased";

{
    "dataset_reader": {
        "lazy": false,
        "type": "questions",
        "tokenizer": {
            "type": 'word',
            "word_splitter": "bert-basic"
        },
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": bert_model,
                "do_lowercase": false
            }
        }
    },
    "train_data_path": "/home/ashkank/allennlp/allennlp/tests/fixtures/bert/q1_train.json",
    "validation_data_path": "/home/ashkank/allennlp/allennlp/tests/fixtures/bert/q1_dev.json",
    "model": {
        "type": "bert_for_classification",
        "bert_model": bert_model,
        "dropout": 0.05
    },
    "iterator": {
        "type": "basic",
        "batch_size": 5
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 1,
        "num_epochs": 3,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": -1
    }
}

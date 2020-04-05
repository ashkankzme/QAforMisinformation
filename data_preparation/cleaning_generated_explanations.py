import json


def clean_generated_explanations(begin, end):
    for i in range(begin, end):
        print("Loading data for question {}...".format(i))

        with open('../data/ttt/q{}_train.json'.format(i)) as train_file:
            train_set = json.load(train_file)

        with open('../data/ttt/q{}_test.json'.format(i)) as test_file:
            test_set = json.load(test_file)

        print("Data loading completed.")

        for article in train_set + test_set:
            if 'explanation_gpt2_sep_sat' in article and 'EXPLANATION:' in article['explanation_gpt2_sep_sat']:
                article['explanation_gpt2_sep_sat']
                after_explanation_token = article['explanation_gpt2_sep_sat'].split("EXPLANATION:",1)[1].strip()
                article['explanation_gpt2_sep_sat'] = after_explanation_token

        with open('../data/ttt/q{}_train.json'.format(i), 'w') as f:
            f.write(json.dumps(train_set))

        with open('../data/ttt/q{}_test.json'.format(i), 'w') as f:
            f.write(json.dumps(test_set))

        print('Results saved for question {}'.format(i))

if __name__ == "__main__":
    clean_generated_explanations(1, 2)
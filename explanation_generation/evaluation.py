import json

import numpy as np
from gpt2_generation import generated_text_is_meaningful, get_generation_prefix
from rouge import Rouge

rouge = Rouge()
scores = []

for i in range(1, 8):
    with open('../data/ttt/q{}_test.json'.format(i)) as test_file:
        articles = json.load(test_file)

    failure_rate = 0
    for article in articles:
        generation_prefix = get_generation_prefix(article['article'], article['question'])
        if not generated_text_is_meaningful(article['explanation_gpt2'], generation_prefix):
            failure_rate += 1
        try:
            scores.append(rouge.get_scores(article['explanation_gpt2'], article['explanation']))
        except:
            print('bad scrapping :/')

    print('Q{} stats:'.format(i))
    failure_rate /= len(articles)
    print('Failure Rate is: {}'.format(failure_rate))

    rouge_1 = [[score[0]['rouge-1']['f'], score[0]['rouge-1']['p'], score[0]['rouge-1']['r']] for score in scores]
    print('ROUGE 1 (f, p, r): {}'.format(np.mean(rouge_1, axis=0)))
    rouge_2 = [[score[0]['rouge-2']['f'], score[0]['rouge-2']['p'], score[0]['rouge-2']['r']] for score in scores]
    print('ROUGE 2 (f, p, r): {}'.format(np.mean(rouge_2, axis=0)))
    rouge_l = [[score[0]['rouge-l']['f'], score[0]['rouge-l']['p'], score[0]['rouge-l']['r']] for score in scores]
    print('ROUGE L (f, p, r): {}'.format(np.mean(rouge_l, axis=0)))
    print('#######################################################################')

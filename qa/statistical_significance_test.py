import json

from statsmodels.stats.contingency_tables import mcnemar


def find_contingency_table_idx(a, b, ground_truth):
    if a == b and b == ground_truth:
        return (0, 0)
    elif a == b and b != ground_truth:
        return (1, 1)
    elif a == ground_truth and b != ground_truth:
        return (0, 1)
    elif a != ground_truth and b == ground_truth:
        return (1, 0)

    return None


def element_under_k_exists(a, k):
    for row in a:
        for element in row:
            if element <= k:
                return True
    return False


def mcnemar_test(qid):
    with open('../data/ttt/q{}_test.json'.format(qid)) as test_file:
        test_set = json.load(test_file)

    # contingency table layout
    # correct/correct correct/incorrect
    # incorrect/correct incorrect/incorrect

    gpt2_textrank_contingency_table = [
        [0, 0],
        [0, 0]
    ]
    textrank_embedding_similarity_contingency_table = [
        [0, 0],
        [0, 0]
    ]

    for i, article in enumerate(test_set):
        gpt2_answer = article['answer_binary_gpt2']
        textrank_answer = article['answer_binary_textrank']
        embedding_similarity_answer = article['answer_binary_embedding_similarity']
        ground_truth = 1 if (qid != 5 and article['answer'] == 1) or (qid == 5 and article['answer'] == 0) else 0

        index = find_contingency_table_idx(gpt2_answer, textrank_answer, ground_truth)
        gpt2_textrank_contingency_table[index[0]][index[1]] += 1

        index = find_contingency_table_idx(textrank_answer, embedding_similarity_answer, ground_truth)
        textrank_embedding_similarity_contingency_table[index[0]][index[1]] += 1

    _, gpt2_textrank_pvalue = mcnemar(gpt2_textrank_contingency_table, exact=True) if element_under_k_exists(
        gpt2_textrank_contingency_table, 25) else mcnemar(gpt2_textrank_contingency_table, exact=False, correction=True)
    _, textrank_embedding_similarity_pvalue = mcnemar(textrank_embedding_similarity_contingency_table,
                                                      exact=True) if element_under_k_exists(
        textrank_embedding_similarity_contingency_table, 25) else mcnemar(
        textrank_embedding_similarity_contingency_table, exact=False, correction=True)

    return gpt2_textrank_pvalue, textrank_embedding_similarity_pvalue

if __name__ == "__main__":
    for i in range(1, 11):
        gpt2_textrank_pvalue, textrank_embedding_similarity_pvalue = mcnemar_test(i)
        print('Q#{}: gpt-textrank p value: {}, textrank-embedding p value: {}'.format(i, gpt2_textrank_pvalue, textrank_embedding_similarity_pvalue))

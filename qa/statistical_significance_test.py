import json

# from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats
import numpy as np


def get_accuracies():
    accuracies = []
    for qid in range(1, 10):
        qi_run = {
            'gpt2': [0] * 10,
            'textrank': [0] * 10,
            'embedding_similarity': [0] * 10
        }

    acc_line_prefix = 'Test Accuracy: '
    f1_pos_line_prefix = 'F1 pos: '
    f1_neg_line_prefix = 'F1 neg: '

    for qid in range(1, 10):
        qi_run = {
            'gpt2_acc': [0] * 10,
            'gpt2_f1_pos': [0] * 10,
            'gpt2_f1_neg': [0] * 10,
            'textrank_acc': [0] * 10,
            'textrank_f1_pos': [0] * 10,
            'textrank_f1_neg': [0] * 10,
            'embedding_similarity_acc': [0] * 10,
            'embedding_similarity_f1_pos': [0] * 10,
            'embedding_similarity_f1_neg': [0] * 10,
            'textrank_gpt2_acc': [0] * 10,
            'textrank_gpt2_f1_pos': [0] * 10,
            'textrank_gpt2_f1_neg': [0] * 10
        }

        for run_idx in range(1, 11):
            with open('../data/qa_runs/downstream_results{}/gpt{}.txt'.format(run_idx, qid), "r") as myfile:
                gpt2 = myfile.readlines()
            gpt2 = [line.strip() for line in gpt2]
            gpt2_acc = [float(line.split(acc_line_prefix, 1)[1]) for line in gpt2 if line.startswith(acc_line_prefix)][0]
            qi_run['gpt2_acc'][run_idx-1] = gpt2_acc
            gpt2_f1_pos = [float(line.split(f1_pos_line_prefix, 1)[1]) for line in gpt2 if line.startswith(f1_pos_line_prefix)][0]
            qi_run['gpt2_f1_pos'][run_idx - 1] = gpt2_f1_pos
            gpt2_f1_neg = [float(line.split(f1_neg_line_prefix, 1)[1]) for line in gpt2 if line.startswith(f1_neg_line_prefix)][0]
            qi_run['gpt2_f1_neg'][run_idx - 1] = gpt2_f1_neg

            with open('../data/qa_runs/downstream_results{}/textrank{}.txt'.format(run_idx, qid), "r") as myfile:
                textrank = myfile.readlines()
            textrank = [line.strip() for line in textrank]
            textrank_acc = [float(line.split(acc_line_prefix, 1)[1]) for line in textrank if line.startswith(acc_line_prefix)][0]
            qi_run['textrank_acc'][run_idx-1] = textrank_acc
            textrank_f1_pos = [float(line.split(f1_pos_line_prefix, 1)[1]) for line in textrank if line.startswith(f1_pos_line_prefix)][0]
            qi_run['textrank_f1_pos'][run_idx - 1] = textrank_f1_pos
            textrank_f1_neg = [float(line.split(f1_neg_line_prefix, 1)[1]) for line in textrank if line.startswith(f1_neg_line_prefix)][0]
            qi_run['textrank_f1_neg'][run_idx - 1] = textrank_f1_neg

            with open('../data/qa_runs/downstream_results{}/be{}.txt'.format(run_idx, qid), "r") as myfile:
                be = myfile.readlines()
            be = [line.strip() for line in be]
            be_acc = [float(line.split(acc_line_prefix, 1)[1]) for line in be if line.startswith(acc_line_prefix)][0]
            qi_run['embedding_similarity_acc'][run_idx-1] = be_acc
            be_f1_pos = [float(line.split(f1_pos_line_prefix, 1)[1]) for line in be if line.startswith(f1_pos_line_prefix)][0]
            qi_run['embedding_similarity_f1_pos'][run_idx - 1] = be_f1_pos
            be_f1_neg = [float(line.split(f1_neg_line_prefix, 1)[1]) for line in be if line.startswith(f1_neg_line_prefix)][0]
            qi_run['embedding_similarity_f1_neg'][run_idx - 1] = be_f1_neg

            with open('../data/qa_runs/downstream_results{}/tr-gpt{}.txt'.format(run_idx, qid), "r") as myfile:
                textrank_gpt2 = myfile.readlines()
            textrank_gpt2 = [line.strip() for line in textrank_gpt2]
            textrank_gpt2_acc = [float(line.split(acc_line_prefix, 1)[1]) for line in textrank_gpt2 if line.startswith(acc_line_prefix)][0]
            qi_run['textrank_gpt2_acc'][run_idx-1] = textrank_gpt2_acc
            textrank_gpt2_f1_pos = [float(line.split(f1_pos_line_prefix, 1)[1]) for line in textrank_gpt2 if line.startswith(f1_pos_line_prefix)][0]
            qi_run['textrank_gpt2_f1_pos'][run_idx - 1] = textrank_gpt2_f1_pos
            textrank_gpt2_f1_neg = [float(line.split(f1_neg_line_prefix, 1)[1]) for line in textrank_gpt2 if line.startswith(f1_neg_line_prefix)][0]
            qi_run['textrank_gpt2_f1_neg'][run_idx - 1] = textrank_gpt2_f1_neg

        accuracies.append(qi_run)

    return accuracies


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
        if 'answer_binary_gpt2' not in article:
            continue
        gpt2_answer = article['answer_binary_gpt2']
        textrank_answer = article['answer_binary_textrank']
        embedding_similarity_answer = article['answer_binary_embedding_similarity']
        ground_truth = 1 if (qid != 5 and article['answer'] == 1) or (qid == 5 and article['answer'] == 0) else 0

        index = find_contingency_table_idx(gpt2_answer, textrank_answer, ground_truth)
        gpt2_textrank_contingency_table[index[0]][index[1]] += 1

        index = find_contingency_table_idx(textrank_answer, embedding_similarity_answer, ground_truth)
        textrank_embedding_similarity_contingency_table[index[0]][index[1]] += 1

    gpt2_textrank_pvalue = mcnemar(gpt2_textrank_contingency_table, exact=True) if element_under_k_exists(
        gpt2_textrank_contingency_table, 25) else mcnemar(gpt2_textrank_contingency_table, exact=False, correction=True)
    # gpt2_textrank_pvalue = mcnemar(gpt2_textrank_contingency_table)
    # gpt2_textrank_pvalue = gpt2_textrank_pvalue.pvalue
    textrank_embedding_similarity_pvalue = mcnemar(textrank_embedding_similarity_contingency_table,
                                                      exact=True) if element_under_k_exists(
        textrank_embedding_similarity_contingency_table, 25) else mcnemar(
        textrank_embedding_similarity_contingency_table, exact=False, correction=True)
    # textrank_embedding_similarity_pvalue = mcnemar(textrank_embedding_similarity_contingency_table)
    # textrank_embedding_similarity_pvalue = textrank_embedding_similarity_pvalue.pvalue

    return gpt2_textrank_pvalue, textrank_embedding_similarity_pvalue


def acc_results_t_test():
    accuracies = get_accuracies()
    avg_acc = {
        'gpt2': 0,
        'textrank': 0,
        'embedding_similarity': 0,
        'textrank_gpt2': 0
    }
    avg_f1_pos = {
        'gpt2': 0,
        'textrank': 0,
        'embedding_similarity': 0,
        'textrank_gpt2': 0
    }
    avg_f1_neg = {
        'gpt2': 0,
        'textrank': 0,
        'embedding_similarity': 0,
        'textrank_gpt2': 0
    }

    for i, qi in enumerate(accuracies):
        gpt2_acc = qi['gpt2_acc']
        textrank_acc = qi['textrank_acc']
        embedding_similarity_acc = qi['embedding_similarity_acc']
        textrank_gpt2_acc = qi['textrank_gpt2_acc']
        _, gpt2_textrank_acc_pvalue = stats.ttest_ind(gpt2_acc, textrank_acc)
        _, textrank_embedding_similarity_acc_pvalue = stats.ttest_ind(textrank_acc, embedding_similarity_acc)
        _, textrank_gpt2_gpt2_acc_pvalue = stats.ttest_ind(textrank_gpt2_acc, gpt2_acc)
        print('Q{}, ACC: GPT-2/TextRank: {}, TextRank/EmbeddingSimilarity: {}, TextRank+GPT-2/GPT-2: {}'.format(i+1, gpt2_textrank_acc_pvalue < 0.05, textrank_embedding_similarity_acc_pvalue < 0.05, textrank_gpt2_gpt2_acc_pvalue < 0.05))
        gpt2_f1_pos = qi['gpt2_f1_pos']
        textrank_f1_pos = qi['textrank_f1_pos']
        embedding_similarity_f1_pos = qi['embedding_similarity_f1_pos']
        textrank_gpt2_f1_pos = qi['textrank_gpt2_f1_pos']
        _, gpt2_textrank_f1_pos_pvalue = stats.ttest_ind(gpt2_f1_pos, textrank_f1_pos)
        _, textrank_embedding_similarity_f1_pos_pvalue = stats.ttest_ind(textrank_f1_pos, embedding_similarity_f1_pos)
        _, textrank_gpt2_gpt2_f1_pos_pvalue = stats.ttest_ind(textrank_gpt2_f1_pos, gpt2_f1_pos)
        print('Q{}, F1 Pos: GPT-2/TextRank: {}, TextRank/EmbeddingSimilarity: {}, TextRank+GPT-2/GPT-2: {}'.format(i + 1, gpt2_textrank_f1_pos_pvalue < 0.05, textrank_embedding_similarity_f1_pos_pvalue < 0.05, textrank_gpt2_gpt2_f1_pos_pvalue < 0.05))
        gpt2_f1_neg = qi['gpt2_f1_neg']
        textrank_f1_neg = qi['textrank_f1_neg']
        embedding_similarity_f1_neg = qi['embedding_similarity_f1_neg']
        textrank_gpt2_f1_neg = qi['textrank_gpt2_f1_neg']
        _, gpt2_textrank_f1_neg_pvalue = stats.ttest_ind(gpt2_f1_neg, textrank_f1_neg)
        _, textrank_embedding_similarity_f1_neg_pvalue = stats.ttest_ind(textrank_f1_neg, embedding_similarity_f1_neg)
        _, textrank_gpt2_gpt2_f1_neg_pvalue = stats.ttest_ind(textrank_gpt2_f1_neg, gpt2_f1_neg)
        print('Q{}, F1 Neg: GPT-2/TextRank: {}, TextRank/EmbeddingSimilarity: {}, TextRank+GPT-2/GPT-2: {}'.format(i + 1, gpt2_textrank_f1_neg_pvalue < 0.05, textrank_embedding_similarity_f1_neg_pvalue < 0.05, textrank_gpt2_gpt2_f1_neg_pvalue < 0.05))
        avg_acc['gpt2'] += np.mean(gpt2_acc)/9
        avg_acc['textrank'] += np.mean(textrank_acc)/9
        avg_acc['embedding_similarity'] += np.mean(embedding_similarity_acc)/9
        avg_acc['textrank_gpt2'] += np.mean(textrank_gpt2_acc) / 9
        avg_f1_pos['gpt2'] += np.mean(gpt2_f1_pos) / 9
        avg_f1_pos['textrank'] += np.mean(textrank_f1_pos) / 9
        avg_f1_pos['embedding_similarity'] += np.mean(embedding_similarity_f1_pos) / 9
        avg_f1_pos['textrank_gpt2'] += np.mean(textrank_gpt2_f1_pos) / 9
        avg_f1_neg['gpt2'] += np.mean(gpt2_f1_neg) / 9
        avg_f1_neg['textrank'] += np.mean(textrank_f1_neg) / 9
        avg_f1_neg['embedding_similarity'] += np.mean(embedding_similarity_f1_neg) / 9
        avg_f1_neg['textrank_gpt2'] += np.mean(textrank_gpt2_f1_neg) / 9
        # print('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(
        #     np.mean(gpt2_acc), np.mean(gpt2_f1_pos), np.mean(gpt2_f1_neg),
        #     np.mean(textrank_acc), np.mean(textrank_f1_pos), np.mean(textrank_f1_neg),
        #     np.mean(textrank_gpt2_acc), np.mean(textrank_gpt2_f1_pos), np.mean(textrank_gpt2_f1_neg),
        #     np.mean(embedding_similarity_acc), np.mean(embedding_similarity_f1_pos), np.mean(embedding_similarity_f1_neg),
        # ))

    print(avg_acc)
    print(avg_f1_pos)
    print(avg_f1_neg)


if __name__ == "__main__":
    # for i in range(1, 11):
    #     gpt2_textrank_pvalue, textrank_embedding_similarity_pvalue = mcnemar_test(i)
    #     print('Q#{}: gpt-textrank p value: {}, textrank-embedding p value: {}'.format(i, gpt2_textrank_pvalue, textrank_embedding_similarity_pvalue))
    acc_results_t_test()

import re, json, string
from utils import get_bert_embeddings, cosine_similarity

translator = str.maketrans('', '', string.punctuation)


# returns all the text between qoutations
# in the input text as a list.
def get_text_between_qoutations(text):
    matches = re.findall(r'\"(.+?)\"', text)
    matches += re.findall(r'\“(.+?)\”', text)
    # matches is now ['String 1', 'String 2', 'String3']
    return matches


# for each question file
# read the ranked files
# separate the ones that have
# qoutations in their explanations
# see if any of the stuff between
# the qoutations are present in
# the extracted text.
# Return percentage of catches
# for each questions file.
def get_ranking_recall(articles):
    selected_articles = [a for a in articles if len(get_text_between_qoutations(a['explanation'])) > 0]

    found = 0
    for a in selected_articles:
        qoutes = [q.lower().strip().translate(translator) for q in get_text_between_qoutations(a['explanation'])]
        a['article'] = a['article'].translate(translator)
        for qoute in qoutes:
            if qoute in a['article'].lower().strip():
                found += 1
                break

    return found, len(selected_articles)


def calculate_hitrate(articles):
    hit = 0
    for article in articles:
        if article['answer'] in [0, 2] and 1 not in article['paragraph_relevance_learned_labels']:
            hit += 1
            continue

        paragraphs = article['paragraphs']
        paragraphs_embeddings = [get_bert_embeddings(p) for p in paragraphs]
        exp_embeddings = get_bert_embeddings(article['explanation'])
        similarities = [0] * len(paragraphs_embeddings)
        similarity_map = {}
        for i, pe in enumerate(paragraphs_embeddings):
            similarities[i] = cosine_similarity(pe, exp_embeddings)
            similarity_map[similarities[i]] = i

        top3_similarities = sorted(similarities, reverse=True)[:1]
        top3_paragraph_indices = [similarity_map[sim] for sim in top3_similarities]
        for i in top3_paragraph_indices:
            if article['paragraph_relevance_learned_labels'][i]:
                hit += 1
                break

    return hit/len(articles)


def main():
    for qid in range(1, 11):
        # with open('../data/question_answering_gold_standard_fine_grained_textrank_50/q{}_train.json'.format(
        #         qid)) as train_file:
        #     articles = json.load(train_file)
        #
        # with open('../data/question_answering_gold_standard_fine_grained_textrank_50/q{}_test.json'.format(
        #         qid)) as test_file:
        #     articles += json.load(test_file)
        #
        # with open('../data/question_answering_gold_standard_fine_grained_textrank_50/q{}_dev.json'.format(
        #         qid)) as dev_file:
        #     articles += json.load(dev_file)
        #
        # found, total = get_ranking_recall(articles)
        # print('found: {}, total: {}, percentage: {}'.format(found, total, found / total))

        with open('../data/paragraph_relevance_classification_input/q{}_test.json'.format(
                qid)) as test_file:
            articles = json.load(test_file)

        print('#Q{}: Hit Rate is == {}'.format(qid, calculate_hitrate(articles)))


if __name__ == "__main__":
    main()

import re, json, string
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


def main():
    for qid in range(1, 11):
        with open('../data/question_answering_gold_standard_fine_grained_textrank_70/q{}_train.json'.format(qid)) as train_file:
            articles = json.load(train_file)

        with open('../data/question_answering_gold_standard_fine_grained_textrank_70/q{}_test.json'.format(qid)) as test_file:
            articles += json.load(test_file)

        with open('../data/question_answering_gold_standard_fine_grained_textrank_70/q{}_dev.json'.format(qid)) as dev_file:
            articles += json.load(dev_file)

        found, total = get_ranking_recall(articles)
        print('found: {}, total: {}, percentage: {}'.format(found, total, found/total))

if __name__ == "__main__":
    main()
import re

# returns all the text between qoutations
# in the input text as a list.
def get_text_between_qoutations(text):
    matches = re.findall(r'\"(.+?)\"', text)
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
        qoutes = get_text_between_qoutations(a['explanation'])
        for qoute in qoutes:
            if qoute in a['article']:
                found += 1
                break

    return found/len(selected_articles)


def main():
    for i in range(1, 11):
        with open('../data/ranking/q{}_train.json'.format(qid)) as train_file:
            train = json.load(train_file)

        with open('../data/ranking/q{}_test.json'.format(qid)) as test_file:
            test = json.load(test_file)

        with open('../data/ranking/q{}_dev.json'.format(qid)) as dev_file:
            dev = json.load(dev_file)
        get_ranking_recall(i)

if __name__ == "__main__":
    main()
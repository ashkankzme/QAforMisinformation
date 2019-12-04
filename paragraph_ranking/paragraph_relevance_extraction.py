import json

from fuzzywuzzy import fuzz


Q_NUMBER = 7

def get_paragraphs(text):
    paragraphs = text.split('\n')
    paragraphs = [p for p in paragraphs if p and not p.isspace()]
    return paragraphs


print("Loading data...")

with open('../data/ttt/q{}_train.json'.format(Q_NUMBER)) as train_file:
    train_set = json.load(train_file)

with open('../data/ttt/q{}_test.json'.format(Q_NUMBER)) as test_file:
    test_set = json.load(test_file)

print("Data loading completed.")

# parsing the articles into paragraphs
for article in train_set + test_set:
    article['paragraphs'] = get_paragraphs(article['article'])

# initial classifier, based on string matching
sat_articles = [a for a in train_set if a['answer'] == 1]
articles_with_referencing_explanations = []

for article in sat_articles:
    exp = article['explanation']
    there_is_reference = exp.count('“') > 0 or exp.count('”') > 0 or exp.count('"') > 0
    if there_is_reference:
        articles_with_referencing_explanations.append(article)

print(len(articles_with_referencing_explanations))
print(len(sat_articles))

for article in articles_with_referencing_explanations:
    exp = article['explanation']
    exp = exp.replace('1clip_filelist.xml" rel="File-List"/>', '')
    exp.strip()

    c = exp.count("\"")
    c += exp.count('“')
    c += exp.count('”')

    if c % 2:
        continue

    i = 0
    j = 0
    quotations = ['“', '”', '"']
    while i < len(exp):
        character = exp[i]
        if character not in quotations:
            i += 1
            continue
        rest = exp[i + 1:]
        first = rest.find('“')
        if first < 0:
            first = len(exp)
        second = rest.find('”')
        if second < 0:
            second = len(exp)
        third = rest.find('"')
        if third < 0:
            third = len(exp)
        j = min(first + i + 1, second + i + 1, third + i + 1)
        if j == len(exp):
            break
        extract = exp[i + 1: j]
        if len(extract.split()) > 1:
            if 'extracts' not in article:
                article['extracts'] = []
            article['extracts'].append(extract)
        i = j + 1

incomplete_articles = []
for article in articles_with_referencing_explanations:
    if 'extracts' not in article:
        continue
    article['paragraph_relevance_extracted_labels'] = [0] * len(article['paragraphs'])
    extract_not_found = True
    for extract in article['extracts']:
        extract_not_found = True
        for i, paragraph in enumerate(article['paragraphs']):
            if fuzz.partial_ratio(extract, paragraph) >= 70 or fuzz.token_set_ratio(extract, paragraph) >= 70:
                article['paragraph_relevance_extracted_labels'][i] = 1
                extract_not_found = False
    if extract_not_found:
        incomplete_articles.append(article)

with open('../data/paragraph_relevance_classification_input/q{}_train.json'.format(Q_NUMBER), 'w') as f:
    f.write(json.dumps(train_set))

with open('../data/paragraph_relevance_classification_input/q{}_test.json'.format(Q_NUMBER), 'w') as f:
    f.write(json.dumps(test_set))

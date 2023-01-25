
import os
import sys
import time
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# import spacy
# nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# def spacy_lemmatize(doc):
#     search = nlp(doc)
#     output = " ".join([token.lemma_ for token in search])
#     return output

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk import word_tokenize          
from nltk.corpus import stopwords
wnl = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) 

def lemmatize(doc):
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    return " ".join(wnl.lemmatize(t) for t in word_tokenize(doc) if t not in ignore_tokens)

def find_no_of_keywords_used(scores):
    count = 0
    for score in scores:
        for value in score:
            if value > 0:
                count += 1
                break
    return count

def find_max_occurance_of_keywords(scores):
    max_occur = 0
    for score in scores:
        occur = 0
        for value in score:
            if value > 0:
                occur += 1
        if occur > max_occur:
            max_occur = occur
    return max_occur

def find_no_of_occurances_together(scores):
    occurances = {1:0, 2:0, 3:0, 4:0, 5:0}
    for score in scores:
        for index, value in enumerate(score):
            if value > 0:
                occurances[index+1] += 1
    occurances = sorted(occurances.values(), reverse=True)
    print(occurances)
    return list(occurances)[0]

df1 = pd.read_csv(sys.argv[1], encoding='unicode_escape', header=0)
df2 = pd.read_csv(sys.argv[2], header=0)

df = pd.DataFrame()
df.insert(0, "file_name", df1['file_name'])
df.insert(1, "keywords", df1['keywords'])
df.insert(2, "caption_1", df2['caption_1'])
df.insert(3, "caption_2", df2['caption_2'])
df.insert(4, "caption_3", df2['caption_3'])
df.insert(5, "caption_4", df2['caption_4'])
df.insert(6, "caption_5", df2['caption_5'])

vectorizer = TfidfVectorizer(stop_words=stop_words)
count = 0
output_df = pd.DataFrame(columns=['File_name', 'Total_no_of_keywords', 'No_of_keywords_used', 'Max_occurance_of_keywords', 'No_of_occurances_together'])

for index, row in df.iterrows():
    print('\n\n\n')
    count = count + 1
    t = time.time() * 1000
    document_scores = []
    keywords = row["keywords"].split(';')
    no_of_keywords_used = 0
    for keyword in keywords:
        keyword = lemmatize(keyword)
        # print(keyword)
        captions = []
        for i in range(5):
            s = 'caption_' + str(i+1)
            captions.append(lemmatize(row[s]))
        # print(captions)

        doc_vectors = vectorizer.fit_transform([keyword] + captions)
        # print(doc_vectors[0:1])
        # l2_df = pd.DataFrame(doc_vectors.toarray(), columns=vectorizer.get_feature_names())
        # print(l2_df)
        cosine_similarities = linear_kernel(doc_vectors[0:1], doc_vectors).flatten()
        document_scores.append([item.item() for item in cosine_similarities[1:]])

    print(document_scores)
    new_row = { 
            'File_name': row['file_name'],
            'Total_no_of_keywords': len(keywords),
            'No_of_keywords_used': find_no_of_keywords_used(document_scores),
            'Max_occurance_of_keywords': find_max_occurance_of_keywords(document_scores),
            'No_of_occurances_together': find_no_of_occurances_together(document_scores)
            }
    output_df = output_df.append(new_row, ignore_index=True)

    o = time.time() * 1000
    print("Time taken: ", o-t)
    # if count == 2:
    #     break

output_df.to_csv('tfidf_keyword_caption_relation.csv', index=True)

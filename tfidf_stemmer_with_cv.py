# encoding: utf-8

"""
https://www.kaggle.com/gaussmake1994/word-character-n-grams-tfidf-regressions-lb-051
"""

from __future__ import print_function
import pandas as pd
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.snowball import EnglishStemmer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from scipy import sparse

"""
Part 1. Research on dataset
"""

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
sample = pd.read_csv('./data/sample_submission.csv')

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# like：1 0 1 0 0 0，故此如果被分类为任何一种 toxic，那么 none 为 0；未被分为任何一钟 toxic，那么 none 就为 1
train['none'] = 1 - train[label_cols].max(axis=1)
# train.describe()

print(len(train),len(test))
# (95851, 226998)

lens = train.comment_text.str.len()
print(lens.mean(), lens.std(), lens.max())
# (395.3418639346486, 595.1020716997102, 5000)

train['comment_text'][pd.isnull(train['comment_text']) == True]
train['comment_text'].fillna('unknown', inplace=True)
test['comment_text'].fillna('unknown', inplace=True)

for label in label_cols:
    train[label] = train[label].astype(np.float32)
    print(label, (train[label] == 1.0).sum() / float(len(train)))

print(train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].corr())


"""
Part 2. preprocessing
"""

stemmer = EnglishStemmer()
decoder = lambda x: x.decode('cp850')
train['comment_text'] = train['comment_text'].apply(str.lower).apply(wordpunct_tokenize).apply(' '.join).apply(decoder).apply(stemmer.stem)
test['comment_text'] = test['comment_text'].apply(str.lower).apply(wordpunct_tokenize).apply(' '.join).apply(decoder).apply(stemmer.stem)


"""
Part 3. model
"""

# vec = CV(ngram_range=(1, 1), tokenizer=wordpunct_tokenize, max_features=1500000)
vec = TV(ngram_range=(1, 2))
model = LR()

# 每个文档使用 vocabulary 中的词的词频表示
train_doc = vec.fit_transform(train['comment_text'])
test_doc = vec.transform(test['comment_text'])


"""
Part 4. Validation
"""

# run model
preds = np.zeros((len(test), len(label_cols)))

n_splits = 5
total_losses = []
for i, j in enumerate(label_cols):
    # 对每个细分类分别进行预测
    preds_splits = np.zeros(len(test), dtype=np.float32)
    losses = []
    for trn, val in StratifiedKFold(n_splits=n_splits).split(train_doc, train[j]):
        model.fit(train_doc[trn], train[j][trn])
        test_scores = model.predict_proba(test_doc)[:, 1]
        preds_splits += test_scores
        val_scores = model.predict_proba(train_doc[val])[:, 1]
        losses.append(log_loss(train[j][val], val_scores))
    preds[:, i] = preds_splits / n_splits
    loss = np.mean(losses)
    total_losses.append(loss)
    print(j, 'val loss: ', loss)
print('overall val loss:', np.mean(total_losses))

# dump to csv
submit = pd.DataFrame({'id': sample['id']})
submission = pd.concat([submit, pd.DataFrame(preds, columns=label_cols)], axis=1)
submission.to_csv('submission.tfidf_lr.csv', index=False)

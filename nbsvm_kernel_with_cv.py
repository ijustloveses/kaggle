# encoding: utf-8

"""
https://www.kaggle.com/gaussmake1994/word-character-n-grams-tfidf-regressions-lb-051
"""

from __future__ import print_function
import pandas as pd
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from scipy import sparse

"""
Part 1. Research on dataset & preprocessing
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
Part 2. model
"""

vec = CV(ngram_range=(1, 1), tokenizer=wordpunct_tokenize, max_features=1500000)

# 每个文档使用 vocabulary 中的词的词频表示
train_doc = vec.fit_transform(train['comment_text'])
test_doc = vec.transform(test['comment_text'])

# 不考虑词频，只考虑是否出现过
train_x = train_doc.sign().astype(np.float32)
test_x = test_doc.sign().astype(np.float32)

# nb model;
def pr(x, y_i, y):
    # x 为 doc X vocab
    # 故此，这里 p 就是对 doc 求和，得到每个词在属于某个分类文档中出现过的文档数 (前面已经去掉了词频，改为是否出现)
    p = x[y == y_i].sum(0)
    # 除以属于某个分类的文档总数，得到各个词在属于某个分类的文档中出现的几率；为 len(vocabulary) 的矢量
    return (p + 1) / ((y == y_i).sum() + 1)


# nb-svm modle
def get_model(x, y):
    y = y.values
    # 这里，只处理二分类，也即对某个特定的细分类，只有属于该细分类和不属于该细分类两个可能的 label，即 1 & 0
    r = np.log(pr(x, 1, y) / pr(x, 0, y))
    x_nb = x.multiply(r)
    m = LR(C=0.1, dual=True)
    return m.fit(x_nb, y), r


"""
Part 3. Validation
"""

# run model
preds = np.zeros((len(test), len(label_cols)))

n_splits = 5
total_losses = []
for i, j in enumerate(label_cols):
    # 对每个细分类分别进行预测
    print('fit', j)
    preds_splits = np.zeros(len(test), dtype=np.float32)
    losses = []
    for trn, val in StratifiedKFold(n_splits=n_splits).split(train_x, train[j]):
        m, r = get_model(train_x[trn], train[j][trn])
        test_scores = m.predict_proba(test_x.multiply(r))[:, 1]
        preds_splits += test_scores
        val_scores = m.predict_proba(train_x[val].multiply(r))[:, 1]
        losses.append(log_loss(train[j][val], val_scores))
    preds[:, i] = preds_splits / n_splits
    loss = np.mean(losses)
    total_losses.append(loss)
    print('val loss: ', loss)
print('overall val loss:', np.mean(total_losses))

# dump to csv
submit = pd.DataFrame({'id': sample['id']})
submission = pd.concat([submit, pd.DataFrame(preds, columns=label_cols)], axis=1)
submission.to_csv('submission.nbsvm.cv.csv', index=False)

# encoding: utf-8

"""
https://www.kaggle.com/jhoward/nb-svm-baseline-and-basic-eda-0-06-lb 
https://www.kaggle.com/danofer/preprocess-toxic-comments-data-nb-svm-baseline
"""

from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.metrics import log_loss

import re
import string

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

# TODO: 检查被分类为 toxic 的 comment 和未被分类为 toxic 的比例；细分到任何一种 toxic
for label in label_cols:
    print(label, (train[label] == 1.0).sum() / float(len(train)))

# special string
re_tok = re.compile(r'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


# pad space on both sides of the special strings
def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()


n = train.shape[0]
vec = CV(ngram_range=(1, 1), tokenizer=tokenize, max_features=1500000)

# 每个文档使用 vocabulary 中的词的词频表示
train_doc = vec.fit_transform(train['comment_text'])
test_doc = vec.transform(test['comment_text'])

# 不考虑词频，只考虑是否出现过
x = train_doc.sign().astype(np.float32)
test_x = test_doc.sign().astype(np.float32)

# nb model;
def pr(y_i, y):
    # x 为 doc X vocab
    # 故此，这里 p 就是对 doc 求和，得到每个词在属于某个分类文档中出现过的文档数 (前面已经去掉了词频，改为是否出现)
    p = x[y == y_i].sum(0)
    # 除以属于某个分类的文档总数，得到各个词在属于某个分类的文档中出现的几率；为 len(vocabulary) 的矢量
    return (p + 1) / ((y == y_i).sum() + 1)


# nb-svm modle
def get_model(y):
    y = y.values
    # 这里，只处理二分类，也即对某个特定的细分类，只有属于该细分类和不属于该细分类两个可能的 label，即 1 & 0
    r = np.log(pr(1, y) / pr(0, y))
    x_nb = x.multiply(r)
    m = LR(C=0.1, dual=True)
    return m.fit(x_nb, y), r


# run model
preds = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    # 对每个细分类分别进行预测
    print('fit', j)
    m, r = get_model(train[j])
    preds[:, i] = m.predict_proba(test_x.multiply(r))[:, 1]


# dump to csv
submit = pd.DataFrame({'id': sample['id']})
submission = pd.concat([submit, pd.DataFrame(preds, columns=label_cols)], axis=1)
submission.to_csv('submission.nbsvm.csv', index=False)

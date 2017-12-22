# encoding: utf-8

"""
https://www.kaggle.com/CVxTz/keras-bidirectional-lstm-baseline-lb-0-051

./GoogleNews-vectors-negative300.bin word2vec model 有 900000000 个字符，过大，直接使用 max_features = 900000000 训练会导致下面错误：
UserWarning: Converting sparse IndexedSlices to a dense Tensor with 900000000 elements. This may consume a large amount of memory.

故此，这里做了过滤，只保留训练语料中有的 words
"""

from __future__ import print_function
import pandas as pd
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.snowball import EnglishStemmer
from sklearn.model_selection import StratifiedKFold

from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

import gensim

"""
Part 1. Research on dataset
"""

train = pd.read_csv('./data/train.csv')
# shuffle
train = train.sample(frac=1)
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

max_features = 20000
embed_size=128
maxlen = 1000

using_pretrained_word2vec_model = True

class EmbeddingWrapper(object):
    def __init__(self, modelfile):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(modelfile, binary=True, encoding='ISO-8859-1')

    def fit(self, texts):
        self.vocab = {}
        pos = 0
        self.matrix = []
        for sen in texts:
            for term in sen.split():
                if term in self.vocab:
                    continue
                if term in self.model.wv.vocab:
                    self.vocab[term] = pos
                    pos += 1
                    self.matrix.append(self.model[term])

    def transform(self, texts):
        return [[self.vocab[term] for term in sen.split() if term in self.vocab] for sen in texts]


if using_pretrained_word2vec_model:
    ew = EmbeddingWrapper('./GoogleNews-vectors-negative300.bin')
    ew.fit(list(train['comment_text']))

    assert len(ew.vocab) == len(ew.matrix)
    max_features = len(ew.vocab)
    embed_size = len(ew.matrix[0])
    print("max_features: {}  embed_size: {}".format(max_features, embed_size))
    
    list_tokenized_train = ew.transform(list(train['comment_text']))
    list_tokenized_test = ew.transform(list(test['comment_text']))
else:
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train['comment_text']))
    list_tokenized_train = tokenizer.texts_to_sequences(train['comment_text'])
    list_tokenized_test = tokenizer.texts_to_sequences(test['comment_text'])

train_doc = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen, padding='post', truncating='post')
test_doc = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen, padding='post', truncating='post')

"""
Part 3. model
"""

def get_model(ew=None):
    inp = Input(shape=(maxlen, ))
    if ew is None:
        x = Embedding(max_features, embed_size)(inp)
    else:
        x = Embedding(max_features, embed_size, weights=[np.array(ew.matrix)])(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation='sigmoid')(x)        # 6 个分类一次性学习，或者说一套参数同时去 fit 所有的分类，而不是 6 套参数
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',    # same as log loss
                  optimizer='adam', metrics=['accuracy'])
    return model


"""
Part 4. Validation
"""

if using_pretrained_word2vec_model:
    model = get_model(ew=ew)
else:
    model = get_model()
batch_size = 32
epochs = 10

file_path = "bilstm.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor='val_loss', mode='min', patience=3)

# 同时 fit in 6 个分类的结果
model.fit(train_doc, train[label_cols].values, batch_size=batch_size, epochs=epochs,
          validation_split=0.25, callbacks=[checkpoint, early])

model.load_weights(file_path)
preds = model.predict(test_doc)

# dump to csv
submit = pd.DataFrame({'id': sample['id']})
submission = pd.concat([submit, pd.DataFrame(preds, columns=label_cols)], axis=1)
submission.to_csv('submission.bilstm.csv', index=False)

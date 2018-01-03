# encoding: utf-8

from __future__ import print_function
import pandas as pd
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.snowball import EnglishStemmer
from sklearn.model_selection import StratifiedKFold

from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional, GlobalMaxPool1D, Dropout, Conv1D, MaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import log_loss

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
# label_cols = ['toxic', 'obscene', 'insult']
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
embed_size = 100
maxlen = 1000
nb_filters = 16
filter_length = 3
pool_length = 10
padding = 'valid'
activation = 'relu'
rnn_size = 128
batch_size = 64
epochs = 10

using_pretrained_word2vec_model = False

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
    ew.fit(list(train['comment_text']) + list(test['comment_text']))

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
    # x = Dropout(0.25)(x)
    x = Conv1D(filters=nb_filters, kernel_size=filter_length, padding=padding, activation=activation, strides=1)(x)
    x = MaxPooling1D(pool_size=pool_length)(x)
    x = LSTM(rnn_size)(x)
    x = Dense(len(label_cols), activation='sigmoid')(x)        # label_cols 个分类一次性学习，或者说一套参数同时去 fit 所有的分类，而不是 6 套参数
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',    # same as log loss
                  optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


"""
Part 4. Validation
"""

if using_pretrained_word2vec_model:
    model = get_model(ew=ew)
else:
    model = get_model()

file_path = "bilstm.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor='val_loss', mode='min', patience=3)

pos = int(len(train) * 0.75)
x_train = train_doc[0:pos]
y_train = train[label_cols].values[0:pos]
x_val = train_doc[pos:]
y_val = train[label_cols].values[pos:]
# 同时 fit in 6 个分类的结果
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          validation_data=(x_val, y_val), callbacks=[checkpoint, early])

model.load_weights(file_path)
val_preds = model.predict(x_val)
preds = model.predict(test_doc)

# show validataion result per class
val_preds = pd.DataFrame(val_preds, columns=label_cols)
val_labels = pd.DataFrame(y_val, columns=label_cols)
for label in label_cols:
    labels = val_labels[label]
    logits = val_preds[label]
    val_loss = log_loss(labels.values, logits.values)
    logits = logits.apply(lambda x: int(x + 0.5))  # 转为 0/1，而不再是 probs
    val_accuracy = (logits == labels).apply(lambda x: 1 if x else 0).mean(0)
    num_true_label = np.sum(labels.values)
    num_true_preds = np.sum(logits.values)
    num_correct_preds = (logits + labels).apply(lambda x: 1 if x == 2 else 0).sum()   # num. of logits == 1 && labels == 1
    print("validation for {} - val_loss: {}  val_accuracy: {}  precise: {}/{}  callback: {}/{}".format(label, val_loss, val_accuracy, num_correct_preds, num_true_preds, num_correct_preds, num_true_label))

# dump to csv
submit = pd.DataFrame({'id': sample['id']})
submission = pd.concat([submit, pd.DataFrame(preds, columns=label_cols)], axis=1)
submission.to_csv('submission.cnnlstm.csv', index=False)

### nbsvm_kernel.py

基础的测试脚本，保证整个能 run 起来，没有提交 kaggle

又由于没有做 train/validation split，故此也没有计算 validation set 中的分数


### nbsvm_kernel_with_cv.py

扩展上面的脚本，假如 train/validation split，作为后面其他脚本的基础

```
toxic val loss:  0.121750245023
severe_toxic val loss:  0.0297927094641
obscene val loss:  0.0655496291093
threat val loss:  0.0111529139225
insult val loss:  0.0815842198811
identity_hate val loss:  0.0266901754883

overall val loss: 0.0560866488147
```

没有提交 kaggle


### tfidf_stemmer_with_cv.py

使用 tfidf + logisticregression 模型

##### 尝试 1

- 原始文章直接 tfidf
- tfidf 模型 TV()
- lr 模型 LR()

可以想到结果应该一般

```
toxic val loss:  0.128803803912
severe_toxic val loss:  0.0290047694194
obscene val loss:  0.0740810058229
threat val loss:  0.0125433130537
insult val loss:  0.086322757157
identity_hate val loss:  0.0293220527367

overall val loss: 0.0600129503503
```

进而，假如 ngram_range=[1,2]，即 bi-grams 模型

```
overall val loss: 0.0698814754146
```

还不如上面 ...

##### 尝试 2

- 原始文字加入 EnglishStemmer().stem

```
toxic val loss:  0.129346617505
severe_toxic val loss:  0.0290045327309
obscene val loss:  0.0743639528991
threat val loss:  0.0126424325137
insult val loss:  0.0867264096812
identity_hate val loss:  0.0294252841158

overall val loss: 0.0602515382409
```

效果一般；同样，假如 ngram_range=[1,2] 结果反而变差

##### 尝试 3

- 原始文字加入 EnglishStemmer().stem
- tfidf 模型 TV()
- linear svm 模型 SVC(kernel='linear', probability=True)

```
toxic val loss:  0.116094507485
severe_toxic val loss:  0.0320463197279
obscene val loss:  0.0717947969416
threat val loss:  0.0118153071692
insult val loss:  0.090121870556
identity_hate val loss:  0.0307155570632

overall val loss: 0.0587647264905
```

##### 尝试 4

- 原始文字加入 EnglishStemmer().stem
- tfidf 模型 TV()
- rbf svm 模型 SVC(kernel='rbf', probability=True)

```
toxic val loss:  0.191127299909
severe_toxic val loss:  0.0767971587352
obscene val loss:  0.10445197993
threat val loss:  0.0202111852735
insult val loss:  0.128223987592
identity_hate val loss:  0.049422686965

overall val loss: 0.0950390497341
```
结果不如线性的好


### keras_bilstm.py

模型：bi-lstm with 2 dropout & 1 fc & sigmoid

##### 尝试 1

- 原始文字直接转 one-hot with max_features = 20000
- maxlen = 1000
- embedding size = 128
- post padding
- batch_size = 32
- validation_split = 1/4 

```
Epoch 3/10
71872/71888 [============================>.] - ETA: 0s - loss: 0.0416 - acc: 0.9841Epoch 00002: val_loss improved from 0.05249 to 0.05095, sav71888/71888 [==============================] - 3687s - loss: 0.0416 - acc: 0.9841 - val_loss: 0.0509 - val_acc: 0.9817
```

提交 Kaggle，0.051， rank 137/360

##### 尝试 2

- 使用 GoogleNews-vectors-negative300.bin word2vec。由于该模型中 words 太多，故此使用 training data 中的 words 做了过滤
- 过滤之后，one-hot with max_features = 88242
- embedding size = 300
- maxlen = 1000
- post padding
- batch_size = 32
- validation_split = 1/4 

```
71888/71888 [==============================] - 3609s - loss: 0.0777 - acc: 0.9745 - val_loss: 0.0613 - val_acc: 0.9786
Epoch 2/10
71888/71888 [==============================] - 3526s - loss: 0.0524 - acc: 0.9812 - val_loss: 0.0620 - val_acc: 0.9784ve
```

结果并不好，还不如随机初始化的 Embedding weights


### keras_cnnlstm.py

```
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 1000)              0
_________________________________________________________________
embedding_1 (Embedding)      (None, 1000, 128)         2560000
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 998, 16)           6160
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 99, 16)            0
_________________________________________________________________
lstm_1 (LSTM)                (None, 128)               74240
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 774
```

##### 尝试 1.

- 原始文字直接转 one-hot with max_features = 20000
- validation_split = 1/4 
- max_features = 20000
- embed_size = 128
- maxlen = 1000
- nb_filters = 16
- filter_length = 3
- pool_length = 10
- padding = 'valid'
- activation = 'relu'
- rnn_size = 128
- batch_size = 64

```
71888/71888 [==============================] - 90s - loss: 0.0527 - acc: 0.9813 - val_loss: 0.0523 - val_acc: 0.9817
Epoch 6/10
71872/71888 [============================>.] - ETA: 0s - loss: 0.0443 - acc: 0.9836Epoch 00005: val_loss improved from 0.05234 to 0.05131, sav71888/71888 [==============================] - 89s - loss: 0.0443 - acc: 0.9836 - val_loss: 0.0513 - val_acc: 0.9820
Epoch 7/10
71888/71888 [==============================] - 90s - loss: 0.0396 - acc: 0.9852 - val_loss: 0.0535 - val_acc: 0.9810rove
```
val_loss = 0.0513，不如上面的 bi-lstm 的模型好，但是训练速度快了 30 多倍

##### 尝试 2.

调整 embed_size：128 ==> 100

```
71888/71888 [==============================] - 90s - loss: 0.0487 - acc: 0.9823 - val_loss: 0.0509 - val_acc: 0.9815
Epoch 6/10
71872/71888 [============================>.] - ETA: 0s - loss: 0.0428 - acc: 0.9839Epoch 00005: val_loss improved from 0.05087 to 0.0571888/71888 [==============================] - 89s - loss: 0.0428 - acc: 0.9839 - val_loss: 0.0501 - val_acc: 0.9818
Epoch 7/10
71888/71888 [==============================] - 89s - loss: 0.0379 - acc: 0.9858 - val_loss: 0.0532 - val_acc: 0.9814rove
```
效果提高了一些，val_loss 降到 0.0501

加入个分类结果：
```
validation for toxic - val_loss: 0.111352190379  val_accuracy: 0.960522472145  precise: 1788/2216  callback: 1788/2306.0
validation for severe_toxic - val_loss: 0.0254191795236  val_accuracy: 0.989650711514  precise: 15/24  callback: 15/254.0
validation for obscene - val_loss: 0.0552941343344  val_accuracy: 0.979343154029  precise: 1021/1253  callback: 1021/1284.0
validation for threat - val_loss: 0.0138899027165  val_accuracy: 0.997162291867  precise: 0/0  callback: 0/68.0
validation for insult - val_loss: 0.0698304903719  val_accuracy: 0.970955222635  precise: 865/1203  callback: 865/1223.0
validation for identity_hate - val_loss: 0.0276717319624  val_accuracy: 0.992196302633  precise: 0/0  callback: 0/187.0
```
我们看到

1. 总体来看，恶意的评论数量过少，属于 imbalance 的分类问题
2. toxic，obscene，insult 三个分类的数据相对还能多一些，结果不是很难看，虽然远称不上好
3. severe_toxic，threat，identity_hate 三个分类的数据就少得可怜了，结果也非常差，就是说基本上不会被标记到。由于 TrueNagative 非常高，所以 log_loss 结果倒是比上面的 3 个分类还好看一些 ... 由于本次比赛评分规则的原因，故此其实就算这三个分类全部都标记为 Negative，影响其实也没有多大 ...

故此，相对重要的是，解决 2 中三个分类的问题


### keras_cnnlstm_perclass.py

前面 keras_cnnlstm 脚本对所有 6 个分类使用同一套网络参数；而 keras_cnnlstm_perclass 对每个分类分别 train 一个模型

```
validation for toxic - val_loss: 0.108966313533  val_accuracy: 0.960689396152  precise: 1565/1735  callback: 1565/2337.0
validation for severe_toxic - val_loss: 0.0406221511241  val_accuracy: 0.989859366523  precise: 0/0  callback: 0/243.0
validation for obscene - val_loss: 0.0608817155468  val_accuracy: 0.980511622084  precise: 1021/1232  callback: 1021/1277.0
validation for threat - val_loss: 0.0206942275454  val_accuracy: 0.996953636857  precise: 0/0  callback: 0/73.0
validation for insult - val_loss: 0.192788940922  val_accuracy: 0.949547218629  precise: 0/0  callback: 0/1209.0
validation for identity_hate - val_loss: 0.0472568635022  val_accuracy: 0.991862454618  precise: 0/0  callback: 0/195.0
```
我们看到，似乎还没有前面的结果好

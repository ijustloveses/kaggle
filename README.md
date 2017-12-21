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

### nbsvm_kernel.py

基础的测试脚本，保证整个能 run 起来，没有提交 kaggle

又由于没有做 train/validation split，故此也没有计算 validation set 中的分数

### nbsvm_kernel_with_cv.py

扩展上面的脚本，假如 train/validation split，作为后面其他脚本的基础

validation score:

```
fit toxic
val loss:  0.121750245023
fit severe_toxic
val loss:  0.0297927094641
fit obscene
val loss:  0.0655496291093
fit threat
val loss:  0.0111529139225
fit insult
val loss:  0.0815842198811
fit identity_hate
val loss:  0.0266901754883

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

validation score:

```
fit toxic
val loss:  0.128803803912
fit severe_toxic
val loss:  0.0290047694194
fit obscene
val loss:  0.0740810058229
fit threat
val loss:  0.0125433130537
fit insult
val loss:  0.086322757157
fit identity_hate
val loss:  0.0293220527367

overall val loss: 0.0600129503503
```

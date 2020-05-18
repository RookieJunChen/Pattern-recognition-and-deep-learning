## [综述](#1)

## [Sine函数预测](#2)

## [文本情感预测](#3)



<h2 id = "1">综述</h2>

数据（为了减小占用空间并未提交）应存放在**data**文件夹中:

```markdown
data
 └── 相关数据
```

**Model**文件夹中存放相关的模型：

```markdown
Model
 └── 相关模型
```

**MyDataSet**文件夹中存放构造数据集所使用的Dataset的子类：

```markdown
MyDataSet
 └── 构造数据集所使用的Dataset的子类
```

**mylog**文件夹中存放tensorboardX的绘图记录：

```markdown
mylog
 └──tensorboardX的绘图记录
```





<h2 id = "2">Sine函数预测</h2>

Adam优化器：

```markdown
python sine_predict.py
```

L-BFGS优化器：

```markdown
python sine_predict_lbfgs.py
```





<h2 id = "3">文本情感预测</h2>

先预处理数据：

```markdown
python preprocess.py
```

而后：

```markdown
python Text_sentiment_analysis.py
```


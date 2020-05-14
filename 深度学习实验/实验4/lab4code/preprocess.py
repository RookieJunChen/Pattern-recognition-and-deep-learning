import codecs
import numpy as np


# 读取.neg与.pos文件并用映射转换数据
def get_data(file, label, word2vector):
    # 使用UltraEdit查看到其编码格式为Windows-1252
    file = codecs.open(file, 'r', 'Windows-1252')
    text = file.readlines()  # 读取所有文本
    zero_padding = [0.0] * 50  # 零填充要使用
    max_len, data = 0, []
    # 按行处理
    for line in text:
        words = line.lower().strip().split()
        # 空的忽略
        if len(words) == 0:
            continue
        max_len = len(words) if len(words) > max_len else max_len
        words_vector = []  # 行向量
        for word in words:
            # 空的忽略
            if word == '':
                continue
            # 如果glove.6B.50d.txt不存在的映射就用零填充
            if word not in word2vector:
                words_vector.append(zero_padding)
                continue
            words_vector.append((word2vector[word]))
        data.append((words_vector, label))
    return data, max_len


# 读取glove.6B.50d.txt并构造单词到向量的字典的方法
def get_dict(file):
    word2vector = {}
    # 使用UltraEdit查看到其编码格式为utf-8
    word2vec_file = codecs.open(file, 'r', 'utf-8')
    word2vec_text = word2vec_file.readlines()
    for line in word2vec_text:
        temp = line.strip().split()
        # 空的忽略
        if len(temp) == 0:
            continue
        # 构造向量
        vector = [float(vector_item) for vector_item in temp[1:]]
        # 第一个是单词，而后是相应向量，由此构造出单词到向量的映射以供后面使用
        word2vector[temp[0].lower()] = vector
    return word2vector


# 因为读取文档并处理这一步过程不是很快，所以通过预处理解决后并将其保存下来提高训练速度
file_pos = 'data/rt-polarity.pos'
file_neg = 'data/rt-polarity.neg'
file_dict = 'data/glove.6B.50d.txt'

word2vector = get_dict(file_dict)
print('Construction mapping complete.')

# 以1作为positive的标签
data_pos, max_len1 = get_data(file_pos, 1, word2vector)
print('Positive text finished.')

# 以0作为negative的标签
data_neg, max_len2 = get_data(file_neg, 0, word2vector)
print('Negative text finished.')

max_len = max_len1 if max_len1 > max_len2 else max_len2  # 用作统一数据格式
zero_padding = [0.0] * 50

# 统一数据格式，不到最长长度的补零
data = []
data.extend(data_pos)
data.extend(data_neg)
for d, l in data:
    d.extend([zero_padding] * (max_len - len(d)))
print('Padding finish.')

# 划分训练集和测试集
train_num = int(len(data) * 0.8)  # 80%用作训练集，剩下的20%用作测试集
data_train = data[:train_num]
data_test = data[train_num:]

np.save('data/glove_train.npy', data_train)
np.save('data/glove_test.npy', data_test)

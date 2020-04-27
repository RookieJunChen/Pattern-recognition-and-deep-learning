import numpy as np
import warnings
from cv2 import cv2
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB  # 使用以高斯分布为先验分布的分类器


def rotate(image, degree):
    rows, cols = image.shape[:2]
    medium = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)
    dst = cv2.warpAffine(image, medium, (cols, rows))
    return dst


def expandimg(img, label):
    imgmake = img[0:]
    labelmake = label[0:]
    length = img.shape[0]
    for j in range(-10, 11, 2):  # 得到旋转-10度到10度间的图像
        for i in range(0, length):
            image = np.reshape(imgmake[i], (16, 16))
            img = np.r_[img, np.reshape(rotate(image, j), (1, 256))]
            labelnew = np.reshape(labelmake[i], (1, 10))
            label = np.r_[label, labelnew]
        print(j)
    print(img.shape)
    return img, label


def getTemplate(img, label):
    template = []
    for i in range(0, 10):
        tmp = np.where(label[:, i])
        choice = img[tmp[0], :]
        matrixsum = choice[0]
        for j in range(1, choice.shape[0]):
            matrixsum = np.add(matrixsum, choice[j])
        mean = matrixsum / (choice.shape[0])
        template.append(mean)
    return template


def getY(label):
    Y = []
    for i in range(0, label.shape[0]):
        Y.append(np.argwhere(label[i] == 1)[0][0])
    return Y


def SVMtrain(img, Y):
    clf = svm.SVC(C=10.0, kernel='rbf', gamma=0.03, probability=True)
    clf.fit(img, Y)
    return clf


def LDAtrain(img, Y):
    ldamod = LinearDiscriminantAnalysis(n_components=9)
    return ldamod.fit(img, Y)


def naive_bayestrain(img, Y):
    GaussianClassifier = GaussianNB()
    # GaussianClassifier = BernoulliNB()
    GaussianClassifier.fit(img, Y)
    return GaussianClassifier


# warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


semeion = np.loadtxt("semeion.data")
img = semeion[:, 0:256]
label = semeion[:, 256:266]

img, label = expandimg(img, label) # 数据增强

# print(rotate(image).shape)
# cv2.imshow("display", imgs)
# cv2.waitKey(0)

# 划分训练集与测试集
imgtrain = img[1594:]
labeltrain = label[1594:]
imgtest = img[:1594]
labeltest = label[:1594]

# PCA降维
pca = PCA(n_components=120)
pcaimg = pca.fit_transform(img)
# print(pca.explained_variance_ratio_)
pcatrain = pcaimg[1594:]
pcatest = pcaimg[:1594]

# LDA降维
ldamod = LDAtrain(imgtrain, getY(labeltrain))
ldatrain = ldamod.transform(imgtrain)
ldatest = ldamod.transform(imgtest)

# 模板匹配法
template = getTemplate(imgtrain, labeltrain) # 构建模板
sum1 = 0
for i in range(0, imgtest.shape[0]):
    # 匹配模板
    myguess = 0
    for j in range(1, 10):
        if np.linalg.norm(imgtest[i] - template[myguess]) > np.linalg.norm(imgtest[i] - template[j]):
            myguess = j
    # 判断是否与标签相符
    classnum = np.argwhere(labeltest[i] == 1)[0][0]
    if myguess == classnum:
        sum1 += 1
print("Template Matching:\t" + str(sum1 / imgtest.shape[0]))

# 模板匹配法 + PCA
template = getTemplate(pcatrain, labeltrain)
sum2 = 0
for i in range(0, pcatest.shape[0]):
    # 匹配模板
    myguess = 0
    for j in range(1, 10):
        if np.linalg.norm(pcatest[i] - template[myguess]) > np.linalg.norm(pcatest[i] - template[j]):
            myguess = j
    # 判断是否与标签相符
    classnum = np.argwhere(labeltest[i] == 1)[0][0]
    if myguess == classnum:
        sum2 += 1
print("PCA_Template Matching:\t" + str(sum2 / imgtest.shape[0]))

# 模板匹配法 + LDA
template = getTemplate(ldatrain, labeltrain)
sum3 = 0
for i in range(0, ldatest.shape[0]):
    # 匹配模板
    myguess = 0
    for j in range(1, 10):
        if np.linalg.norm(ldatest[i] - template[myguess]) > np.linalg.norm(ldatest[i] - template[j]):
            myguess = j
    # 判断是否与标签相符
    classnum = np.argwhere(labeltest[i] == 1)[0][0]
    if myguess == classnum:
        sum3 += 1
print("LDA_Template Matching:\t" + str(sum3 / imgtest.shape[0]))

# 朴素贝叶斯
bayes = naive_bayestrain(imgtrain, getY(labeltrain))
bayespredict = bayes.predict(imgtest)
sum5 = 0
for i in range(0, len(bayespredict)):
    # 判断是否与标签相符
    classnum = np.argwhere(labeltest[i] == 1)[0][0]
    if bayespredict[i] == classnum:
        sum5 += 1
print("Bayes:\t" + str(sum5 / imgtest.shape[0]))

# 朴素贝叶斯 + LDA
bayes = naive_bayestrain(ldatrain, getY(labeltrain))
bayespredict = bayes.predict(ldatest)
sum6 = 0
for i in range(0, len(bayespredict)):
    # 判断是否与标签相符
    classnum = np.argwhere(labeltest[i] == 1)[0][0]
    if bayespredict[i] == classnum:
        sum6 += 1
print("LDA_Bayes:\t" + str(sum6 / imgtest.shape[0]))

# imgtrain = img[0:2000]
# labeltrain = label[0:2000]
# imgtest = img[2000:2200]
# labeltest = label[2000:2200]

# SVM
clf = SVMtrain(imgtrain, getY(labeltrain))  # SVM训练
svmpredict = clf.predict(imgtest)  # 进行SVM预测
sum4 = 0
for i in range(0, len(svmpredict)):
    # 判断是否与标签相符
    classnum = np.argwhere(labeltest[i] == 1)[0][0]
    if svmpredict[i] == classnum:
        sum4 += 1
print("SVM:\t" + str(sum4 / imgtest.shape[0]))

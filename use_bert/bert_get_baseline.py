# -*- coding: utf-8 -*-
"""
@Time    : 2020/4/8 18:23
@Author  : hejipei
"""

'''
提示：受GPU性能的影响，只能运行基础版的bert预训练模型，若出现OOM 适当调整batch_size,maxlen，如果使用cup运行为非常忙
我用的是numpy==1.16.4其他版本可能会有提示
'''

import pandas as pd
import codecs, gc
import numpy as np
from sklearn.model_selection import KFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.metrics import top_k_categorical_accuracy
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn import metrics
# 读取训练集和测试集
from sklearn.model_selection import train_test_split

# 参数配置
maxlen      = 100   # 设置序列长度为100，要保证序列长度不超过512
Batch_size  = 16    #批量运行的个数
Epoch       = 1     #迭代次数

def get_train_test_data():
    train_df = pd.read_excel(r'data\data_train.xlsx' ).astype(str)
    test_df = pd.read_excel(r'data\data_test.xlsx').astype(str)

    # 训练数据、测试数据和标签转化为模型输入格式
    DATA_LIST = []
    for data_row in train_df.iloc[:].itertuples():
        DATA_LIST.append((data_row.contents, to_categorical(data_row.labels, 2)))
    DATA_LIST = np.array(DATA_LIST)

    DATA_LIST_TEST = []
    for data_row in test_df.iloc[:].itertuples():
        DATA_LIST_TEST.append((data_row.contents, to_categorical(data_row.labels, 2)))
    DATA_LIST_TEST = np.array(DATA_LIST_TEST)

    data = DATA_LIST
    data_test = DATA_LIST_TEST

    X_train,X_valid = train_test_split(data,test_size=0.2,random_state = 0)
    return X_train,X_valid,data_test

# 预训练好的模型 roberta_wwm_ext_large
# config_path     = r'roberta_wwm_ext_large\bert_config.json' # 加载配置文件
# checkpoint_path = r'roberta_wwm_ext_large\bert_model.ckpt'
# dict_path       = r'roberta_wwm_ext_large\vocab.txt'

# 预训练好的模型 bert base
config_path     = r'bert\bert_config.json' # 加载配置文件
checkpoint_path = r'bert\bert_model.ckpt'
dict_path       = r'bert\vocab.txt'


def get_token_dict():
    """
    # 将词表中的字编号转换为字典
    :return: 返回自编码字典
    """
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict

# 重写tokenizer
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # 用[unused1]来表示空格类字符
            else:
                R.append('[UNK]')  # 不在列表的字符用[UNK]表示   UNK是unknown的意思
        return R

# 获取新的tokenizer
tokenizer = OurTokenizer(get_token_dict())


def seq_padding(X, padding=0):
    """
    :param X: 文本列表
    :param padding: 填充为0
    :return: 让每条文本的长度相同，用0填充
    """
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([ np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])


class data_generator:
    """
    data_generator只是一种为了节约内存的数据方式
    """
    def __init__(self, data, batch_size=Batch_size, shuffle=True):
        """
        :param data: 训练的文本列表
        :param batch_size:  每次训练的个数
        :param shuffle: 文本是否打乱
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))

            if self.shuffle:
                np.random.shuffle(idxs)

            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y[:, 0, :]
                    [X1, X2, Y] = [], [], []



def acc_top2(y_true, y_pred):
    """
    :param y_true: 真实值
    :param y_pred: 训练值
    :return: # 计算top-k正确率,当预测值的前k个值中存在目标类别即认为预测正确
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


# bert模型设置
def build_bert(nclass):
    """
    :param nclass: 文本分类种类
    :return: 构建的bert模型
    """
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)  # 加载预训练模型

    for l in bert_model.layers:
        l.trainable = True
    #构建模型
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
    p = Dense(nclass, activation='softmax')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-5),  # 用足够小的学习率
                  metrics=['accuracy', acc_top2])
    print(model.summary())
    return model


def run_kb():
    """
    训练模型
    :return: 验证预测集，测试预测集，训练号的模型
    """
    # 搭建模型参数
    print('正在加载模型，请耐心等待....')
    model = build_bert(2)  # 二分类模型
    print('模型加载成功，开始训练....')
    early_stopping = EarlyStopping(monitor='val_acc', patience=3)  # 早停法，防止过拟合
    plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=2)  # 当评价指标不在提升时，减少学习率
    checkpoint = ModelCheckpoint(r'use_bert\bert_dump1.hdf5', monitor='val_acc', verbose=2,
                                 save_best_only=True, mode='max', save_weights_only=True)  # 保存最好的模型
    # 获取数据并文本序列化
    X_train, X_valid, data_test = get_train_test_data()
    train_D = data_generator(X_train, shuffle=True)
    valid_D = data_generator(X_valid, shuffle=True)
    test_D = data_generator(data_test, shuffle=False)

    # 模型训练
    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=Epoch,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D),
        callbacks=[early_stopping, plateau, checkpoint],
        )
    # 对验证集和测试集进行预测
    valid_D = data_generator(X_valid, shuffle=False)
    train_model_pred = model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=1)
    test_model_pred  = model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=1)
    # 将预测概率值转化为类别值
    train_pred = [np.argmax(x) for x in train_model_pred]
    test_pred = [np.argmax(x) for x in test_model_pred]
    y_true = [np.argmax(x) for x in X_valid[:, 1]]

    return train_pred,test_pred,y_true,model,data_test


def bk_metrics(y_true,y_pred,type ='metrics'):
    """
    :param y_true: 真实值
    :param y_pred: 预测值
    :param type: 预测种类
    :return: 评估指标
    """
    print(type,'...')
    print(metrics.confusion_matrix(y_true,y_pred))
    print('准确率：',metrics.accuracy_score(y_true,y_pred))
    print('类别精度：',metrics.precision_score(y_true,y_pred,average = None)) #不求平均
    print('宏平均精度：',metrics.precision_score(y_true,y_pred,average = 'macro'))
    print('微平均召回率:',metrics.recall_score(y_true,y_pred,average = 'micro'))
    print('加权平均F1得分:',metrics.f1_score(y_true,y_pred,average = 'weighted'))

#
if __name__ == '__main__':

    # 训练和预测
    train_pred, test_pred, y_true,model,data_test = run_kb()

    # 评估验证集
    bk_metrics(train_pred,y_true,type =' train metrics')
    # 评估测试集

    bk_metrics(test_pred,[np.argmax(x) for x in data_test[:, 1]],type =' test metrics')
    # 将模型保存
    model_path =r'use_bert\bertkeras_model.h5'
    model.save(model_path)


    # 模型加载
    from keras_bert import get_custom_objects
    from keras.models import load_model
    custom_objects = get_custom_objects()
    my_objects = {'acc_top2': acc_top2}
    custom_objects.update(my_objects)
    model = load_model(model_path, custom_objects=custom_objects)


    # 单独评估一个本来分类
    text = '这家餐厅的菜味道可以'
    DATA_text = []
    DATA_text.append((text, to_categorical(0, 2)))
    DATA_text = np.array(DATA_text)
    text= data_generator(DATA_text, shuffle=False)
    test_model_pred  = model.predict_generator(text.__iter__(), steps=len(text), verbose=1)
    print('预测结果',test_model_pred)
    print(np.argmax(test_model_pred))


    del model # 删除模型减少缓存
    gc.collect()  # 清理内存
    K.clear_session()  # clear_session就是清除一个session


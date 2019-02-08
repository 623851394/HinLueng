# coding:utf-8
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import random


neg_filepath = './data/neg_idsMatrix.npy'
pos_filepath = './data/pos_idsMatrix.npy'
wordVectors_path = './npy/wordVectors.npy'

data = []
label = []

# 训练使用得超参数
MAX_WORD_LENGTH = 250
BATCH_SIZE = 24
lstmUnits = 128
NUM_LAYERS = 2
iterations = 100000
LSTM_KEEP_PROB = 0.75
num_classes = 2

# 学习率
LR = 0.1
LR_DECAY = 0.99
LR_STEP = 1

# ids = np.load('./npy/idsMatrix.npy')
# loading pos and neg data from *.npy,generating label and splitting train_data and test_data
def GetData(neg_file, pos_file):
    global data, label
    neg_data = np.load(neg_file)
    pos_data = np.load(pos_file)
    neg_data = neg_data[:12100]
    pos_data = pos_data[:12100]

    data.extend(neg_data)
    data.extend(pos_data)
    pos_label = []
    neg_label = []
    for i in range(12100):
        neg_label.append([0, 1])
        pos_label.append([1, 0])
    label = neg_label + pos_label
    # print(np.shape(label))
    # data = np.load(file)

    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.05, random_state=random.randint(0, 100))
    return train_x, test_x, train_y, test_y

def train():

    train_x, test_x, train_y, test_y = GetData(neg_filepath, pos_filepath)

    Max_train_legth = len(train_x)
    Max_test_legth = len(test_x)
    wordVectors = np.load(wordVectors_path,)
    # define placeholder
    with tf.variable_scope('inputs'):
        x = tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE, MAX_WORD_LENGTH], name='input_x')
        y = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE,num_classes], name='output_y')

        global_step = tf.Variable(0, trainable=False)
        learing_rate = tf.train.exponential_decay(LR, global_step, LR_STEP, LR_DECAY, staircase=True)


        keep_prob = tf.placeholder(tf.float32)
    # 将词编号映射到词向量
        data = tf.nn.embedding_lookup(wordVectors, x)

    # define lstm
    # lstmCell = [tf.nn.rnn_cell.LSTMCell(lstmUnits) for _ in range(2)]
    with tf.name_scope('lstm'):
        lstmCell = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(lstmUnits),
                output_keep_prob=keep_prob)
            for _ in range(NUM_LAYERS)
        ]

        # lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=keep_prob)
        cells = tf.nn.rnn_cell.MultiRNNCell(lstmCell)

        # s = cells.zero_state(BATCH_SIZE, tf.float32)


        value, s = tf.nn.dynamic_rnn(cells, data, dtype=tf.float32)

    # define layer
    # weight = tf.Variable(tf.truncated_normal([lstmUnits, num_classes]))
    # bias = tf.Variable(tf.constant(0.1,shape=[num_classes]))
    # 交换维度，第一唯和第二唯交换
        value = tf.transpose(value,(1, 0, 2))
    # 获取最后一维
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
    # 连接层
    # y_prediction = tf.matmul(last, weight) + bias
    with tf.name_scope('prediction'):
        y_prediction = tf.layers.dense(last, 2, use_bias=True)

    trainable_variables = tf.trainable_variables()  # 返回所有需要训练的参数

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_prediction, labels=y))
        loss = tf.log(tf.clip_by_value(loss, 1e-8, tf.reduce_max(loss)))
    # 控制梯度大小，定义优化方法和训练步骤
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(loss, trainable_variables), 5
    )  # 用来解决梯度爆炸或者梯度弥散问题 有一个最大阈值 返回截取过的梯度张量和一个所有张量的全局范数。
    optimizer = tf.train.AdamOptimizer()
    trainer = optimizer.apply_gradients(zip(grads, trainable_variables))  # 执行对所需要优化的参数进行优化


    # trainer = tf.train.AdamOptimizer(learing_rate).minimize(loss)

    # define acc
    with tf.name_scope('acc'):
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_prediction, 1), tf.argmax(y, 1))))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.summary.scalar('accuracy', acc)
    tf.summary.scalar('loss', loss)
    merge_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter('C:/Users/HinLeung/log', sess.graph)     # write to file

    for i in range(iterations):
        start = (i * BATCH_SIZE) % (Max_train_legth-BATCH_SIZE)
        # end = min(start + BATCH_SIZE, Max_train_legth)
        end = start + BATCH_SIZE
        # nextBatch, nextBatchLabels = getTrainBatch()
        sess.run([trainer], feed_dict={x:train_x[start:end], y:train_y[start:end],keep_prob:0.75})
        # print("step is %s, loss is %s"%(i, losses))
        if (i+1) % 100 == 0:
            # print("开始预测")
            start = (i * BATCH_SIZE) % (Max_test_legth - BATCH_SIZE)
            end = start + BATCH_SIZE
            # 存储loss 和 准确率
            add = sess.run(merge_op, feed_dict={x: test_x[start:end], y: test_y[start:end], keep_prob: 1.0})
            # 打印准确率
            print("Accuracy for %s batch:" %(i+1),(sess.run(acc, {x: test_x[start:end], y: test_y[start:end],keep_prob:1.0})))
            writer.add_summary(add, i+1)


if __name__ == '__main__':
    train()

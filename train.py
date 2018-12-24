#!/user/bin/env python
#-*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#下载mnist数据集
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

def weight_variable(shape):
    # 这个函数产生的随机数与均值的差距不会超过两倍的标准差
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,dtype=tf.float32,name='weight')
def bias_variable(shape):
    #产生一个维度为shape的元素全为0.1的矩阵
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial,dtype=tf.float32,name='biases')
def conv2d(x,W):
    # strides第一个和第四个都是1，然后中间俩个代表x方向和y方向的步长,这个函数用来定义卷积神经网络
    # x是图片张量，w是卷积核
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
    # x是图片张量，ksize中间两个参数是maxpooling大小，stride中间两个参数是步长
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def variable_summaries(name, var):
    with tf.name_scope(name + '_summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(name + '_mean', mean)
        with tf.name_scope(name + '_stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
        tf.summary.scalar(name + '_stddev', stddev)
        tf.summary.scalar(name + '_max', tf.reduce_max(var))
        tf.summary.scalar(name + '_min', tf.reduce_min(var))
        tf.summary.histogram(name + '_histogram', var)


def compute_accuracy(v_xs,v_ys):
    global prediction#定义全局变量
    y_pre=sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})#
    # 将实际值和预测值进行比较，返回Bool数据类型
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    # 将上面的bool类型转为float，求得矩阵中所有元素的平均值
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    # 运行得到上面的平均值，这个值越大说明预测的越准确，因为都是0-1类型，所以平均值不超过1
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result

#输入是一个28*28的像素点的数据
with tf.name_scope('input'):
    xs=tf.placeholder(tf.float32,[None,784],name='x')
    ys=tf.placeholder(tf.float32,[None,10],name='y_')
keep_prob=tf.placeholder(tf.float32)

#卷积层1
#xs的维度暂时不管，用-1表示，28,28表示xs的数据，1表示该数据是一个黑白照片，如果是彩色的，则写成3
with tf.name_scope('image_reshape'):
    x_image=tf.reshape(xs,[-1,28,28,1])
    tf.summary.image('input', x_image, 10)
#抽取一个5*5像素，高度是32的点,每次抽出原图像的5*5的像素点，高度从1变成32
with tf.name_scope('conv_lay1'):
    W_conv1=weight_variable([5,5,1,32])
    variable_summaries('W1', W_conv1)
    b_conv1=bias_variable([32])
    variable_summaries('B1', b_conv1)
    #输出 28*28*32的图像
    h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    ##输出14*14*32的图像，因为这个函数的步长是2*2，图像缩小一半。
    h_pool1=max_pool_2x2(h_conv1)

#卷积层2
#随机生成一个5*5像素，高度是64的点,抽出原图像的5*5的像素点，高度从32变成64
with tf.name_scope('conv_lay2'):
    W_conv2=weight_variable([5,5,32,64])
    variable_summaries('W2', W_conv2)
    b_conv2=bias_variable([64])
    variable_summaries('B2', b_conv2)
    #输出14*14*64的图像
    h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    ##输出7*7*64的图像，因为这个函数的步长是2*2，图像缩小一半。
    h_pool2=max_pool_2x2(h_conv2)

#fully connected
with tf.name_scope('FullyConnected'):
    W_fc1=weight_variable([7*7*64,1024])
    variable_summaries('W_fc1', W_fc1)
    b_fc1=bias_variable([1024])
    variable_summaries('b_fc1', b_fc1)
    #将输出的h_pool2的三维数据变成一维数据，平铺下来，（-1）代表的是有多少个例子
    h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
    h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#输出层
with tf.name_scope('fc2'):
    W_fc2=weight_variable([1024,10])
    variable_summaries('W_fc2', W_fc2)
    b_fc2=bias_variable([10])
    variable_summaries('B_fc2', b_fc2)
    prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2,name='fc2_softmax')
log_dir=''
#开始训练数据
#相当于loss（代价函数）
with tf.name_scope('cross_entropy'):
    cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
#训练函数，降低cross_entropy（loss）,AdamOptimizer适用于大的神经网络
with tf.name_scope('train'):
    train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/savertrain', sess.graph)
    for i in range(200):
        # 每次从mnist数据集里面拿50个数据训练
        batch_xs,batch_ys=mnist.train.next_batch(50)
        sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
        if i%20==0:
            print(compute_accuracy(mnist.test.images[:200],mnist.test.labels[:200]))
    save_path = saver.save(sess, "my_model/save_net.ckpt")




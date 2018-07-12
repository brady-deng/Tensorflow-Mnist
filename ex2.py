import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
#采用tensorflow对MNIST数据集进行分类
#简单的将输入图像展成784位向量作为输入
#但是会忽略掉像素之间的纹理信息
#在Tensorboard中记录了w,b,accuracy,cross_entropy


data = input_data.read_data_sets("MNIST_data/", one_hot=True)   #import the MNIST database which is labeled by one_hot
with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32,[None,784])   #定义一个占位符，每张图片是28*28，也就是784个输入神经元？
    y_ = tf.placeholder("float",[None,10])
with tf.name_scope('Inference'):
    w = tf.Variable(tf.zeros([784,10]))     #定义一个可修改的权重张量，维度是784*10，784代表隐层神经元，10代表输出神经元
    b = tf.Variable(tf.zeros([10]))     #定义一个可修改的偏置张量，10代表的是输出神经元，
    y = tf.nn.softmax(tf.matmul(x,w)+b)     #预测y
    tf.summary.histogram('w',w)
    tf.summary.histogram('b',b)
with tf.name_scope('Loss'):
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))    #计算交叉熵
    tf.summary.histogram('loss',cross_entropy)
with tf.name_scope('Train'):
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)    #采用梯度下降的方法优化算法
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('Accuracy',accuracy)
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./ex2",sess.graph)
sess.run(init)
for i in range(1000):
    batch_x, batch_y = data.train.next_batch(100)
    sess.run(train_step,feed_dict={x: batch_x,y_: batch_y})
    if i % 50 == 0:
        result = sess.run(merged,feed_dict={x:batch_x,y_:batch_y})
        writer.add_summary(result,i)



writer.close()





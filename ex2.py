import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf

data = input_data.read_data_sets("MNIST_data/", one_hot=True)   #import the MNIST database which is labeled by one_hot
x = tf.placeholder(tf.float32,[None,784])   #定义一个占位符，每张图片是28*28，也就是784个输入神经元？
w = tf.Variable(tf.zeros([784,10]))     #定义一个可修改的权重张量，维度是784*10，784代表隐层神经元，10代表输出神经元
b = tf.Variable(tf.zeros([10]))     #定义一个可修改的偏置张量，10代表的是输出神经元，
y = tf.nn.softmax(tf.matmul(x,w)+b)     #预测y
y_ = tf.placeholder("float",[None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))    #计算交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)    #采用梯度下降的方法优化算法
init = tf.initialize_all_variables()    #初始化模型参数
sess = tf.Session()
sess.run(init)
for i in range(1000):   #循环训练1000次
    batch_x,batch_y = data.train.next_batch(100)    #每次训练100张图片
    sess.run(train_step,feed_dict={x:batch_x,y_:batch_y})
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print(sess.run(accuracy,feed_dict={x:data.test.images,y_:data.test.labels}))
    

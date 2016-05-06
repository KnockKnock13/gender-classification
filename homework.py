import tensorflow as tf
import read_image
import random
from tensorflow.examples.tutorials.mnist import input_data
#
#
train_data,train_lable,test_data,test_label = read_image.read()
sess = tf.InteractiveSession()
#debug
print(train_data.shape)
#
batch_size=100
patch1=5
patch2=5
image_size=100
num_channels=1

# Input placehoolders
with tf.name_scope('input'):
  x = tf.placeholder(tf.float32, [None, 100, 100], name='x_input')
  image_shaped_input = tf.reshape(x, [-1, 100, 100, 1])
  tf.image_summary('input', image_shaped_input, 10)
  y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')

x_=tf.reshape(x, [-1, 100, 100, 1])
print(tf.Print(x_,[x_]))

#
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.05)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def variable_summaries(var, name):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/'+name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.sqrt(var-mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
'''
#Convilution layer 0
with tf.name_scope('Conv_0'):
  with tf.name_scope('weights'):
    weights = weight_variable([patch1, patch1, 1, 6])#channels=1 depth=6
    variable_summaries(weights, 'Conv_0/weights')
  with tf.name_scope('biases'):
    biases = bias_variable([6])#depth = 6
    variable_summaries(biases, 'Conv_0/biases')
  conv_0 = tf.nn.conv2d(x_, weights, [1, 1, 1, 1], padding='VALID', name='conv_1')
  hidden_0 = tf.nn.max_pool(conv_0, [1, 3, 3, 1], [1, 3, 3, 1], padding='SAME', name='pool_0') + biases
tf.histogram_summary('Conv_0/activations', hidden_0)
'''
#Convilution layer 1
with tf.name_scope('Conv_1'):
  with tf.name_scope('weights'):
    weights = weight_variable([patch1, patch1, 1, 6])#channels=1 depth=6
    variable_summaries(weights, 'Conv_1/weights')
  with tf.name_scope('biases'):
    biases = bias_variable([6])#depth = 6
    variable_summaries(biases, 'Conv_1/biases')
  conv_1 = tf.nn.conv2d(x_, weights, [1, 1, 1, 1], padding='VALID', name='conv_1')
  temp = tf.nn.relu(conv_1 + biases)
  hidden_1 = tf.nn.max_pool(temp, [1, 4, 4, 1], [1, 4, 4, 1], padding='SAME', name='pool_1')
tf.histogram_summary('Conv_1/activations', hidden_1)
#test
print(tf.Print(hidden_1,[hidden_1]))#[100, 14, 14, 6]
#Convilution layer 2
with tf.name_scope('conv_2'):
  with tf.name_scope('weights'):
    weights_1 = weight_variable([patch1, patch1, 6, 16])
    variable_summaries(weights, 'Conv_2/weights')
  with tf.name_scope('biases'):
    biases_1 = bias_variable([16])
    variable_summaries(biases, 'Conv_2/biases')
  conv_21 = tf.nn.conv2d(hidden_1, weights_1, [1, 1, 1, 1], padding='VALID', name='conv_21') + biases_1
  temp = tf.nn.relu(conv_21)
  hidden_21 = tf.nn.max_pool(temp, [1, 4, 4, 1], [1, 4, 4, 1], padding='SAME', name='hidden_21')
#debug
print(tf.Print(hidden_21,[hidden_21]))
# Convilution layer 3
with tf.name_scope('conv_3'):
  with tf.name_scope('weights'):
    weights = weight_variable([patch2, patch2, 16, 300])
    variable_summaries(weights, 'Conv_3/weights')
  with tf.name_scope('biases'):
    biases = bias_variable([300])
    variable_summaries(biases, 'Conv_3/biases')
  conv_3 = tf.nn.relu(tf.nn.conv2d(hidden_21, weights, [1, 1, 1, 1], padding='VALID', name='conv_3') + biases)
  
 # hidden_3 = tf.nn.max_pool(conv_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='hidden_3')
#debug
print(tf.Print(conv_3,[conv_3]))
# Fully connected layer F6
with tf.name_scope('F_Layer'):  
  with tf.name_scope('weights'):
    weights = weight_variable([300,2])
    variable_summaries(weights, 'F6/weights')
  with tf.name_scope('biases'):
    biases = bias_variable([2])
    variable_summaries(biases, 'F6/biases')
  conv_3 = tf.reshape(conv_3, [-1, 1*1*300])
  F_Layer = tf.matmul(conv_3, weights) + biases
#softmax
y = tf.nn.softmax(F_Layer, name='softmax')
print('ok1')
# 
with tf.name_scope('cross_entropy'):
  diff = y_ * tf.log(y)
  with tf.name_scope('total'):
    cross_entropy = -tf.reduce_mean(diff)
  tf.scalar_summary('cross entropy', cross_entropy)
#
with tf.name_scope('train'):
  with tf.name_scope('global_step'):
    global_step = tf.Variable(0, trainable=False)
  with tf.name_scope('leaning_rate'):
    #learning_rate = tf.train.exponential_decay(0.1, global_step, 50, 0.95, staircase=True)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy,global_step = global_step)
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
print('ok2')
#
with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(F_Layer, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.scalar_summary('accuracy', accuracy)
#
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('./train', sess.graph)
test_writer = tf.train.SummaryWriter('./test')
tf.initialize_all_variables().run()
#
def feed_dict(train, num):
  if train: 
    xs = train_data[num:num+batch_size, :, :]
    ys = train_lable[num:num+batch_size, :]
   # print(xs.shape)
   # print(tf.Print(xs,[xs]))
  else:
    xs, ys = test_data, test_label
    print('test')
  return {x: xs, y_: ys}

for i in range(1000000):
  summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True, random.randint(1,760-batch_size)))#feed_dict(True, i)
  train_writer.add_summary(summary, i)
  if i % 10 ==0:
     summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False, 0))
     test_writer.add_summary(summary, i)
     print('Accuracy of step %d is: %s' % (i , acc))


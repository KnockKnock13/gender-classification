import tensorflow as tf
import random

import read_image

#read data
train_data, train_label, test_data, test_label = read_image.read(1)

#
sess = tf.InteractiveSession()


#macro
batch = 50
patch = 5
imageSize =100
times = 1000

init_wSigma = 0.1
init_b = 0.1

learning_rate=0.01
#feature map
conv1_in = 1
conv1_out = 6
conv2_out = 16
conv3_out = 300
#Input 
with tf.name_scope('input'):
  x = tf.placeholder(tf.float32, [None, imageSize, imageSize], name = 'x_input')
  image_shaped_input =  tf.reshape(x, [-1, imageSize, imageSize, 1])
  tf.image_summary('input', image_shaped_input, 10)
  y_ = tf.placeholder(tf.float32, [None, 2], name = 'y_input')


#reshap input image
x_ = tf.reshape(x, [-1, imageSize, imageSize, 1])


#function
def weight_variable(shape):
  initial = tf. truncated_normal(shape, stddev = init_wSigma)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(init_b, shape = shape)
  return tf.Variable(initial)

def variable_summary(var, name):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.sqrt(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
#convolution layer 1
with tf.name_scope('convlution1'):
  with tf.name_scope('weights'):
    weights = weight_variable([patch, patch, 1, conv1_out])
    variable_summary(weights, 'convlution1/weights')
  with tf.name_scope('bias'):
    bias = bias_variable([conv1_out])
    variable_summary(bias, 'convlution1/bias')
  conv1 = tf.nn.conv2d(x_, weights, [1, 1, 1, 1], padding = 'VALID', name = 'convlution1')
  relu = tf.nn.relu(conv1 + bias, name = 'relu1')
  hidden1 = tf.nn.max_pool(relu, [1, 4, 4, 1], [1, 4, 4, 1], padding = 'SAME', name='pool1')
  
#convolution layer 2
with tf.name_scope('convolution2'):
  with tf.name_scope('weights'):
    weights = weight_variable([patch, patch, conv1_out, conv2_out])
    variable_summary(weights, 'convolution2/weights')
  with tf.name_scope('bias'):
    bias = bias_variable([conv2_out])
    variable_summary(bias, 'convolution2/bias')
  conv2 = tf.nn.conv2d(hidden1, weights, [1, 1, 1, 1], padding = 'VALID', name = 'convlution2')
  relu = tf.nn.relu(conv2 + bias, name = 'relu2')
  hidden2 = tf.nn.max_pool(relu, [1, 4, 4, 1], [1, 4, 4, 1], padding = 'SAME', name = 'pool2')

#convolution layer 3
with tf.name_scope('convolution3'):
  with tf.name_scope('weights'):
    weights = weight_variable([patch, patch, conv2_out, conv3_out])
    variable_summary(weights, 'convolution3/weights')
  with tf.name_scope('bias'):
    bias = bias_variable([conv3_out])
    variable_summary(bias, 'convolution3/bias')
  conv3 = tf.nn.conv2d(hidden2, weights, [1, 1, 1, 1], padding = 'VALID', name = 'convolution3')
  relu = tf.nn.relu(conv3 + bias , name = 'relu3')
  #reshape to a vector
  relu = tf.reshape(relu, [-1, 1*1*conv3_out])
#dropout
#keep_prob = tf.placeholder(tf.float32)
#relu = tf.nn.dropout(relu, keep_prob)
#fully connected layer
with tf.name_scope('fully'):
  with tf.name_scope('weights'):
    weights = weight_variable([conv3_out,2])
    variable_summary(weights, 'fully/weights')
  with tf.name_scope('bias'):
    bias = bias_variable([2])
    variable_summary(bias, 'fully/bias')
  f_out = tf.matmul(relu, weights) + bias
#

#softmax
y = tf.nn.softmax(f_out, name = 'softmax')
#

with tf.name_scope('cross_entropy'):
  diff = y_ * tf.log(y)
  with tf.name_scope('total'):
    cross_entropy = -tf.reduce_mean(diff)
  tf.scalar_summary('cross_entropy', cross_entropy)

#training
with tf.name_scope('train'):
  with tf.name_scope('global_step'):
    global_step = tf.Variable(0, trainable = False)
  with tf.name_scope('leaning_rate'):
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
#
with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(f_out, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)
#so far,the graph is complete
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('./train', sess.graph)
test_writer = tf.train.SummaryWriter('./test')
tf.initialize_all_variables().run()
#save the varible
saver = tf.train.Saver()

#
def feed_dict(train, num):
  if train:
    xs = train_data[num:num+batch, :, :]
    ys = train_label[num:num+batch, :]
  else: 
    xs, ys = test_data, test_label
  return {x: xs, y_: ys}

for i in range(times):
  summary, _ = sess.run([merged, train_step], feed_dict = feed_dict(True, random.randint(1, 760-batch)))
  train_writer.add_summary(summary, i)
  if i % 10 == 0:
    summary, acc =sess.run([merged, accuracy], feed_dict = feed_dict(False, 0))
    test_writer.add_summary(summary, i)
    print('The accuracy at step %d is %s' % (i, acc))
save_path = saver.save(sess, "./model.ckpt")
print("Model saved in file:%s" % save_path)

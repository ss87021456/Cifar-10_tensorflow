import tensorflow as tf
import data_helper

pp = pprint.PrettyPrinter(indent=4)

# Load CIFAR-10 data
data_sets = data_helper.load_data()

# define functions
def weight_variable(shape, stddev=0.05):
        initial = tf.random_normal(shape, stddev=stddev, dtype=tf.float32)
        return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')



#define placeholder 32*32*3 = 3072
x = tf.placeholder(tf.float32, [None, 3072])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_temp = tf.reshape(x, [-1,3,32,32])
x_image = tf.transpose(x_temp, (0,2,3,1))

## conv1 layer 
#Conv1 filter size = 5x5, # of filter =192, pad = 'same', stride = 1 Act.=ReLU
W_conv1 = weight_variable([5, 5, 3, 192], stddev=0.01)
b_conv1 = tf.Variable(tf.random_normal([192], stddev=0.01, dtype=tf.float32))
output = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#mlp 1 filter size = 1x1, # of filter =160, pad = 'same', stride = 1 Act.=ReLU
W_MLP11 = weight_variable([1, 1, 192, 160])
b_MLP11 = bias_variable([160])
output = tf.nn.relu(conv2d(output, W_MLP11) + b_MLP11)
#mlp 2 filter size = 1x1, # of filter =96, pad = 'same', stride = 1 Act.=ReLU
W_MLP12 = weight_variable([1, 1, 160, 96])
b_MLP12 = bias_variable([96])
output = tf.nn.relu(conv2d(output, W_MLP12) + b_MLP12)
#Pool 1 3x3 max pooling, stride = 2, pad = ('same')
output = max_pool_3x3(output)		#size: 16x16x96
#dropout pro = 0.5
output = tf.nn.dropout(output, keep_prob)


## conv2 layer
#Conv2 filter size = 5x5, # of filter =192, pad = 'same', stride = 1 Act.=ReLU
W_conv2 = weight_variable([5, 5, 96, 192])
b_conv2 = tf.Variable(tf.random_normal([192], stddev=0.01, dtype=tf.float32))
output = tf.nn.relu(conv2d(output, W_conv2) + b_conv2)
#mlp 2-1 filter size = 1x1, # of filter =192, pad = 'same', stride = 1 Act.=ReLU
W_MLP21 = weight_variable([1, 1, 192, 192])
b_MLP21 = bias_variable([192])
output = tf.nn.relu(conv2d(output, W_MLP21) + b_MLP21)
#mlp 2-2 filter size = 1x1, # of filter =192, pad = 'same', stride = 1 Act.=ReLU
W_MLP22 = weight_variable([1, 1, 192, 192])
b_MLP22 = bias_variable([192])
output = tf.nn.relu(conv2d(output, W_MLP22) + b_MLP22)
#Pool 2 3x3 max (avg) pooling, stride = 2, pad =  ('same')
output = max_pool_3x3(output)	#size: 8x8x192
#dropout pro = 0.5
output = tf.nn.dropout(output, keep_prob)


#Conv3 filter size = 3x3, # of filter =192, pad = 'same', stride = 1 Act.=ReLU
W_conv3 = weight_variable([3, 3, 192, 192])
b_conv3 = tf.Variable(tf.random_normal([192], stddev=0.01, dtype=tf.float32))
output = tf.nn.relu(conv2d(output, W_conv3) + b_conv3)
#mlp 3-1 filter size = 1x1, # of filter =192, pad = 'same', stride = 1 Act.=ReLU
W_MLP31 = weight_variable([1, 1, 192, 192])
b_MLP31 = bias_variable([192])
output = tf.nn.relu(conv2d(output, W_MLP31) + b_MLP31)
#mlp 3-2 filter size = 1x1, # of filter =10, pad = 'same', stride = 1 Act.=ReLU
W_MLP32 = weight_variable([1, 1, 192, 10])
b_MLP32 = bias_variable([10])
output = tf.nn.relu(conv2d(output, W_MLP32) + b_MLP32)
#Global Pool 8x8 average pooling, stride =1, pad =  ('same')
output = tf.nn.avg_pool(output, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')

output = tf.reshape(output, [-1, 1 * 1 * 10])


#the loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))


# Weight decay
Weight_decay = 0.0001
l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

train_step = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(cross_entropy + l2 * Weight_decay, global_step=global_step)

#prediction
correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(y_,1) )
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
#start session
zipped_data = zip(data_sets['images_train'], data_sets['labels_train'])
batches = data_helper.gen_batch(list(zipped_data), 128, 64124)

zipped_data_test = zip(data_sets['images_test'], data_sets['labels_test'])
batches_test = data_helper.gen_batch(list(zipped_data_test), 128, 64124)


#pp.pprint(zip(*next(batches_test))[1])
epoch_num = 0
max_train_accuracy = 0
max_test_accuracy = 0
min_loss = 99999
for i in range(64124): #391 interation_num * 164 epoch_num
	
 	batch = next(batches)
 	images_batch, labels_batch = zip(*batch)
 	batch_test = next(batches_test)
 	images_batch_test, labels_batch_test = zip(*batch_test)

 	sess.run(train_step, feed_dict = {x:images_batch, y_:labels_batch, keep_prob: 0.5})
 	if i % 100 == 0:
 		
 		_loss, _accuracy = sess.run([cross_entropy, accuracy], feed_dict={
 			x: images_batch, y_: labels_batch , keep_prob: 1.0})
 		_accuracy_test = sess.run(accuracy,feed_dict={
 			x: images_batch_test, y_: labels_batch_test , keep_prob: 1.0})
 		min_loss = min(_loss,min_loss)
 		max_train_accuracy = max(_accuracy,max_train_accuracy)
 		max_test_accuracy = max(_accuracy,max_test_accuracy)
 		print ("epoch num: [%d/164] training_loss:%f" % ((i/391),_loss))
 		print ("step %d, Train accuracy %g"%(i, _accuracy))
 		print ("step %d, Test accuracy %g"%(i, _accuracy_test))
 	
print ("max_train_accuracy:%f ,max_test_accuracy:%f, min_loss:%f" ,(max_train_accuracy,max_test_accuracy,min_loss))



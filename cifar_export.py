from __future__ import division, print_function, absolute_import
# library for optmising inference
from tensorflow.python.tools import optimize_for_inference_lib
import tensorflow as tf
# Higher level API tflearn
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import numpy as np

# Data loading and preprocessing
#helper functions to download the CIFAR 10 data and load them dynamically
from tflearn.datasets import cifar10
(X, Y), (X_test, Y_test) = cifar10.load_data()
X, Y = shuffle(X, Y)
Y = to_categorical(Y,10)
Y_test = to_categorical(Y_test,10)


#input image
x=tf.placeholder(tf.float32,shape=[None, 32, 32, 3] , name="ipnode")
#input class
y_=tf.placeholder(tf.float32,shape=[None, 10] , name='input_class')


# AlexNet architecture
input_layer=x
network = conv_2d(input_layer, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = fully_connected(network, 10, activation='linear')
y_predicted=tf.nn.softmax(network , name="opnode")

#loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_predicted+np.exp(-10)), reduction_indices=[1]))
#optimiser -
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#calculating accuracy of our model
correct_prediction = tf.equal(tf.argmax(y_predicted,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#TensorFlow session
sess = tf.Session()
#initialising variables
init = tf.global_variables_initializer()
sess.run(init)
#tensorboard for better visualisation
writer =tf.summary.FileWriter('tensorboard/', sess.graph)
epoch=50 # run for more iterations according your hardware's power
#change batch size according to your hardware's power. For GPU's use batch size in powers of 2 like 2,4,8,16...
batch_size=32
no_itr_per_epoch=len(X)//batch_size
n_test=len(X_test) #number of test samples


# Commencing training process
for iteration in range(epoch):
    print("Iteration no: {} ".format(iteration))

    previous_batch=0
    # Do our mini batches:
    for i in range(no_itr_per_epoch):
        current_batch=previous_batch+batch_size
        x_input=X[previous_batch:current_batch]
        x_images=np.reshape(x_input,[batch_size,32,32,3])

        y_input=Y[previous_batch:current_batch]
        y_label=np.reshape(y_input,[batch_size,10])
        previous_batch=previous_batch+batch_size

        _,loss=sess.run([train_step, cross_entropy], feed_dict={x: x_images,y_: y_label})
        #if i % 100==0 :
            #print ("Training loss : {}" .format(loss))



    x_test_images=np.reshape(X_test[0:n_test],[n_test,32,32,3])
    y_test_labels=np.reshape(Y_test[0:n_test],[n_test,10])
    Accuracy_test=sess.run(accuracy,
                           feed_dict={
                        x: x_test_images ,
                        y_: y_test_labels
                      })
    # Accuracy of the test set
    Accuracy_test=round(Accuracy_test*100,2)
    print("Accuracy ::  Test_set {} %  " .format(Accuracy_test))





saver = tf.train.Saver()
model_directory='model_files/'
#saving the graph
tf.train.write_graph(sess.graph_def, model_directory, 'savegraph.pbtxt')

saver.save(sess, 'model_files/model.ckpt')
# Freeze the graph
MODEL_NAME = 'CIFAR'
input_graph_path = 'model_files/savegraph.pbtxt'
checkpoint_path = 'model_files/model.ckpt'
input_saver_def_path = ""
input_binary = False
output_node_names = "opnode"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'model_files/frozen_model_'+MODEL_NAME+'.pb'
output_optimized_graph_name = 'model_files/optimized_inference_model_'+MODEL_NAME+'.pb'
clear_devices = True
#Freezing the graph and generating protobuf files
freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")
#Optimising model for inference only purpose
output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        sess.graph_def,
        ["ipnode"], # an array of the input node(s)
        ["opnode"], # an array of output nodes
        tf.float32.as_datatype_enum)

with tf.gfile.GFile(output_optimized_graph_name, "wb") as f:
            f.write(output_graph_def.SerializeToString())
sess.close()

import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from vgg16 import VGG16
from datagenerator import ImageDataGenerator

"""
Configuration settings
"""

# data params
data_size='3_new'

if data_size == '256': rgb_mean = 127.5093
elif data_size == '128': rgb_mean = 127.0391
elif data_size == '71': rgb_mean = 127.0901
elif data_size == '71_new': rgb_mean = 124.9396
elif data_size == '1_new': rgb_mean = 32.1265
elif data_size == '2_new': rgb_mean = 30.2056
elif data_size == '3_new': rgb_mean = 11.9783
elif data_size == '4_new': rgb_mean = 26.5960


# Path to the textfiles for the trainings and validation set
train_file = 'datatxt/train_'+data_size+'.txt'
val_file = 'datatxt/val_'+data_size+'.txt'

# Learning params
learning_rate = 0.0001
num_epochs = 100
batch_size = 128

# Network params
dropout_rate = 0.5
num_classes = 2
train_layers = ['fc8', 'fc7','fc6']



# How often we want to write the tf.summary data to disk
display_step = 5
checkpoint_step = 5

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = 'result/data_'+data_size+'/tensorboard'
checkpoint_path = 'result/data_'+data_size+'/checkpoint'

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)
if not os.path.isdir(filewriter_path): os.mkdir(filewriter_path)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = VGG16(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))  

# Train op
with tf.name_scope("train"):
  # Get gradients of all trainable variables
  gradients = tf.gradients(loss, var_list)
  gradients = list(zip(gradients, var_list))
  
  # Create optimizer and apply gradient descent to the trainable variables
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary  
for gradient, var in gradients:
  tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary  
for var in var_list:
  tf.summary.histogram(var.name, var)
  
# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)



# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
  correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  
# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
# writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver(max_to_keep=50)

# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(class_list=train_file, 
                                  horizontal_flip = True, shuffle = True, 
                                  mean = np.array([rgb_mean, rgb_mean, rgb_mean]), 
                                  nb_classes=num_classes)
val_generator = ImageDataGenerator(class_list=val_file, shuffle = False, 
                                  mean = np.array([rgb_mean, rgb_mean, rgb_mean]), 
                                  nb_classes=num_classes) 

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)

val_accuracy=[]
train_loss=[]
train_accuracy=[]
val_loss=[]
# Start Tensorflow session
with tf.Session() as sess:
 
  # Initialize all variables
  sess.run(tf.global_variables_initializer())
  
  # Add the model graph to TensorBoard
  writer = tf.summary.FileWriter(filewriter_path)
  writer.add_graph(sess.graph)
  
  # Load the pretrained weights into the non-trainable layer
  model.load_initial_weights(sess)
  
  print("{} Start training...".format(datetime.now()))
  starttime=datetime.now()
  print("{} Open Tensorboard at --logdir {}".format(datetime.now(), 
                                                    filewriter_path))
  
  # Loop over number of epochs

  for epoch in range(num_epochs):
    
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        
        step = 1
        
        while step < train_batches_per_epoch:
            
            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size)
            
            # And run the training op
            sess.run(train_op, feed_dict={x: batch_xs, 
                                          y: batch_ys, 
                                          keep_prob: dropout_rate})
            
            # Generate summary with the current batch of data and write to file
            # if step%display_step == 0:
                
            s = sess.run(merged_summary, feed_dict={x: batch_xs, 
                                                    y: batch_ys, 
                                                    keep_prob: 1.})
            writer.add_summary(s, epoch*train_batches_per_epoch + step)
                
            step += 1
        train_cost, train_acc = sess.run([loss, accuracy], feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_rate}) 
        print("{} train loss = {:.2f}, train accuracy = {:.2f}".format(datetime.now(), train_cost, train_acc*100.00))
        train_loss=np.r_[train_loss,train_cost]
        np.savetxt('result/data_'+data_size+'/train_loss.txt',train_loss)
        train_accuracy=np.r_[train_accuracy,train_acc]
        np.savetxt('result/data_'+data_size+'/train_accuracy.txt',train_accuracy)
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(batch_size)
            val_cost, val_acc = sess.run([loss,accuracy], feed_dict={x: batch_tx, 
                                                y: batch_ty, 
                                                keep_prob: 1.})
            test_acc += val_acc
            test_count += 1
        test_acc /= test_count

        print("{} val loss = {:.2f}, val accuracy = {:.2f}".format(datetime.now(), val_cost, val_acc*100.00))

        val_accuracy=np.r_[val_accuracy,val_acc]
        val_loss=np.r_[val_loss,val_cost]
        np.savetxt('result/data_'+data_size+'/val_accuracy.txt',val_accuracy)
        np.savetxt('result/data_'+data_size+'/val_loss.txt',val_loss)
        
        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()
        
        # print("{} Saving checkpoint of model...".format(datetime.now()))  

        #save checkpoint of the model
        if (epoch+1)%checkpoint_step == 0:
            checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
            save_path = saver.save(sess, checkpoint_name)  
        
        # print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))


endtime=datetime.now()
print('Train time: ',(endtime-starttime).seconds)


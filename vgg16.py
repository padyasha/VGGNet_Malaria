import tensorflow as tf
import numpy as np

class VGG16(object):
  
  def __init__(self, x, keep_prob, num_classes, skip_layer, 
               weights_path = 'DEFAULT'):
    
    # Parse input arguments into class variables
    self.X = x
    self.NUM_CLASSES = num_classes
    self.KEEP_PROB = keep_prob
    self.SKIP_LAYER = skip_layer
    
    if weights_path == 'DEFAULT':      
      self.WEIGHTS_PATH = 'vgg16.npy'
    else:
      self.WEIGHTS_PATH = weights_path
    
    # Call the create function to build the computational graph of AlexNet
    self.create()
    
  def create(self):
    
    # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
    self.conv1_1 = conv(self.X, 3, 3, 64, 1, 1, padding = 'SAME', name = 'conv1_1')
    self.conv1_2 = conv(self.conv1_1, 3, 3, 64, 1, 1, padding = 'SAME', name = 'conv1_2')
    self.pool1 = max_pool(self.conv1_2, 'pool1')
    
    # 2nd Layer:
    self.conv2_1 = conv(self.pool1, 3, 3, 128, 1, 1, padding = 'SAME', name = 'conv2_1')
    self.conv2_2 = conv(self.conv2_1, 3, 3, 128, 1, 1, padding = 'SAME', name = 'conv2_2')
    self.pool2 = max_pool(self.conv2_2, 'pool2')
    
    # 3rd Layer:
    self.conv3_1 = conv(self.pool2, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv3_1')
    self.conv3_2 = conv(self.conv3_1, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv3_2')
    self.conv3_3 = conv(self.conv3_2, 3, 3, 256, 1, 1, padding = 'SAME', name = 'conv3_3')
    self.pool3 = max_pool(self.conv3_3, 'pool3')
    
    # 4th Layer:
    self.conv4_1 = conv(self.pool3, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv4_1')
    self.conv4_2 = conv(self.conv4_1, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv4_2')
    self.conv4_3 = conv(self.conv4_2, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv4_3')
    self.pool4 = max_pool(self.conv4_3, 'pool4')
    
    # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    self.conv5_1 = conv(self.pool4, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv5_1')
    self.conv5_2 = conv(self.conv5_1, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv5_2')
    self.conv5_3 = conv(self.conv5_2, 3, 3, 512, 1, 1, padding = 'SAME', name = 'conv5_3')
    self.pool5 = max_pool(self.conv5_3, 'pool5')
    
    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    flattened = tf.reshape(self.pool5, [-1, 7*7*512])
    self.fc6 = fc(flattened, 7*7*512, 4096, name='fc6')
    dropout6 = dropout(self.fc6, self.KEEP_PROB)
    
    # 7th Layer: FC (w ReLu) -> Dropout
    self.fc7 = fc(dropout6, 4096, 4096, name = 'fc7')
    dropout7 = dropout(self.fc7, self.KEEP_PROB)
    
    # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
    self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu = False, name='fc8')

    
  def load_initial_weights(self, session):

    # Load the weights into memory
    weights_dict = np.load(self.WEIGHTS_PATH, allow_pickle=True, encoding='latin1').item()
    
    # Loop over all layer names stored in the weights dict
    for op_name in weights_dict:
        
      # Check if the layer is one of the layers that should be reinitialized
      if op_name not in self.SKIP_LAYER:
        
        with tf.variable_scope(op_name, reuse = True):
            
          # Loop over list of weights/biases and assign them to their corresponding tf variable
          for data in weights_dict[op_name]:
            
            # Biases
            if len(data.shape) == 1:
              
              var = tf.get_variable('biases', trainable = False)
              session.run(var.assign(data))
              
            # Weights
            else:
              
              var = tf.get_variable('weights', trainable = False)
              session.run(var.assign(data))
            
  

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):

  # Get number of input channels
  input_channels = int(x.get_shape()[-1])
  
  # Create lambda function for the convolution
  convolve = lambda i, k: tf.nn.conv2d(i, k, 
                                       strides = [1, stride_y, stride_x, 1],
                                       padding = padding)
  
  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases of the conv layer
    weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels/groups, num_filters])
    biases = tf.get_variable('biases', shape = [num_filters])  
    
    
    conv = convolve(x, weights)
      
    # Add biases 
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    
    # Apply relu function
    relu = tf.nn.relu(bias, name = scope.name)
        
    return relu
  
def fc(x, num_in, num_out, name, relu = True):
  with tf.variable_scope(name) as scope:
    
    # Create tf variables for the weights and biases
    weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
    biases = tf.get_variable('biases', [num_out], trainable=True)
    
    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    
    if relu == True:
      # Apply ReLu non linearity
      relu = tf.nn.relu(act)      
      return relu
    else:
      return act
    
def max_pool(bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def dropout(x, keep_prob):
  return tf.nn.dropout(x, keep_prob)

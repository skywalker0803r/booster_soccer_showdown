Model Guidelines
How to structure your model classes for compatibility with the SAI platform

To support seamless submissions through the SAI client, models should follow a few simple structural conventions. These don’t affect how you train your model, but they ensure the platform can properly save, load, and run it. While you're free to design the internals however you'd like, following the structure below is highly recommended.

Through the SAI CLI, you can create a template in a few simple steps for any of the frameworks defined below!

If you do not follow the conventions below, you’ll need to export and submit your model files manually.

  PyTorch
Compatibility Notes for PyTorch model classes:

Class must inherit from nn.Module

Must have a forward method, which performs inference given the model input


Copy
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Create the network however you like
        # You can use nn.ModuleList, nn.Sequential, etc

    def forward(self, x):
        # Forward pass through the network
        # Return the output of the forward pass
        return output
  TensorFlow 1.x
Compatibility Notes for TensorFlow 1.x model classes:

Tensorflow 1.x is currently unstable with continuous action space environments, please consider using another framework or exporting the model as an onnx file.

When running TF 1.x models, eager execution must be disabled

Must have states, policy, and sess attributes in the class

states is the placeholder for the model inputs (must be named "states")

policy is the output of the model (must be named "policy")

sess is the TensorFlow session used to run operations on the graph


Copy
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

class Model:
    def __init__(self, *args, **kwargs):
        self.states = tf.placeholder(tf.float32, [None, self.n_features], name="states")
        
        self.policy = tf.nn.softmax(self.logits, name="policy")
        
        self.sess = tf.Session()
 TensorFlow 2.x
Compatibility Notes for TensorFlow 2.x model classes:

Class must inherit from tf.Module 

Requires a get_concrete_function for submission, with the tf.TensorSpec defining the input dimensionality

Must have a __call__ method, which performs inference given the model input


Copy
import tensorflow as tf

class Model(tf.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Create the network by defining the variables with tf.Variable

    def get_concrete_function(self):
        return self.__call__.get_concrete_function(
            tf.TensorSpec([None, self.n_features], tf.float32)
        )            

    @tf.function
    def __call__(self, x):
        # Forward pass through the network
        # Return the output of the forward pass
        return output
 Keras
Compatibility Notes for Keras model classes:

Class must inherit from keras.Module

Must have a call method, which performs inference given the model input

Must have a get_hidden_layers method, which returns an array of keras.layers.<LayerType> hidden layers

Must have a get_output_layer method, which returns the final keras.layers.<LayerType> output layer


Copy
import keras

class Model(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Create the network however you like
        # For example, you can create layers with keras.layers.Dense     

    def call(self, inputs):
        # Forward pass through the network
        # Return the output of the forward pass
        return output
        
    def get_hidden_layers(self):
        # Returns an array of the hidden layers
        return hidden_layers
        
    def get_output_layer(self):
        # Returns the output layer
        return output_layer        
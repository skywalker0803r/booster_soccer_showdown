Model Saving
If you followed the model guidelines to create your model, then saving a model is as easy as:


Copy
# To save in the framework's native format
sai.save_model("Model Name", model)

# To save the model as ONNX
sai.save_model("Model Name", model, use_onnx=True)
Saving the model as an onnx file with "use_onnx=True" is an experimental feature that has not been rigorously tested. If you encounter any issues with the feature please reach out for support.

However, if you used another convention for defining model classes, then you might find the exporting instructions below helpful.

  PyTorch
We no longer support the use of torch.save. If you receive the following error:


Copy
Validation failed: PytorchStreamReader failed locating file constants.pkl: file not found
Please attempt to save the model as a Torchscript model by following the instructions below or using the sai.save_model() function. The experimental option of saving the model as onnx is also available.

To save the model as a TorchScript model:


Copy
import torch
import numpy as np

# state_space is defined as `env.observation_space`

def convert_model(model):
    obs_tensor = torch.from_numpy(
        np.zeros((1, *state_space.shape), dtype=state_space.dtype)
    )

    model = self._torch.jit.trace(
        model,
        torch.randn(*obs_tensor.shape, dtype=obs_tensor.dtype),
    )
    
if not isinstance(model, self._torch.jit.ScriptModule):
    model = convert_model(model)

torch.jit.save(model, model_path)
  TensorFlow 1.x
To save a TensorFlow 1.x model:


Copy
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.graph_util import convert_variables_to_constants

sess = model.sess
output_node_names = [model.policy.name.split(":")[0]]
frozen_graph_def = convert_variables_to_constants(
    sess, 
    sess.graph.as_graph_def(), 
    output_node_names=output_node_names
)
tf.io.write_graph(frozen_graph_def, logdir=".", name=model_path, as_text=False)
 TensorFlow 2.x
To save a TensorFlow 2.x model which inherits from tf.Module:


Copy
import tensorflow as tf

tf.saved_model.save(model, model_path)
 Keras
To save a sequential keras model:


Copy
import keras

def convert_model(model):
    # Create a sequential model from the trained model's layers
    # Note: Your model's attributes might be named differently
    inference_model = keras.Sequential(
        [
            keras.layers.InputLayer(shape=(model.n_features,)),
            *model.hidden_layers,
            model.output_layer,
        ]
    )

    # Copy weights from trained model to the inference model
    for src_layer, dst_layer in zip(model.layers, inference_model.layers):
        dst_layer.set_weights(src_layer.get_weights())

    return inference_model
    
if conversion_required:
    model = convert_model(model)
    
model.save(model_path)
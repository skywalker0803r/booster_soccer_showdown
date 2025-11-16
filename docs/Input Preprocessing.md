Input Preprocessing
Customize how your model sees the environment state before inference

By default, if you submit a model without specifying a Preprocessor, the SAI platform will use the default state returned from the environment as the input for your model. However, you can modify the state with a custom preprocessor class, allowing more bespoke models (e.g., RNNs or goal-conditioned models).

Preprocessor classes have access to numpy as np and the environment as env. No other imports or external variables are allowed.

Example: Goal-Conditioned State
Below is an example of a custom preprocessor which fetches a goal and appends it to the observation:


Copy
class Preprocessor():
    def __init__(self):
        # Initialize with appropriate attributes
        pass

    def get_goal(self, info):
        # Method to get the current goal
        pass

    def modify_state(self, obs, info):
        # Append the goal to the observation
        return np.hstack((
            np.expand_dims(obs, axis=0), 
            np.expand_dims(self.get_goal(info), axis=0)
        ))
Submitting with a Preprocessor
To use a custom preprocessor with your model submission, you can add it with the preprocessor_class keyword argument.


Copy
sai.submit(
    "my-model",
    model,
    preprocessor_class=Preprocessor
)
Without using a keyword argument, then the preprocessor_class must be the fourth argument in the function.


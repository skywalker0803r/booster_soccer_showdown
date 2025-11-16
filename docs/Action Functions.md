Action Functions
Customize how your agent selects actions at each time step

By default, if you submit a model without specifying an action function, the SAI platform will use the following:

Discrete Actions: Assumes the output is softmax and then selects using argmax.

Continuous Actions: Assumes the output is tanh and then rescales it based on the action space definition.

However, you can customize this behavior by creating and submitting your own action function, allowing more advanced or probabilistic strategies (e.g., sampling from a policy distribution).

Action functions have access to numpy as np and the environment as env. No other imports or external variables are allowed.

Example: Probabilistic Sampling
Below is an example of a custom action function that samples from a policy's softmax output:


Copy
def action_function(policy):
    return np.array(
        [np.random.choice(policy.shape[1], p=policy[i]) for i in range(policy.shape[0])]
    )
Submitting with an Action Function
To use a custom action function with your model submission, you can add it as the third argument:


Copy
sai.submit(
    "my-model",
    model,
    action_function
)
SAI Client
Documentation for the SAI Python client, including its features and usage examples.

The SAIClient class is the primary interface for interacting with the SAI (Reinforcement Learning Competition Platform) API. It provides methods for managing competitions, environments, scenes, benchmarks, and model submissions.

Initialization

Copy
from arenax_sai import SAIClient

sai = SAIClient(
    env_id: Optional[str] = None
    api_key: Optional[str] = None,
    *,
    comp_id: Optional[str] = None, 
    scene_id: Optional[str] = None,
    api_base: Optional[str] = None,
    is_server: bool = False
)
Parameters
Parameter
Type
Description
Default
env_id

str

ID of environment to load

None

api_key

str

API key for authentication

None

comp_id

str

ID of competition to load

None

scene_id

str

ID of scene to load

None

api_base

str

Custom API endpoint

None

is_server

bool

Whether running in server mode

False

If api_key is not provided, the client will attempt to retrieve it from the SAI_API_KEY environment variable or a stored location.

Methods
make_env
Creates and returns a Gymnasium environment for the loaded competition.


Copy
env = sai.make_env(render_mode = "human", **kwargs)
Parameter
Type
Description
Default
render_mode

Literal["human", "rgb_array"]

The render mode for the environment

"human"

**kwargs

Any

Set kwargs for the environment, this may override competition settings

None

Returns: A Gymnasium environment instance

Example:


Copy
env = sai.make_env(render_mode="rgb_array")
watch
Watch a model play through an environment. If no model is provided, it will use random action sampling.


Copy
sai.watch(
    model: Optional[ModelType] = None,
    action_function: Optional[str | Callable] = None,
    preprocessor_class: Optional[str | type] = None,
    *,
    model_type: Optional[ModelLibraryType] = None,
    algorithm: Optional[str] = None,
    num_runs: int = 1,
)
Parameter
Type
Description
Default
model

ModelType

Model to watch

None

action_function

Callable | str

Pass in the action function to use with the model

None

preprocessor_class

type | str

A class that can be used to pre-process inputs into the model.

None

model_type

ModelLibraryType

Framework used, should be auto detected from model

None

algorithm

str

Name of algorithm used for Stable Baslines 3 model

None

num_runs

int

Number of episodes to watch, will run sequentially

1

benchmark
Runs a benchmark for the specified model on the loaded competition's environment.


Copy
results = sai.benchmark(
    model: Optional[ModelType] = None,
    action_function: Optional[str | Callable] = None,
    preprocessor_class: Optional[str | type] = None,
    *,
    model_type: Optional[ModelLibraryType] = None,
    algorithm: Optional[str] = None,
    num_envs: Optional[int] = None,
    record: bool = False,
    video_dir: Optional[str] = "results",
    show_progress: bool = True,
    throw_errors: bool = True,
    timeout: int = 600,
    use_custom_eval: bool = False,
)
Parameter
Type
Description
Default
model

ModelType

Model to evaluate

None

action_function

Callable | str

Pass in the action function to use with the model

None

preprocessor_class

type | str

A class that can be used to pre-process inputs into the model

None

model_type

ModelLibraryType

Framework used, should be auto detected from model

None

algorithm

str

Name of algorithm used for Stable Baslines 3 model

None

num_envs

int

Number of episodes to run

20

record

bool

Whether to record the best, worse, and average episodes.

False

video_dir

str

Path to save video recording, if enabled

"results"

show_progress

bool

Whether to show progress bar

True

throw_errors

bool

Whether to raise exceptions

True

timeout

int

Global timeout in seconds.

600

use_custom_eval

bool

Use the custom evaluation function for a competition

False

Returns: BenchmarkResults containing status, score, duration and any errors

Example: 


Copy
# Benchmark random agent
results = sai.benchmark()
print(f"Score: {results['score']}")

# Benchmark PyTorch model with video
results = sai.benchmark(model)
submit
Submits a model to the loaded competition. `submit_model` is an alias for this function and has the same functionality.


Copy
submission = sai.submit(
    name: str,
    model: ModelType,
    action_function: Optional[str | Callable] = None,
    preprocessor_class: Optional[str | type] = None,
    *,
    model_type: Optional[ModelLibraryType] = None,
    algorithm: Optional[str] = None,
    use_onnx: bool = False,
    tag: str = "default",
)
Parameter
Type
Description
Default
model

ModelType

Model to 

-

name

str

Name for the submission

-

algorithm

str

Name of algorithm used for Stable Baslines 3 model

None

action_function

Callable | str

Action function to submit alongside the model.

None

preprocessor_class

type | str

A class that can be used to pre-process inputs into the model.

None

model_type

ModelLibraryType

Framework used

None

use_onnx

bool

Convert model to ONNX format

False

tag

str

Group models together with similar tags

"default"

Returns: A dictionary containing the submission details

Example:


Copy
submission = sai.submit(
    "My First CNN Model", 
    model,
    tag="cnn"
)
save_model
Saves a model to a specified path.


Copy
path = sai.save_model(
    name: str,
    model: ModelType,
    model_type: Optional[ModelLibraryType] = None,
    algorithm: Optional[str] = None,
    use_onnx: bool = False,
    output_path: str = "./",
)
Parameter
Type
Description
Default
name

str

Name for saved model file

-

model

ModelType

Model to save

-

model_type

ModelLibraryType

Framework used

None

algorithm

str

Name of algorithm used for Stable Baslines 3 model

None

use_onnx

bool

Convert to ONNX format

False

output_path

str

Directory to save file

"./"


Returns: str containing full path to saved model file

Example:


Copy
path = sai.save_model(
    name="my_model",
    model=model,
    model_type="pytorch",
    output_path="./models"
)  # Saves to ./models/my_model.pt
load_competition
Loads the details of a specific competition, used to override a previously set competition or environment.


Copy
sai.load_competition(competition_id: str)
Parameter
Type
Description
competition_id

str

The ID of the competition to load

Returns: dict containing loaded competition information

Example: 


Copy
sai.load_competition("comp-12345")
sai.watch()  # Watch random agents
load_environment
Loads the details of a specific environment, used to override a previously set competition or environment.


Copy
 sai.load_environment(environment_id: str)
Parameter
Type
Description
environment_id

str

The ID of the environment to load

Returns: dict containing loaded environment information

Example: 


Copy
sai.load_environment("SquidHunt-v0")
sai.watch()  # Watch random agents
Error Handling
The SAIClient methods may raise exceptions in case of errors. It's recommended to wrap calls in try-except blocks to handle potential errors gracefully.


Copy
try:
    sai.load_competition("competition_id")
except Exception as e:
    print(f"An error occurred: {e}")
Best Practices
Always check if a competition is loaded before calling methods that require it.

Use environment variables or the store_api_key method to securely manage your API key.

Close the environment (env.close()) after you're done using it to free up resources.

Use the benchmark method to test your model locally before submitting.

For more detailed information on specific use cases and advanced features, please refer to the SAI platform documentation and API reference.
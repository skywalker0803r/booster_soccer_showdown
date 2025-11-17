Basic Flow
Covering the basics to create and submit your first AI model in SAI.

Ready to dive into your first SAI competition? Let's get you up and running with a basic model in no time!

1. Choose a Competition
First, head over to the SAI website and log in to your account. Navigate to the "Competitions" page to see the list of available challenges.


Competitions Page
Choose a competition that interests you. For this guide, we'll use the "Squid Hunt Challenge" as an example.

2. Get the Competition ID
Once you've selected a competition, you'll need its unique ID. You can find this at the end of the URL on the competition page, which will start with "cmp_". Or you can copy the slug by clicking on the  icon beside the competition name in the header, which will look something like your-competition-id. Both the slug and the competition id will work for initializing the competition in the SAI client.

3. Create the Environment
Now, let's use the SAI Client to create the environment in Python. Open your favourite Python editor and start with this code:


Copy
from sai_rl import SAIClient

# Initialize the SAI Client with your competition ID
sai = SAIClient(comp_id="your-competition-id")

# Create the environment
env = sai.make_env()
Note: For more details on the SAI Client and its capabilities, check out our SAI Client Documentation.

4. Train a Basic Model
We'll use Stable Baselines 3 to quickly train a basic model:


Copy
# Create and train the model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2048)
5. Local Evaluation
If you are using a version of sai-rl < 0.1.23 then this function will be named sai.benchmark. We changed the name to sai.evaluate in order to remove confusion with our platform benchmarks.

Before submitting, let's run a local evaluation to see how our model performs:


Copy
# Run the evaluation
sai.evaluate(model)
You should get an output in your console that looks like this:


Copy
Results:
{
    Evaluation of 20 episodes completed in 2.00 seconds
    Average Episode Score: 4.90
    Average Duration per Batch: 8.17 seconds
    Average FPS Across All Batches: 4892.65
    Total Timesteps: 9800
}
Some competitions use an evaluation function that is different than the reward function for the underlying environment. As such, you will notice that the score you get on a local evaluation will deviate from the score received on a submission. To use the same evaluation function locally, you can set sai.benchmark(model, use_custom_eval=True).

When using a custom eval, you are executing arbitrary code on your local machine, so use at your own discretion. All evaluation code will be visible on the SAI platform, so if you have any concerns, please check it out on the website before agreeing to use the custom eval function.

6. Save and Submit Your Model
Finally, let's save our model and submit it to the platform! There are two ways to submit a model to a competition and below we show how to do it through python:


Copy
sai.submit("My First Model", model)
7. See the Results

Dashboard
When you submit a model, you will be able to see it "running" in the submissions table on the dashboard. Once it's done benchmarking it will be marked as "completed".

Once the submission is done running, you will be able to click on it to see more information. A detailed overview of the results can be found here.

Complete Script
Here's the complete script for quick reference:


Copy
from sai_rl import SAIClient
from stable_baselines3 import PPO

sai = SAIClient(comp_id="your-competition-id")
env = sai.make_env()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2048)

sai.benchmark(model)

sai.submit("My First Squid Model", model)

env.close()
Congratulations! You've just created, trained, benchmarked, and submitted your first model to a SAI competition. Head back to the SAI website to track your submission's performance and see how it stacks up against other competitors.

Happy coding, and may the best AI win!
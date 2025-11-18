Description
This competition challenges participants to build a single, versatile soccer-playing agent capable of mastering three distinct, but related, environments:

Penalty Kick with Goalie, where the agent must score against a moving goalkeeper
Targeted Shot with Obstacles, where the agent aims for specific regions of the net while avoiding multiple moving blockers
Precision Pass, where the agent delivers an accurate kick to a designated target on the field.
Competitors must design policies that not only learn each individual skill but also generalize across all three tasks without per-task tuning, showcasing robust soccer intelligence.

Competition Ends

Fri Jan 09 2026

To learn more about this scene, please click here.

Evaluation
Submissions will be evaluated on aggregate performance across the full scene: the same trained model must run unmodified on all 3 environments. Each environment provides its own episodic reward, and a model's final score will be the weighted average of normalized returns over extensive evaluation episodes in each task. The weights for each task are defined below:

Task 1

LowerT1GoaliePenaltyKick-v0

Weight

1

Task 2

LowerT1ObstaclePenaltyKick-v0

Weight

1

Task 3

LowerT1KickToTarget-v0

Weight

1

Description
The Soccer Scene provides a simple, consistent setting to explore ball-control and kicking skills. Agents face a moving goalie, a target-shooting challenge with static defenders, and a passing drill to a marked spot. Because each task uses the same physics and action space, progress depends on learning soccer fundamentals that carry across all three situations.


Penalty Kick with Goalie (Lower T1)


Penalty Kick with Obstacles (Lower T1)


Kick to Target (Lower T1)

Models benchmarked on this scene are evaluated on all soccer tasks using a single model. The final score reflects that agent's ability to generalize across penalty kicks, target shots, and passing drills, highlighting true cross-task robustness.

Action Space	Box(shape=(12,), low=[-45,-45,-30,-65,-24,-15,-45,-45,-30,-65,-24,-15], high=[45,45,30,65,24,15,45,45,30,65,24,15])
Actions
The action space is a continuous vector of shape (12,), where each dimension corresponds to a joint torque command for the T1's articulated parts. The table below describes each dimension, interpreted by the joint torque controllers to compute joint commands.

The action space below is shared by all environments in this scene:

Index	Action
0	hip_y_left torque
1	hip_x_left torque
2	hip_z_left torque
3	knee_y_left torque
4	ankle_y_left torque
5	ankle_x_left torque
6	hip_y_right torque
7	hip_x_right torque
8	hip_z_right torque
9	knee_y_right torque
10	ankle_y_right torque
11	ankle_x_right torque
To understand how the indices can be grouped to understand actions at a higher level, refer to the descriptions below:

Left Leg Control

Description: Controls the left leg joints including hip, knee, and ankle for walking, balance and scoring a goal.
Indices: 0-5
Right Leg Control

Description: Controls the right leg joints including hip, knee, and ankle for walking, balance and scoring a goal.
Indices: 6-11
Observations
Below we will show the shared features across all environments in this scene. In total there are 33 that are shared:

Feature Group	Indices
Joint Positions	[0, 11]
Joint Velocities	[12, 23]
Ball Position (Relative to Robot)	[24, 26]
Ball Linear Velocity (Relative to Robot)	[27, 29]
Ball Angular Velocity (Relative to Robot)	[30, 32]
Each environment also has the follow info that is shared:

Name	Type	Shape	Example
length	float	N/A	10.97
width	float	N/A	6.87
goal_width	float	N/A	1.6
goal_height	float	N/A	1.9
goal_depth	float	N/A	1.6
goal_team_0_rel_robot	ndarray	3	[-4.2, 0, -0.7]
goal_team_1_rel_robot	ndarray	3	[17.74, 0, -0.7]
goal_team_0_rel_ball	ndarray	3	[-2.2, 0, 0]
goal_team_1_rel_ball	ndarray	3	[19.74, 0, 0]
ball_xpos_rel_robot	ndarray	3	[-2, 0, -0.7]
ball_velp_rel_robot	ndarray	3	[0, 0, 0]
ball_velr_rel_robot	ndarray	3	[0, 0, 0]
player_team	ndarray	2	[1, 0]
robot_accelerometer	ndarray	3	[0, 0, 0]
robot_gyro	ndarray	3	[0, 0, 0]
robot_velocimeter	ndarray	3	[0, 0, 0]
robot_quat	ndarray	4	[0, 0, 1, 0.0007999999797903001]
goalkeeper_team_0_xpos_rel_robot	ndarray	3	[0, 0, 0]
goalkeeper_team_0_velp_rel_robot	ndarray	3	[0, 0, 0]
goalkeeper_team_1_xpos_rel_robot	ndarray	3	[0, 0, 0]
goalkeeper_team_1_velp_rel_robot	ndarray	3	[0, 0, 0]
target_xpos_rel_robot	ndarray	3	[-4.2, -1.44, 0.1333]
target_velp_rel_robot	ndarray	3	[0, 0, 0]
defender_xpos	ndarray	9	[-10.9707, -0.4792, 0.2724, -10.9707, 0.4808, 0.2724, -10.9707, 1.4408, 0.2724]
success	bool	N/A	false
An example of how we can use the preprocessor to create a shared state space across tasks in the scene is as follows:

python

Copy

class Preprocessor():
    def get_task_onehot(self, info):
        if "task_index" in info:
            return info["task_index"]
        else:
            return np.array([])

    def modify_state(self, obs, info):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        obs = obs[:,:32]

        task_onehot = self.get_task_onehot(info)
        if len(task_onehot.shape) == 1:
            task_onehot = np.expand_dims(task_onehot, axis=0)
        
        return np.hstack((obs, task_onehot))
This is only an example, and we encourage you to create your own!

Create Scene
To instantiate this scene and load a particular task, you can use:

python

Copy

from sai_rl import SAIClient

sai = SAIClient(scene_id="scn_fk2IPfTF7cVe")
env = sai.make_env("LowerT1KickToTarget-v0")
Alternatively, you can use the task index to initialize a scene environment instead of the gym id as follows: env = sai.make_env(2)

However, regardless which environment you create with the SAI client, when you run a local evaluation, it will evaluate the entire scene, which means it evaluates all tasks in the scene.
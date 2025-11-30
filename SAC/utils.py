# utils.py
import numpy as np

class Preprocessor:
    def get_task_onehot(self, info):
        if 'task_index' in info:
            return info['task_index']
        else:
            return np.array([])

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        q_w = q[:, [-1]]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v).reshape(-1,1) * 2.0)
        return a - b + c 

    def modify_state(self, obs, info):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)

        task_onehot = self.get_task_onehot(info)
        if len(task_onehot.shape) == 1:
            task_onehot = np.expand_dims(task_onehot, axis=0)

        # Expand dims for all relevant info fields
        for key in ["robot_quat", "robot_gyro", "robot_accelerometer", "robot_velocimeter",
                    "goal_team_0_rel_robot", "goal_team_1_rel_robot", "goal_team_0_rel_ball",
                    "goal_team_1_rel_ball", "ball_xpos_rel_robot", "ball_velp_rel_robot",
                    "ball_velr_rel_robot", "player_team", "goalkeeper_team_0_xpos_rel_robot",
                    "goalkeeper_team_0_velp_rel_robot", "goalkeeper_team_1_xpos_rel_robot",
                    "goalkeeper_team_1_velp_rel_robot", "target_xpos_rel_robot", 
                    "target_velp_rel_robot", "defender_xpos"]:
            if key in info and len(info[key].shape) == 1:
                info[key] = np.expand_dims(info[key], axis=0)

        robot_qpos = obs[:, :12]
        robot_qvel = obs[:, 12:24]
        quat = info["robot_quat"]
        base_ang_vel = info["robot_gyro"]
        project_gravity = self.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))

        obs = np.hstack((
            robot_qpos,
            robot_qvel,
            project_gravity,
            base_ang_vel,
            info["robot_accelerometer"],
            info["robot_velocimeter"],
            info["goal_team_0_rel_robot"],
            info["goal_team_1_rel_robot"],
            info["goal_team_0_rel_ball"],
            info["goal_team_1_rel_ball"],
            info["ball_xpos_rel_robot"],
            info["ball_velp_rel_robot"],
            info["ball_velr_rel_robot"],
            info["player_team"],
            info["goalkeeper_team_0_xpos_rel_robot"],
            info["goalkeeper_team_0_velp_rel_robot"],
            info["goalkeeper_team_1_xpos_rel_robot"],
            info["goalkeeper_team_1_velp_rel_robot"],
            info["target_xpos_rel_robot"],
            info["target_velp_rel_robot"],
            info["defender_xpos"],
            task_onehot
        ))

        return obs


def action_function(policy, env):
    expected_bounds = [-1, 1]
    action_percent = (policy - expected_bounds[0]) / (expected_bounds[1] - expected_bounds[0])
    bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
    return env.action_space.low + (env.action_space.high - env.action_space.low) * bounded_percent

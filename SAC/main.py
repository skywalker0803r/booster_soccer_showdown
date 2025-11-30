from sac_agent import SACAgent
from utils import Preprocessor
from sai_rl import SAIClient

# 環境
sai = SAIClient(
    comp_id="booster-soccer-showdown", 
    api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv",
)
env = sai.make_env()
obs_raw,info = env.reset()
obs_dim = len(Preprocessor().modify_state(obs_raw, info))
act_dim = env.action_space.shape[0]

agent = SACAgent(obs_dim, act_dim, env)

for episode in range(1000):
    obs_raw,info = env.reset()
    done = False
    while not done:
        action = agent.select_action(obs_raw, info)
        next_obs_raw, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_obs = Preprocessor().modify_state(next_obs_raw, info)
        agent.buffer.push(
            Preprocessor().modify_state(obs_raw, info),
            action,
            reward,
            next_obs,
            done
        )
        agent.update()
        obs_raw = next_obs_raw

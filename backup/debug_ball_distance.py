"""
Debug ball distance issue in success_focused_training.py
"""

import numpy as np
from sai_rl import SAIClient

def debug_ball_info():
    """Debug what ball info we actually get"""
    
    print("🔍 Debugging Ball Distance Issue")
    print("="*50)
    
    sai = SAIClient(comp_id="booster-soccer-showdown", api_key="sai_LFcuaCZiqEkUbNVolQ3wbk5yU7H11jfv")
    env = sai.make_env()
    
    obs, info = env.reset()
    
    print("📊 Info keys available:")
    for key in sorted(info.keys()):
        print(f"   {key}: {type(info[key])}")
    
    print("\n🎾 Ball-related info:")
    ball_keys = [k for k in info.keys() if 'ball' in k.lower()]
    for key in ball_keys:
        value = info[key]
        print(f"   {key}:")
        print(f"      Type: {type(value)}")
        print(f"      Shape: {getattr(value, 'shape', 'N/A')}")
        print(f"      Value: {value}")
        
        if hasattr(value, 'shape') and len(value.shape) > 0:
            if len(value.shape) > 1:
                print(f"      [0]: {value[0]}")
            
            # Try to compute distance
            try:
                if len(value.shape) > 1:
                    dist = np.linalg.norm(value[0])
                else:
                    dist = np.linalg.norm(value)
                print(f"      Distance: {dist:.3f}")
            except Exception as e:
                print(f"      Distance error: {e}")
        print()
    
    # Test a few steps to see if it changes
    print("🔄 Testing ball position over 5 steps:")
    for step in range(5):
        action = env.action_space.sample() * 0.1
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        
        ball_pos = next_info.get("ball_xpos_rel_robot", None)
        print(f"   Step {step}: {ball_pos}")
        
        if ball_pos is not None:
            try:
                if len(ball_pos.shape) > 1:
                    dist = np.linalg.norm(ball_pos[0])
                else:
                    dist = np.linalg.norm(ball_pos)
                print(f"      → Distance: {dist:.3f}")
            except Exception as e:
                print(f"      → Error: {e}")
        
        if terminated or truncated:
            break

if __name__ == "__main__":
    debug_ball_info()
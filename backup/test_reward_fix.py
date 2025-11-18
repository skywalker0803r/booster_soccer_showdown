"""
Quick test to verify our reward fixing works correctly
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reward_shaping_wrapper import SmartRewardShaper

def test_step_penalty_replacement():
    """Test that we correctly replace the -1.0 step penalty"""
    
    shaper = SmartRewardShaper(step_penalty_scale=0.01)
    
    # Test scenarios
    test_cases = [
        {
            'name': 'Standard step penalty (-1.0)',
            'original_reward': -1.0,
            'info': {
                'ball_xpos_rel_robot': np.array([2.0, 0, 0]),
                'robot_quat': np.array([0, 0, 0.1, 0.995]),  # Upright
                'goal_team_1_rel_robot': np.array([5.0, 0, 0]),
                'ball_velp_rel_robot': np.array([0, 0, 0]),
                'robot_velocimeter': np.array([0.5, 0, 0])
            },
            'expected_improvement': True
        },
        {
            'name': 'Zero reward',
            'original_reward': 0.0,
            'info': {
                'ball_xpos_rel_robot': np.array([2.0, 0, 0]),
                'robot_quat': np.array([0, 0, 0.1, 0.995]),  # Upright
                'goal_team_1_rel_robot': np.array([5.0, 0, 0]),
                'ball_velp_rel_robot': np.array([0, 0, 0]),
                'robot_velocimeter': np.array([0.5, 0, 0])
            },
            'expected_improvement': True
        },
        {
            'name': 'Robot fallen (-1.5)',
            'original_reward': -1.5,
            'info': {
                'ball_xpos_rel_robot': np.array([2.0, 0, 0]),
                'robot_quat': np.array([0, 0, 0.8, 0.6]),  # Fallen
                'goal_team_1_rel_robot': np.array([5.0, 0, 0]),
                'ball_velp_rel_robot': np.array([0, 0, 0]),
                'robot_velocimeter': np.array([0, 0, 0])
            },
            'expected_improvement': False  # Should stay negative
        },
        {
            'name': 'Positive reward (+2.5)',
            'original_reward': 2.5,
            'info': {
                'ball_xpos_rel_robot': np.array([0.5, 0, 0]),
                'robot_quat': np.array([0, 0, 0.1, 0.995]),  # Upright
                'goal_team_1_rel_robot': np.array([5.0, 0, 0]),
                'ball_velp_rel_robot': np.array([1.0, 0, 0]),
                'robot_velocimeter': np.array([0.5, 0, 0])
            },
            'expected_improvement': True  # Should get bonus
        }
    ]
    
    print("🧪 Testing Step Penalty Replacement")
    print("=" * 50)
    
    for test_case in test_cases:
        obs = np.random.randn(89)
        
        shaped_reward = shaper.shape_reward(
            obs,
            test_case['info'],
            test_case['original_reward'],
            terminated=False,
            truncated=False
        )
        
        improvement = shaped_reward - test_case['original_reward']
        
        print(f"\n{test_case['name']}:")
        print(f"  Original:  {test_case['original_reward']:+8.3f}")
        print(f"  Shaped:    {shaped_reward:+8.3f}")
        print(f"  Change:    {improvement:+8.3f}")
        
        if test_case['expected_improvement'] and improvement <= 0:
            print("  ❌ Expected improvement but got worse!")
        elif not test_case['expected_improvement'] and test_case['original_reward'] < 0 and shaped_reward > 0:
            print("  ❌ Negative reward became positive unexpectedly!")
        else:
            print("  ✅ Behavior as expected")
        
        shaper.reset()
    
    print("\n" + "=" * 50)
    
    # Simulate a full episode
    print("🎯 Simulating Episode with -1.0 Step Penalty")
    episode_length = 200
    total_original = 0
    total_shaped = 0
    
    shaper.reset()
    for step in range(episode_length):
        original_reward = -1.0  # Standard step penalty
        
        # Simulate some stability
        info = {
            'ball_xpos_rel_robot': np.array([2.0, 0, 0]),
            'robot_quat': np.array([0, 0, 0.1, 0.995]),  # Upright
            'goal_team_1_rel_robot': np.array([5.0, 0, 0]),
            'ball_velp_rel_robot': np.array([0, 0, 0]),
            'robot_velocimeter': np.array([0.5, 0, 0])
        }
        
        shaped_reward = shaper.shape_reward(
            np.random.randn(89),
            info,
            original_reward,
            terminated=False,
            truncated=False
        )
        
        total_original += original_reward
        total_shaped += shaped_reward
    
    print(f"Episode Results ({episode_length} steps):")
    print(f"  Original Total: {total_original:+8.3f}")
    print(f"  Shaped Total:   {total_shaped:+8.3f}")
    print(f"  Improvement:    {total_shaped - total_original:+8.3f}")
    
    if total_shaped > -50:  # Much less negative than -200
        print("  ✅ MAJOR IMPROVEMENT! Step penalty successfully reduced!")
    else:
        print("  ❌ Step penalty reduction failed!")

if __name__ == "__main__":
    test_step_penalty_replacement()
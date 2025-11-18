"""
Quick patch for the ball distance issue
"""

def create_robust_ball_distance_function():
    """Create a robust function to get ball distance"""
    
    def get_ball_distance(info):
        """Robustly extract ball distance from info"""
        
        # Try different ball position keys
        ball_keys = ['ball_xpos_rel_robot', 'ball_pos_rel_robot', 'ball_position']
        ball_pos = None
        
        for key in ball_keys:
            if key in info:
                ball_pos = info[key]
                break
        
        if ball_pos is None:
            print(f"⚠️ No ball position found! Available keys: {list(info.keys())}")
            return float('inf')
        
        # Handle different shapes/types
        try:
            # Convert to numpy if needed
            if not isinstance(ball_pos, np.ndarray):
                ball_pos = np.array(ball_pos)
            
            # Handle different dimensions
            if len(ball_pos.shape) > 1:
                ball_pos = ball_pos[0]  # Take first element
            
            # Ensure it's 3D position
            if len(ball_pos) < 3:
                print(f"⚠️ Ball position has wrong dimension: {ball_pos}")
                return float('inf')
            
            # Calculate distance (only X,Y - ignore Z for 2D distance)
            distance = np.linalg.norm(ball_pos[:2])
            
            # Sanity check
            if distance < 0 or distance > 100:  # Unreasonable distance
                print(f"⚠️ Unreasonable ball distance: {distance}")
                return float('inf')
                
            return float(distance)
            
        except Exception as e:
            print(f"❌ Error calculating ball distance: {e}")
            print(f"   Ball pos: {ball_pos}")
            print(f"   Type: {type(ball_pos)}")
            return float('inf')
    
    return get_ball_distance

# Patch for robot upright calculation
def get_robot_upright(info):
    """Robustly get robot upright status"""
    
    try:
        robot_quat = info.get("robot_quat", None)
        if robot_quat is None:
            return 0.5  # Default moderate upright
        
        if isinstance(robot_quat, np.ndarray) and len(robot_quat.shape) > 1:
            robot_quat = robot_quat[0]
        
        # Quaternion [x, y, z, w] - z component indicates tilt
        if len(robot_quat) >= 4:
            z_component = robot_quat[2]
            upright = 1.0 - abs(z_component)
            return max(0.0, min(1.0, upright))  # Clamp to [0,1]
        else:
            return 0.5
            
    except Exception as e:
        print(f"❌ Error calculating robot upright: {e}")
        return 0.5

# Test the patches
if __name__ == "__main__":
    import numpy as np
    
    # Test ball distance function
    get_ball_distance = create_robust_ball_distance_function()
    
    # Mock info data for testing
    test_cases = [
        {'ball_xpos_rel_robot': np.array([2.0, 1.0, 0.5])},
        {'ball_xpos_rel_robot': np.array([[2.0, 1.0, 0.5]])},
        {'ball_xpos_rel_robot': [2.0, 1.0, 0.5]},
        {'other_key': 'no_ball'},
        {}
    ]
    
    print("🧪 Testing ball distance function:")
    for i, test_info in enumerate(test_cases):
        dist = get_ball_distance(test_info)
        print(f"   Test {i+1}: {dist:.3f}")
    
    print("\n🧪 Testing robot upright function:")
    test_quats = [
        {'robot_quat': np.array([0, 0, 0.1, 0.995])},  # Upright
        {'robot_quat': np.array([[0, 0, 0.8, 0.6]])},   # Tilted  
        {'robot_quat': [0, 0, 0.1, 0.995]},             # List format
        {}  # Missing
    ]
    
    for i, test_info in enumerate(test_quats):
        upright = get_robot_upright(test_info)
        print(f"   Test {i+1}: {upright:.3f}")
"""
Test script to verify SimpleDreamerV3 implementation
"""

import torch
import numpy as np
from simple_dreamerv3 import SimpleDreamerV3

def test_model_creation():
    """Test if model can be created without errors"""
    print("Testing model creation...")
    
    try:
        model = SimpleDreamerV3(
            obs_dim=89,
            action_dim=12,
            hidden_dim=256,
            stoch_dim=32,
            discrete_dim=16
        )
        print("✓ Model created successfully")
        return model
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return None

def test_forward_pass(model):
    """Test forward pass with dummy data"""
    print("Testing forward pass...")
    
    try:
        # Test single observation
        obs = np.random.randn(89)
        action, state = model.select_action(obs)
        print(f"✓ Single observation processed, action shape: {action.shape}")
        
        # Test batch observation
        obs_tensor = torch.randn(1, 89)
        action_tensor = model(obs_tensor)
        print(f"✓ Batch observation processed, action shape: {action_tensor.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def test_training_step(model):
    """Test a single training step"""
    print("Testing training step...")
    
    try:
        # Create dummy data
        batch_size = 4
        seq_len = 10
        obs_dim = 89
        action_dim = 12
        
        obs_seq = torch.randn(batch_size, seq_len, obs_dim)
        action_seq = torch.randn(batch_size, seq_len, action_dim)
        reward_seq = torch.randn(batch_size, seq_len)
        
        # Training step
        losses = model.train_step(obs_seq, action_seq, reward_seq)
        
        print("✓ Training step completed")
        print(f"  World Model Loss: {losses['world_model_loss']:.4f}")
        print(f"  Actor Loss: {losses['actor_loss']:.4f}")
        print(f"  Critic Loss: {losses['critic_loss']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== Testing SimpleDreamerV3 ===\n")
    
    # Test model creation
    model = test_model_creation()
    if model is None:
        return
    
    print()
    
    # Test forward pass
    if not test_forward_pass(model):
        return
    
    print()
    
    # Test training step
    if not test_training_step(model):
        return
    
    print("\n=== All tests passed! ===")

if __name__ == "__main__":
    main()
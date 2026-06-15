#!/usr/bin/env python3
from matplotlib import collections
import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define joint names in the exact dataset order
JOINT_NAMES = [
    'arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint', 'arm_left_4_joint',
    'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint', 'arm_right_1_joint',
    'arm_right_2_joint', 'arm_right_3_joint', 'arm_right_4_joint', 'arm_right_5_joint',
    'arm_right_6_joint', 'arm_right_7_joint', 'torso_lift_joint', 'head_1_joint', 'head_2_joint'
]

# physical joint limits (in rad, except torso which is in m)
LEFT_LIMITS_DEG = [(-280.0, 150.0), (-140.0, 0.0), (-160.0, 160.0), (-140.0, 0.0), (-100.0, 100.0), (-108.0, 108.0), (-150.0, 150.0)]
RIGHT_LIMITS_DEG = [(-40.0, 270.0), (-140.0, 0.0), (-160.0, 160.0), (-140.0, 0.0), (-220.0, -20.0), (-108.0, 108.0), (-150.0, 150.0)]
EXTRA_LIMITS_RAD = [(0.0, 0.35), (-1.309, 1.309), (-1.0472, 0.34907)]

def get_joint_limits():
    limits = []
    for lo, hi in LEFT_LIMITS_DEG:
        limits.append((np.radians(lo), np.radians(hi)))
    for lo, hi in RIGHT_LIMITS_DEG:
        limits.append((np.radians(lo), np.radians(hi)))
    limits.extend(EXTRA_LIMITS_RAD)
    return np.array(limits)

JOINT_LIMITS = get_joint_limits()

# avg gradient per joint (from evaluate_nn.py)
AVG_GRADIENTS = np.array([
    0.454807, 0.136608, 0.172546, 0.083021, 0.030820, 0.042739, 0.009463, # left arm
    0.405223, 0.177221, 0.198139, 0.092835, 0.033405, 0.045261, 0.009018, # right arm
    0.048439,                                                             # torso
    0.014133, 0.015917                                                    # head
])

# neural network architecture definition
class DynamicCollisionPredictor(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.num_layers = len(layer_sizes)
        in_size = 17
        for i, out_size in enumerate(layer_sizes):
            setattr(self, f'fc{i+1}', nn.Linear(in_size, out_size))
            in_size = out_size
        self.out = nn.Linear(in_size, 2)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        for i in range(self.num_layers):
            fc = getattr(self, f'fc{i+1}')
            x = F.relu(fc(x))
            if i < self.num_layers - 1:
                x = self.drop(x)
        return self.out(x)

def load_resources(model_name="model_8"):
    # load model performance structure to determine layer configurations dynamically
    with open("model_performance.json", "r") as f:
        performance_data = json.load(f)
    
    m_info = performance_data[model_name]
    layer_sizes = m_info["layer_sizes"]
    
    # instantiate model and load weights
    model = DynamicCollisionPredictor(layer_sizes)
    model.load_state_dict(torch.load(f"models/{model_name}.pt", map_location="cpu"))
    model.eval()
    
    # load feature scaler
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    return model, scaler

def adjust_pose_to_safe(initial_pose, model_name="model_8", lr=0.05, max_iters=100, target_prob=0.1, use_importance_weighting=False):

    model, scaler = load_resources(model_name)
    current_pose = np.array(initial_pose, dtype=np.float32).copy()
    
    print(f"pose adjusting using {model_name.upper()}")
    
    # retrieve scaling parameters to compute gradients in raw joint space correctly
    mean = scaler.mean_
    scale = scaler.scale_
    
    success = False
    history_prob = []
    consecutive_zeros = 0
    
    for i in range(max_iters):
        # prepare input tensor with requires_grad=True
        # scale the joints manually to preserve tracking of the gradient through scaling:
        # x_scaled = (x_raw - mean) / scale
        pose_tensor = torch.tensor(current_pose, dtype=torch.float32, requires_grad=True)
        scaled_tensor = (pose_tensor - torch.tensor(mean, dtype=torch.float32)) / torch.tensor(scale, dtype=torch.float32)
        
        # forward pass to compute collision probability
        logits = model(scaled_tensor.unsqueeze(0))
        probs = F.softmax(logits, dim=1)[0]
        prob_collision = probs[1].item()
        prediction = torch.argmax(probs).item()
        
        history_prob.append(prob_collision)
        
        # print current state
        status = "COLLISION" if prediction == 1 else "SAFE"
        print(f"iter {i:2d} | P(collision): {prob_collision:.4f} | prediction: {status}")
        
        # check stop condition (pose safe, probability < 10%)
        if prediction == 0 and prob_collision < target_prob:
            print(f"\n success! safe pose was found after {i} iterations")
            success = True
            break
            
        # backward pass to find gradient of collision logit 
        model.zero_grad()
        loss = logits[0, 1] - logits[0, 0]
        loss.backward()
        
        # extract gradient in raw joint space
        grad_raw = pose_tensor.grad.numpy()
        
        # normalise the gradient to control the step size
        # this prevents wild, massive joint jumps and guarantees that the step size
        if use_importance_weighting:
            # scale the gradient direction for each joint by its average absolute gradient (importance)
            normalized_importance = AVG_GRADIENTS / np.max(AVG_GRADIENTS)
            grad_processed = grad_raw * normalized_importance
        else:
            grad_processed = grad_raw
            
        grad_norm = np.linalg.norm(grad_processed)
        print(f"  -> grad_norm: {grad_norm:.6f}")
        if grad_norm > 1e-8:
            step = lr * (grad_processed / grad_norm)
            consecutive_zeros = 0
        else:
            consecutive_zeros += 1
            # fallback
            noise_std = 0.005 * consecutive_zeros
            step = np.random.normal(0, noise_std, size=grad_raw.shape)
            
        # update pose
        current_pose -= step
        
        # clip to physical joint limits
        current_pose = np.clip(current_pose, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        
    if not success:
        print("\n oh ni")
        
    return current_pose, success, history_prob

def plot_avoidance_results(history_prob, initial_pose, final_pose, save_path="collision_avoidance_plot.png"):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # plot probability convergence
    ax1.plot(history_prob, marker='o', color='#DC2626', linewidth=2)
    ax1.axhline(y=0.5, color='#64748B', linestyle='--', label='collision boundary (0.5)')
    ax1.axhline(y=0.1, color='#64748B', linestyle='--', label='target P(collision) (0.1)')
    ax1.set_title("collision probability over iterations", fontsize=12, fontweight='bold')
    ax1.set_xlabel("iteration", fontsize=10)
    ax1.set_ylabel("probability of collision", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()
    
    # plot joint changes
    changes = np.array(final_pose) - np.array(initial_pose)
    colors = ['#1D4ED8' if c >= 0 else '#B91C1C' for c in changes]
    
    y_pos = np.arange(len(JOINT_NAMES))
    ax2.barh(y_pos, changes, color=colors, edgecolor='none')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(JOINT_NAMES, fontsize=8)
    ax2.invert_yaxis()
    ax2.axvline(x=0, color='#0F172A', linestyle='-', linewidth=0.8)
    ax2.set_title("joint adjustment (adjusted - original)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("change (radians/meters in torso case)", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n visualization saved to '{save_path}'")

if __name__ == "__main__":
    # example preset poses
    collision_pose = [
        2.58869805,-2.26445124,-1.60204854,-1.79545409,1.51236053,1.43582411,1.98585435,1.30120226,-2.05801273,1.86398346,-0.72438862,-1.70456616,1.83682538,0.80621811,0.00273809,0.83017863,-0.62918643
    ]
    collision_pose = [
        0.76995829,-0.11231412,-1.10614310,-0.67712572,-1.70541101,0.57937264,1.00933398,-0.36200672,-2.15458315,-1.07899694,-1.45284129,-2.08559700,1.48956116,1.06582214,0.10884228,-1.00160558,0.23196537
    ]
    collision_pose = [
        -2.23490459,-2.25108026,-1.10195872,-2.23906447,1.21599294,0.82904078,-1.26018883,3.04797741,-0.08634503,0.30231728,-2.36245067,-2.92726750,-0.71575781,-2.43694747,0.05190530,0.99539199,-0.68744898
    ]
    # run standard gradient descent adjustment
    safe_pose, is_safe, history_prob = adjust_pose_to_safe(
        initial_pose=collision_pose,
        model_name="model_8",
        lr=0.08,
        max_iters=100,
        target_prob=0.1,
        use_importance_weighting=False
    )
    
    # display joint angle differences
    print("\n pose modification dets")
    print(f"{'joint name':<25} | {'original (rad)':<15} | {'adjusted (rad)':<15} | {'change (rad)':<12}")
    print("-" * 75)
    for idx, name in enumerate(JOINT_NAMES):
        diff = safe_pose[idx] - collision_pose[idx]
        print(f"{name:<25} | {collision_pose[idx]:15.5f} | {safe_pose[idx]:15.5f} | {diff:+12.5f}")
        
    # generate and save the visualization plots
    plot_avoidance_results(history_prob, collision_pose, safe_pose)

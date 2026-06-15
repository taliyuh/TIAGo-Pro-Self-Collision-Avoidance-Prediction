import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import time
import numpy as np

class SelfCollisionPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(17, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)

        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))


        return self.out(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading data...")
data = pd.read_csv('tiago_collision_dataset_17dof_1M.csv', engine='pyarrow', dtype=float)
X = data.iloc[:, :17].values
y = data['collision'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit on train to prevent data leakage
X_test_scaled = scaler.transform(X_test)

test_dataset = TensorDataset(
    torch.tensor(X_test_scaled, dtype=torch.float32), 
    torch.tensor(y_test, dtype=torch.long)
)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = SelfCollisionPredictor().to(device)

MODEL_PATH = 'models/model_1.pt' 

print(f"Loading model from: {MODEL_PATH}")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Running evaluation...")
all_preds = []
all_labels = []
all_probs = []

start_time = time.perf_counter()

with torch.no_grad():
    for vinputs, vlabels in test_loader:
        vinputs, vlabels = vinputs.to(device), vlabels.to(device)
        outputs = model(vinputs)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(vlabels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

if device.type == 'cuda':
    torch.cuda.synchronize()
end_time = time.perf_counter()

print("\n--- Model Assessment ---")
print(f"Accuracy : {accuracy_score(all_labels, all_preds):.4f}")
print(f"Precision: {precision_score(all_labels, all_preds):.4f}")
print(f"Recall   : {recall_score(all_labels, all_preds):.4f}")
print(f"F1-score : {f1_score(all_labels, all_preds):.4f}")
print(f"AUC-ROC  : {roc_auc_score(all_labels, all_probs):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))


inference_time = end_time - start_time
total_samples = len(all_labels)
print(f"Total Inference Time: {inference_time:.4f} seconds")
print(f"Average Time per Sample: {(inference_time / total_samples) * 1000:.4f} ms")
print(f"Samples per Second: {total_samples / inference_time:.2f}")

### gradient

# JOINT_NAMES = [
#     'arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint', 'arm_left_4_joint',
#     'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint', 'arm_right_1_joint',
#     'arm_right_2_joint', 'arm_right_3_joint', 'arm_right_4_joint', 'arm_right_5_joint',
#     'arm_right_6_joint', 'arm_right_7_joint', 'torso_lift_joint', 'head_1_joint', 'head_2_joint'
# ]
# print("\n" + "="*60)
# print("Gradient computation (∂collision_prob / ∂joint_angles) for all validation samples")
# print("="*60)

# X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)  # <----- POPRAWA
# y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

# model.eval()

# all_grad_norms = []
# all_gradients = []
# all_probs_grad = []
# all_labels_grad = []

# grad_start_time = time.perf_counter()

# for i in range(len(X_test_tensor)):
#     x_i = X_test_tensor[i:i+1].clone().detach().requires_grad_(True)
    
#     logits = model(x_i)
#     prob_collision = F.softmax(logits, dim=1)[0, 1]
    
#     grad = torch.autograd.grad(prob_collision, x_i, retain_graph=False)[0]
#     grad_np = grad.cpu().detach().numpy().flatten()
    
#     all_grad_norms.append(np.linalg.norm(grad_np))
#     all_gradients.append(grad_np)
#     all_probs_grad.append(prob_collision.item())
#     all_labels_grad.append(y_test[i].item())
    
#     if (i + 1) % 10000 == 0:
#         print(f"Processed {i+1} samples...")

# if device.type == 'cuda':
#     torch.cuda.synchronize()
# grad_end_time = time.perf_counter()

# grad_total_time = grad_end_time - grad_start_time
# total_grad_samples = len(all_grad_norms)
# print(f"\nGradient computation completed for {total_grad_samples} samples in {grad_total_time:.2f} seconds.")
# print(f"Average time per sample for gradient: {(grad_total_time / total_grad_samples) * 1000:.4f} ms")

# all_grad_norms = np.array(all_grad_norms)
# all_gradients = np.array(all_gradients)
# all_labels_grad = np.array(all_labels_grad)

# print("\n--- Summary statistics of gradient norms ---")
# print(f"Mean gradient norm: {np.mean(all_grad_norms):.6f}")
# print(f"Std gradient norm:  {np.std(all_grad_norms):.6f}")
# print(f"Min gradient norm:  {np.min(all_grad_norms):.6f}")
# print(f"Max gradient norm:  {np.max(all_grad_norms):.6f}")
# print(f"Median gradient norm: {np.median(all_grad_norms):.6f}")

# mask_collision = (all_labels_grad == 1)
# mask_no_collision = (all_labels_grad == 0)

# if np.any(mask_collision):
#     print(f"\nFor collision samples (true label=1):")
#     print(f"  Mean gradient norm: {np.mean(all_grad_norms[mask_collision]):.6f}")
#     print(f"  Median: {np.median(all_grad_norms[mask_collision]):.6f}")
# if np.any(mask_no_collision):
#     print(f"\nFor non-collision samples (true label=0):")
#     print(f"  Mean gradient norm: {np.mean(all_grad_norms[mask_no_collision]):.6f}")
#     print(f"  Median: {np.median(all_grad_norms[mask_no_collision]):.6f}")

# avg_abs_grad = np.mean(np.abs(all_gradients), axis=0)
# print("\n--- Average absolute gradient per joint (importance) ---")
# for j, name in enumerate(JOINT_NAMES):
#     print(f"  {name:20s}: {avg_abs_grad[j]:.6f}")

# top10_idx = np.argsort(all_grad_norms)[-10:][::-1]
# print("\n--- Top 10 most sensitive configurations (highest gradient norm) ---")
# for idx in top10_idx:
#     print(f"Sample {idx}: true label={all_labels_grad[idx]}, prob={all_probs_grad[idx]:.6f}, grad_norm={all_grad_norms[idx]:.6f}")

# corr = np.corrcoef(all_grad_norms, all_probs_grad)[0,1]
# print(f"\nCorrelation between gradient norm and collision probability: {corr:.4f}")

# print("\nGradient computation for all samples finished.")

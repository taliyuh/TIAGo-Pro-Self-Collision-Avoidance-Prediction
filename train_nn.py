import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import time
import numpy as np

# select between available datasets
LIST = ['1m', '100k', '10k']
SELECTED_LIST = LIST[0]

# paths to datasets
data1m_path = 'tiago_collision_dataset_17dof_1M.csv'
data100k_path = 'tiago_collision_dataset_17dof_100k.csv'
data10k_path = 'tiago_collision_dataset_17dof_10k.csv'

# load the data
if SELECTED_LIST == '1m':   
    data = pd.read_csv(data1m_path, engine='pyarrow', dtype=float)
elif SELECTED_LIST == '100k':
    data = pd.read_csv(data100k_path, engine='pyarrow', dtype=float)
else:
    data = pd.read_csv(data10k_path, engine='pyarrow', dtype=float)

# split features (first 17 columns) and target (collision)
X = data.iloc[:, :17].values
y = data['collision'].values

# split the data 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# standardise to help network converge faster
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# check if the classes are balanced
print(data['collision'].value_counts())

# convert data into tensor pairs (feature + label)
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32), 
    torch.tensor(y_train, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32), 
    torch.tensor(y_test, dtype=torch.long)
)

# split dataset into batches of 128
training_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=128, shuffle=False) 

# nn definition
class SelfCollisionPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(17, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)

        # randomly set 20% inputs to 0 to prevent overfitting
        self.drop = nn.Dropout(0.2)
        # final layer has 2 outputs (collision or not)
        self.out = nn.Linear(128, 2)

    # data flow
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        x = F.relu(self.fc4(x))       


        return self.out(x)

# initialise model    
model = SelfCollisionPredictor()
# move to gpu if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# account for class imbalance
class_weights = torch.tensor([1.0, 1.5], device=device)

# calculate loss and apply the weight
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# adaptive learning rate
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.0
    last_loss = 0.0

    # iterate over batches, on gpu if possible
    for i, batch_data in enumerate(training_loader):
        inputs, labels = batch_data
        inputs, labels = inputs.to(device), labels.to(device)

        # clear old gradients
        optimiser.zero_grad()
        outputs = model(inputs)

        # compate logits with true labels
        loss = loss_fn(outputs, labels)

        # compute gradients
        loss.backward()

        # update parameters with optimiser
        optimiser.step()

        # every 1000 batches log
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print(f'  batch {i + 1} loss: {last_loss}')
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.0
            
    return last_loss

# create logs dir    
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/tiago_trainer_{timestamp}')

# completed epochs tracker
epoch_number = 0

# max epoch names
EPOCHS = 300

# how many rounds without improvement can happen
patience = 15
early_stop_counter = 0

# store lowest validation loss
best_vloss = 1_000_000.

# main training loop
for epoch in range(EPOCHS):
    print(f"epoch {epoch_number + 1}:")

    # set mode to training
    model.train(True)

    # call training, return avg loss for the epoch
    avg_loss = train_one_epoch(epoch_number, writer)
    running_vloss = 0.0

    # set mode to eval
    model.eval()

    # disable gradient tracking for validation to save memory and compute
    # calculate validation loss for each batch and accumulate it
    with torch.no_grad():
        for i, v_data in enumerate(validation_loader):
            vinputs, vlabels = v_data
            vinputs, vlabels = vinputs.to(device), vlabels.to(device) 

            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.item()

    # avg loss accross all batches
    avg_vloss = running_vloss / (i + 1)
    print(f'LOSS train {avg_loss} valid {avg_vloss}')

    # log
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # if validation imporved, save the model
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        early_stop_counter = 0
        best_model_path = f'models/best_model_{timestamp}.pt'
        torch.save(model.state_dict(), best_model_path)
    else:
        # if not, increase the counter
        early_stop_counter += 1
        print(f'Early stopping counter: {early_stop_counter} / {patience}')

    if early_stop_counter >= patience:
        # if hasn't improved for a long time, stop training
        print("Early stopping triggered. Training finished.")
        break

    epoch_number += 1


# assesment of the model

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# load the best model (the one with lowest validation loss)
model.load_state_dict(torch.load(f'models/best_model_{timestamp}.pt', map_location=device))
model.eval()

all_preds = []
all_labels = []
all_probs = []

start_time = time.perf_counter()

with torch.no_grad():
    for vinputs, vlabels in validation_loader:
        vinputs, vlabels = vinputs.to(device), vlabels.to(device)

        outputs = model(vinputs)

        # converts logits to probabiities
        probs = F.softmax(outputs, dim=1)

        # return class with highest logit
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(vlabels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

if device.type == 'cuda':
    torch.cuda.synchronize()
end_time = time.perf_counter()

# basic metrics
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

auc = roc_auc_score(all_labels, all_probs)
print(f"AUC-ROC  : {auc:.4f}")

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

# measure time
inference_time = end_time - start_time
total_samples = len(all_labels)
print(f"Total Inference Time: {inference_time:.4f} seconds")
print(f"Average Time per Sample: {(inference_time / total_samples) * 1000:.4f} ms")
print(f"Samples per Second: {total_samples / inference_time:.2f}")


# gradient calculations

# joint names in order (from the dataset)
JOINT_NAMES = [
    'arm_left_1_joint', 'arm_left_2_joint', 'arm_left_3_joint', 'arm_left_4_joint',
    'arm_left_5_joint', 'arm_left_6_joint', 'arm_left_7_joint', 'arm_right_1_joint',
    'arm_right_2_joint', 'arm_right_3_joint', 'arm_right_4_joint', 'arm_right_5_joint',
    'arm_right_6_joint', 'arm_right_7_joint', 'torso_lift_joint', 'head_1_joint', 'head_2_joint'
]

print("\n" + "="*60)
print("Gradient computation (∂collision_prob / ∂joint_angles) for all validation samples")
print("="*60)

# prepare tensors from the original validation data (already standardised)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

model.eval()

all_grad_norms = []
all_gradients = []
all_probs_grad = []
all_labels_grad = []

grad_start_time = time.perf_counter()

# iterate over each sample individually (no batching)
for i in range(len(X_test_tensor)):
    # create a fresh tensor for this sample with requires_grad=True
    x_i = X_test_tensor[i:i+1].clone().detach().requires_grad_(True)
    
    # forward pass
    logits = model(x_i)
    prob_collision = F.softmax(logits, dim=1)[0, 1] 

    # compute gradient of collision probability w.r.t. input joints
    grad = torch.autograd.grad(prob_collision, x_i, retain_graph=False)[0]  # shape [1,17]
    grad_np = grad.cpu().detach().numpy().flatten()
    
    all_grad_norms.append(np.linalg.norm(grad_np))
    all_gradients.append(grad_np)
    all_probs_grad.append(prob_collision.item())
    all_labels_grad.append(y_test[i].item())
    
    # print progress
    if (i + 1) % 10000 == 0:
        print(f"Processed {i+1} samples...")

if device.type == 'cuda':
    torch.cuda.synchronize()
grad_end_time = time.perf_counter()

grad_total_time = grad_end_time - grad_start_time
total_grad_samples = len(all_grad_norms)
print(f"\nGradient computation completed for {total_grad_samples} samples in {grad_total_time:.2f} seconds.")
print(f"Average time per sample for gradient: {(grad_total_time / total_grad_samples) * 1000:.4f} ms")

# stats
all_grad_norms = np.array(all_grad_norms)
all_gradients = np.array(all_gradients)
all_labels_grad = np.array(all_labels_grad)

print("\n--- Summary statistics of gradient norms ---")
print(f"Mean gradient norm: {np.mean(all_grad_norms):.6f}")
print(f"Std gradient norm:  {np.std(all_grad_norms):.6f}")
print(f"Min gradient norm:  {np.min(all_grad_norms):.6f}")
print(f"Max gradient norm:  {np.max(all_grad_norms):.6f}")
print(f"Median gradient norm: {np.median(all_grad_norms):.6f}")

# separate by true label
mask_collision = (all_labels_grad == 1)
mask_no_collision = (all_labels_grad == 0)

if np.any(mask_collision):
    print(f"\nFor collision samples (true label=1):")
    print(f"  Mean gradient norm: {np.mean(all_grad_norms[mask_collision]):.6f}")
    print(f"  Median: {np.median(all_grad_norms[mask_collision]):.6f}")
if np.any(mask_no_collision):
    print(f"\nFor non-collision samples (true label=0):")
    print(f"  Mean gradient norm: {np.mean(all_grad_norms[mask_no_collision]):.6f}")
    print(f"  Median: {np.median(all_grad_norms[mask_no_collision]):.6f}")

# average absolute gradient per joint (importance)
avg_abs_grad = np.mean(np.abs(all_gradients), axis=0)
print("\n--- Average absolute gradient per joint (importance) ---")
for j, name in enumerate(JOINT_NAMES):
    print(f"  {name:20s}: {avg_abs_grad[j]:.6f}")

# top 10 most sensitive configurations
top10_idx = np.argsort(all_grad_norms)[-10:][::-1]
print("\n--- Top 10 most sensitive configurations (highest gradient norm) ---")
for idx in top10_idx:
    print(f"Sample {idx}: true label={all_labels_grad[idx]}, prob={all_probs_grad[idx]:.6f}, grad_norm={all_grad_norms[idx]:.6f}")

# correlation between gradient norm and collision probability
corr = np.corrcoef(all_grad_norms, all_probs_grad)[0,1]
print(f"\nCorrelation between gradient norm and collision probability: {corr:.4f}")

print("\nGradient computation for all samples finished.")

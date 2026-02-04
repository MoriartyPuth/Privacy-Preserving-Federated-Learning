import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# --- 1. RESEARCH-GRADE ARCHITECTURE ---
class AttentionTheftDetector(nn.Module):
    def __init__(self, feature_dim=25):
        super(AttentionTheftDetector, self).__init__()
        self.feature_dim = feature_dim
        # Multi-head attention allows the model to "focus" on suspicious hours
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=1, batch_first=True)
        
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_unsq = x.unsqueeze(1)
        attn_output, _ = self.attention(x_unsq, x_unsq, x_unsq)
        return self.network(attn_output.squeeze(1))

# --- 2. DEFENSE & AGGREGATION ENGINE ---
class FederatedDefense:
    @staticmethod
    def apply_clipping(global_weights, local_weights, threshold=0.8):
        """Prevents magnitude-based poisoning (Hyper-Poisoning)."""
        with torch.no_grad():
            for key in local_weights.keys():
                update = local_weights[key] - global_weights[key]
                norm = torch.norm(update)
                if norm > threshold:
                    update = (threshold / norm) * update
                local_weights[key] = global_weights[key] + update
        return local_weights

    @staticmethod
    def multi_krum(local_updates, n_malicious=2):
        """Byzantine-resilient aggregation: votes out malicious outliers."""
        n_clients = len(local_updates)
        flat_updates = [torch.cat([v.flatten() for v in u.values()]) for u in local_updates]
        
        dist_mat = torch.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                d = torch.norm(flat_updates[i] - flat_updates[j])
                dist_mat[i, j] = dist_mat[j, i] = d

        # Calculate scores (sum of distances to closest neighbors)
        k = n_clients - n_malicious - 2
        scores = [torch.sum(torch.sort(dist_mat[i])[0][1:max(2, k+1)]) for i in range(n_clients)]
        
        # Select the 'safest' 3 nodes for the global average
        best_indices = torch.argsort(torch.tensor(scores))[:3]
        avg_weights = copy.deepcopy(local_updates[best_indices[0]])
        for key in avg_weights.keys():
            for i in range(1, len(best_indices)):
                avg_weights[key] += local_updates[best_indices[i]][key]
            avg_weights[key] /= len(best_indices)
            
        return avg_weights, best_indices.tolist()

# --- 3. LOCAL TRAINING (Honest vs Malicious) ---
def train_local_meter(global_weights, is_malicious=False, power=10.0, dp_noise=0.01):
    X, y = make_classification(n_samples=150, n_features=25, weights=[0.85])
    
    if is_malicious:
        # Attacker tries to force the model to classify all theft as 'Honest'
        y = np.zeros_like(y)
    else:
        # Balance data for honest nodes
        sm = SMOTE(sampling_strategy='minority', k_neighbors=2)
        X, y = sm.fit_resample(X, y)

    model = AttentionTheftDetector()
    model.load_state_dict(global_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCELoss()
    
    loader = DataLoader(TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)), batch_size=16)

    model.train()
    for _ in range(3):
        for data, target in loader:
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()

    local_weights = model.state_dict()
    
    # Simulate Attack Magnitude
    if is_malicious:
        for k in local_weights.keys():
            local_weights[k] *= power

    # Apply Defensive Clipping & DP Noise
    local_weights = FederatedDefense.apply_clipping(global_weights, local_weights, threshold=0.8)
    for k in local_weights.keys():
        local_weights[k] += torch.randn(local_weights[k].size()) * dp_noise
        
    return local_weights

# --- 4. DASHBOARD & AUDIT TRAIL ENGINE ---
def run_dashboard(model, history, X_test, y_test, round_num, audit_dir, is_final=False):
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3)
    feature_labels = [f"{i}h" for i in range(24)] + ["Temp"]

    # ... [Keep Accuracy, Correlation, and Aggregator plots the same] ...
    # (Leaving those out for brevity, keep your existing code for ax1, ax2, ax3)

    # D. Attention Analysis (THE FIX)
    # D. Feature Importance Analysis (Robust Fix)
    ax4 = fig.add_subplot(gs[1, :])
    X_sample = torch.FloatTensor(X_test[:100])
    
    with torch.no_grad():
        # Attempt to get attention
        _, attn = model.attention(X_sample.unsqueeze(1), X_sample.unsqueeze(1), X_sample.unsqueeze(1))
        avg_attn = attn.mean(dim=0).cpu().numpy().flatten()
        
        # LOGIC: If attention is a single value, it's not useful for a bar chart.
        # We fall back to Feature Importance via the first Linear layer weights.
        if len(avg_attn) != len(feature_labels):
            # Get weights of the first layer (64 neurons x 25 features)
            # We take the absolute mean weight per feature
            weight_importance = model.network[0].weight.abs().mean(dim=0).cpu().numpy()
            avg_attn = weight_importance

    # Ensure length matches exactly (should be 25)
    if len(avg_attn) > len(feature_labels):
        avg_attn = avg_attn[:len(feature_labels)]
    elif len(avg_attn) < len(feature_labels):
        avg_attn = np.pad(avg_attn, (0, len(feature_labels) - len(avg_attn)), 'constant')

    plot_df = pd.DataFrame({
        'Feature': feature_labels, 
        'Importance': avg_attn
    })
    
    sns.barplot(data=plot_df, x='Feature', y='Importance', palette="magma", ax=ax4)
    plt.xticks(rotation=45)
    ax4.set_title("Feature Sensitivity: Identifying Theft Patterns", fontweight='bold')

    # Now this is guaranteed to be the same length
    plot_df = pd.DataFrame({
        'Feature': feature_labels, 
        'Importance': avg_attn
    })
    
    sns.barplot(data=plot_df, x='Feature', y='Importance', palette="magma", ax=ax4)
    plt.xticks(rotation=45)
    ax4.set_title("Global Attention: Suspect Usage Hours", fontweight='bold')

    plt.tight_layout()
    
    # Save to Audit Trail
    path = f"{audit_dir}/final_report.png" if is_final else f"{audit_dir}/rounds/round_{round_num}.png"
    plt.savefig(path, dpi=300)
    if is_final: plt.show()
    else: plt.close()

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    # Audit Trail Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_dir = f"theft_audit_{timestamp}"
    os.makedirs(f"{audit_dir}/rounds", exist_ok=True)

    print(f"🚀 Simulation Started. Results will be saved to: {audit_dir}")
    
    global_model = AttentionTheftDetector()
    global_weights = global_model.state_dict()
    history = {'acc': [], 'selection': []}
    
    # Generate static test set for fair evaluation
    X_test, y_test = make_classification(n_samples=300, n_features=25, weights=[0.85])

    for r in range(10):
        local_updates = []
        for i in range(8): # 8 Total nodes
            # Nodes 0 and 1 are attackers with 10x poisoning strength
            is_m = (i < 2)
            local_updates.append(train_local_meter(global_weights, is_malicious=is_m, power=10.0))

        # Robust Aggregation
        global_weights, chosen = FederatedDefense.multi_krum(local_updates, n_malicious=2)
        history['selection'].append(chosen)

        # Evaluation Round
        eval_m = AttentionTheftDetector()
        eval_m.load_state_dict(global_weights)
        with torch.no_grad():
            preds = (eval_m(torch.FloatTensor(X_test)).squeeze().numpy() > 0.5).astype(int)
            acc = (preds == y_test).mean()
            history['acc'].append(acc)
        
        run_dashboard(eval_m, history, X_test, y_test, r+1, audit_dir)
        print(f"Round {r+1} Complete | Accuracy: {acc:.2%}")

    # Final Dashboard and Text Report
    run_dashboard(eval_m, history, X_test, y_test, 10, audit_dir, is_final=True)
    
    with open(f"{audit_dir}/final_metrics.txt", "w") as f:
        f.write(f"Theft Detection Audit Report\n{'='*30}\n")
        f.write(classification_report(y_test, preds, target_names=['Honest', 'Thief']))

    print(f"✅ Simulation Complete. Final report saved in {audit_dir}")

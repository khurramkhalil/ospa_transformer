import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from datasets import load_dataset # Hugging Face datasets
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import math
import time
import os
import json
from datetime import datetime, timedelta
from tqdm import tqdm
from transformers import AutoTokenizer # Hugging Face tokenizer
import psutil # For system memory
import gc # For garbage collection

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Model Hyperparameters
# VOCAB_SIZE will be determined by tokenizer
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 2 # Reduced for quicker ablation runs
HIDDEN_DIM = 512
K_LANDMARKS_DEFAULT = 64 # Default K, will be overridden in run_experiment
KMEANS_ITERS = 5 # Not directly used by MiniBatchKMeans constructor if n_init is set
DROPOUT = 0.1

# Training Hyperparameters
MAX_LEN = 128 # Reduced for quicker tokenization and smaller memory footprint in tests
BATCH_SIZE = 16 # Can be 16 or 32
LR = 1e-4
NUM_EPOCHS_MAIN = 5 # For a full run (not used in ablation script directly)
CLIP_GRAD = 1.0

# Logging Configuration
MODEL_BASE_NAME = "GPA_Ablation"
LOG_DIR = "new_logs_gpa_imdb_ablations"
os.makedirs(LOG_DIR, exist_ok=True)

# Global log_file variable, will be updated by run_experiment
LOG_FILE = ""
experiment_log = {} # Will be re-initialized by run_experiment


# Function to update the log file
def update_log_file():
    if LOG_FILE: # Ensure LOG_FILE is set
        with open(LOG_FILE, 'w') as f:
            json.dump(experiment_log, f, indent=2)

# --- GPA Implementation ---
class GeometricProgressiveAttention(nn.Module):
    def __init__(self, d_model, nhead, k_landmarks, kmeans_iters_unused, dropout=0.1,
                 landmark_selection_strategy="kmeans"):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.k_landmarks = k_landmarks
        self.landmark_selection_strategy = landmark_selection_strategy
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.l_q_proj = nn.Linear(d_model, d_model)
        self.l_k_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout) # Renamed to avoid conflict with self.dropout if it were a value

    def _select_landmarks_kmeans(self, X_queries_orig):
        batch_size, seq_len, d_model = X_queries_orig.shape
        all_centroids = []
        for i in range(batch_size):
            queries_np = X_queries_orig[i].detach().cpu().numpy()
            
            # Ensure k_landmarks is not greater than the number of unique samples if seq_len is small
            effective_k = self.k_landmarks
            if seq_len < self.k_landmarks:
                 # If all queries are identical (e.g. after padding), unique_queries_np might be tiny
                unique_queries_np, unique_indices = np.unique(queries_np, axis=0, return_index=True)
                num_unique = unique_queries_np.shape[0]
                if num_unique == 0: # All padding tokens, or empty sequence
                    centroids_np = np.zeros((self.k_landmarks, d_model), dtype=queries_np.dtype)
                    all_centroids.append(torch.from_numpy(centroids_np).float().to(X_queries_orig.device))
                    continue
                effective_k = min(self.k_landmarks, num_unique)
                if effective_k == 0: # Should not happen if num_unique > 0
                    effective_k = 1 # Ensure at least 1 cluster
                
                # Use only unique queries for k-means if seq_len is small
                queries_to_cluster = unique_queries_np
            else:
                queries_to_cluster = queries_np
                effective_k = self.k_landmarks

            if effective_k == 0 or queries_to_cluster.shape[0] == 0 : # Should be caught above
                centroids_np = np.zeros((self.k_landmarks, d_model), dtype=queries_np.dtype)
            elif queries_to_cluster.shape[0] < effective_k :
                # Not enough unique samples for k distinct clusters, use all unique samples as centroids
                # and pad if necessary
                centroids_np = queries_to_cluster
                if queries_to_cluster.shape[0] < self.k_landmarks:
                    padding_needed = self.k_landmarks - queries_to_cluster.shape[0]
                    pad_values = np.zeros((padding_needed, d_model), dtype=queries_np.dtype)
                    centroids_np = np.vstack((centroids_np, pad_values))
            else:
                current_kmeans = MiniBatchKMeans(n_clusters=effective_k,
                                                 random_state=i, # Batch-item specific random state
                                                 n_init='auto', # Requires scikit-learn >= 1.1.0, else use 3 or 10
                                                 batch_size=min(1024, queries_to_cluster.shape[0]))
                current_kmeans.fit(queries_to_cluster)
                centroids_np = current_kmeans.cluster_centers_
                # If k-means returned fewer than effective_k centroids (can happen with MiniBatchKMeans or sparse data)
                # or if effective_k was less than self.k_landmarks, pad to self.k_landmarks
                if centroids_np.shape[0] < self.k_landmarks:
                    padding_needed = self.k_landmarks - centroids_np.shape[0]
                    pad_values = np.zeros((padding_needed, d_model), dtype=queries_np.dtype)
                    centroids_np = np.vstack((centroids_np, pad_values))


            all_centroids.append(torch.from_numpy(centroids_np).float().to(X_queries_orig.device))
        return torch.stack(all_centroids)

    def _select_landmarks_random(self, X_queries_orig, key_padding_mask):
        batch_size, seq_len, d_model = X_queries_orig.shape
        all_random_landmarks = []
        for i in range(batch_size):
            current_queries = X_queries_orig[i]
            
            if key_padding_mask is not None:
                # Select from non-padded tokens
                non_padded_indices = (~key_padding_mask[i]).nonzero(as_tuple=True)[0]
                num_non_padded = len(non_padded_indices)
                if num_non_padded == 0: # All tokens are padded
                    selected_indices = torch.zeros(self.k_landmarks, dtype=torch.long, device=current_queries.device)
                elif num_non_padded < self.k_landmarks:
                    # Sample with replacement from non-padded tokens
                    permuted_indices = non_padded_indices[torch.randint(0, num_non_padded, (self.k_landmarks,), device=current_queries.device)]
                    selected_indices = permuted_indices
                else:
                    # Sample without replacement from non-padded tokens
                    permuted_indices = non_padded_indices[torch.randperm(num_non_padded, device=current_queries.device)]
                    selected_indices = permuted_indices[:self.k_landmarks]
            else: # No padding mask, assume all tokens are valid
                if seq_len == 0:
                    selected_indices = torch.zeros(self.k_landmarks, dtype=torch.long, device=current_queries.device) # Should not happen with actual data
                elif seq_len < self.k_landmarks:
                    selected_indices = torch.randint(0, seq_len, (self.k_landmarks,), device=current_queries.device)
                else:
                    selected_indices = torch.randperm(seq_len, device=current_queries.device)[:self.k_landmarks]
            
            landmarks_for_item = current_queries[selected_indices]
            all_random_landmarks.append(landmarks_for_item)
        return torch.stack(all_random_landmarks)


    def forward(self, x, key_padding_mask=None):
        batch_size, seq_len, d_model = x.shape
        q_orig = self.q_proj(x)
        k_orig = self.k_proj(x)
        v_orig = self.v_proj(x)

        C = None
        if self.landmark_selection_strategy == "kmeans":
            C = self._select_landmarks_kmeans(q_orig)
        elif self.landmark_selection_strategy == "random":
            C = self._select_landmarks_random(q_orig, key_padding_mask) # Pass padding mask
        else:
            raise ValueError(f"Unknown landmark selection strategy: {self.landmark_selection_strategy}")

        l_q = self.l_q_proj(C)
        l_k = self.l_k_proj(C)

        q = q_orig.reshape(batch_size, seq_len, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        k = k_orig.reshape(batch_size, seq_len, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        v = v_orig.reshape(batch_size, seq_len, self.nhead, self.head_dim).permute(0, 2, 1, 3)

        l_q = l_q.reshape(batch_size, self.k_landmarks, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        l_k = l_k.reshape(batch_size, self.k_landmarks, self.nhead, self.head_dim).permute(0, 2, 1, 3)

        attn_scores_lk = torch.matmul(l_q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            mask_lk = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores_lk = attn_scores_lk.masked_fill(mask_lk, float('-inf'))
        attn_weights_lk = torch.softmax(attn_scores_lk, dim=-1)
        attn_weights_lk = self.dropout_layer(attn_weights_lk)
        v_prime = torch.matmul(attn_weights_lk, v)

        attn_scores_ql = torch.matmul(q, l_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights_ql = torch.softmax(attn_scores_ql, dim=-1)
        attn_weights_ql = self.dropout_layer(attn_weights_ql)
        out = torch.matmul(attn_weights_ql, v_prime)

        out = out.permute(0, 2, 1, 3).contiguous().reshape(batch_size, seq_len, d_model)
        out = self.out_proj(out)
        return out

# --- Standard Transformer Components ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerGPALayer(nn.Module):
    def __init__(self, d_model, nhead, k_landmarks, kmeans_iters, dim_feedforward, dropout, landmark_selection_strategy):
        super().__init__()
        self.self_attn = GeometricProgressiveAttention(d_model, nhead, k_landmarks, kmeans_iters, dropout, landmark_selection_strategy)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_ff = nn.Dropout(dropout) # Renamed to avoid confusion
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_key_padding_mask=None):
        src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout_ff(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# --- Full Model ---
class TransformerWithGPA(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, hidden_dim,
                 k_landmarks, kmeans_iters, dropout, max_len, landmark_selection_strategy="kmeans"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=tokenizer.pad_token_id if tokenizer else 0) # Use tokenizer.pad_token_id
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        gpa_layers = [
            TransformerGPALayer(d_model, nhead, k_landmarks, kmeans_iters, hidden_dim, dropout, landmark_selection_strategy)
            for _ in range(num_layers)
        ]
        self.transformer_encoder = nn.ModuleList(gpa_layers)
        self.d_model = d_model
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, src, src_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        for layer in self.transformer_encoder:
            src = layer(src, src_key_padding_mask=src_padding_mask)
        if src_padding_mask is not None:
            non_pad_mask = (~src_padding_mask).float().unsqueeze(-1)
            # Ensure non_pad_mask_sum is not zero to avoid division by zero
            non_pad_mask_sum = non_pad_mask.sum(dim=1).clamp(min=1e-9)
            pooled_output = (src * non_pad_mask).sum(dim=1) / non_pad_mask_sum
        else:
            pooled_output = src.mean(dim=1)
        output_logits = self.classifier(pooled_output)
        return output_logits.squeeze(-1)

# --- Data Preparation ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.model_max_length = MAX_LEN # Ensure tokenizer respects MAX_LEN
if tokenizer.pad_token is None:
    print("Tokenizer does not have a pad token, setting to eos_token.")
    tokenizer.pad_token = tokenizer.eos_token
PAD_IDX = tokenizer.pad_token_id

class IMDbDataset(Dataset):
    def __init__(self, hf_dataset): # Pass the loaded Hugging Face dataset
        self.dataset = hf_dataset
        self.tokenized_data = []
        print(f"Tokenizing dataset partition...")
        for item in tqdm(self.dataset):
            label = 1 if item['label'] == 1 else 0
            # Tokenize with padding to MAX_LEN and truncation
            encoded = tokenizer(
                item['text'],
                truncation=True,
                padding='max_length', # Pad to MAX_LEN
                max_length=MAX_LEN,
                return_tensors=None # Get lists of ids
            )
            self.tokenized_data.append({
                'label': label,
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask']
            })
    def __len__(self):
        return len(self.tokenized_data)
    def __getitem__(self, idx):
        item = self.tokenized_data[idx]
        return (
            item['label'],
            torch.tensor(item['input_ids'], dtype=torch.long),
            torch.tensor(item['attention_mask'], dtype=torch.long) # 1 for real tokens, 0 for padding
        )

def collate_batch_hf(batch):
    label_list, input_ids_list, attention_mask_list = [], [], []
    for _label, _input_ids, _attention_mask in batch:
        label_list.append(_label)
        input_ids_list.append(_input_ids)
        attention_mask_list.append(_attention_mask)

    labels = torch.tensor(label_list, dtype=torch.float32)
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    
    # src_padding_mask should be True for padded tokens
    src_padding_mask = (attention_mask == 0)

    return labels.to(DEVICE), input_ids.to(DEVICE), src_padding_mask.to(DEVICE)


# --- Training and Evaluation Functions (mostly same, ensure logging works with global experiment_log) ---
def train_epoch(model, dataloader, optimizer, criterion, clip_val, epoch_num, config_name):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    num_batches = len(dataloader)
    batch_losses, batch_accuracies, batch_times = [], [], []
    
    epoch_start_time = time.time()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(DEVICE)
    start_memory_epoch = psutil.virtual_memory().used / (1024 * 1024) if not torch.cuda.is_available() else torch.cuda.memory_allocated(DEVICE) / (1024 * 1024)


    pbar = tqdm(dataloader, desc=f"Train E{epoch_num+1} ({config_name})", leave=False)
    for batch_idx, (labels, texts, padding_mask) in enumerate(pbar):
        batch_start = time.time()
        optimizer.zero_grad()
        predictions = model(texts, src_padding_mask=padding_mask)
        loss = criterion(predictions, labels)
        predicted_labels = torch.round(torch.sigmoid(predictions))
        correct = (predicted_labels == labels).float()
        acc = correct.sum() / len(correct)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        batch_losses.append(loss.item()); batch_accuracies.append(acc.item()); batch_times.append(time.time() - batch_start)
        pbar.set_postfix(loss=loss.item(), acc=acc.item())

    avg_loss = epoch_loss / num_batches
    avg_acc = epoch_acc / num_batches
    epoch_duration = time.time() - epoch_start_time
    
    # Update epoch log (ensure correct epoch_data is found or created)
    current_epoch_data = None
    for i, ed in enumerate(experiment_log["epochs"]):
        if ed["epoch"] == epoch_num + 1:
            current_epoch_data = experiment_log["epochs"][i]
            break
    if current_epoch_data is None:
        current_epoch_data = {"epoch": epoch_num + 1, "batch_logs": []} # batch_logs can be added if needed
        experiment_log["epochs"].append(current_epoch_data)
    
    current_epoch_data.update({
        "train_loss": avg_loss, "train_accuracy": avg_acc, "duration_seconds": epoch_duration,
        "samples_per_second": len(dataloader.dataset) / epoch_duration if epoch_duration > 0 else 0,
        "avg_batch_time": np.mean(batch_times) if batch_times else 0,
    })
    if torch.cuda.is_available():
         current_epoch_data["memory_stats_epoch_end_gpu_mb"] = torch.cuda.memory_allocated(DEVICE) / (1024 * 1024)
         current_epoch_data["peak_memory_epoch_gpu_mb"] = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)
    current_epoch_data["memory_stats_epoch_end_sys_mb"] = psutil.virtual_memory().used / (1024 * 1024)

    update_log_file()
    gc.collect(); 
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return avg_loss, avg_acc


def evaluate_epoch(model, dataloader, criterion, split_name="val", epoch_num=None, config_name=""):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    num_batches = len(dataloader)
    all_predictions_probs, all_labels = [], []
    
    epoch_start_time = time.time()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(DEVICE)

    pbar = tqdm(dataloader, desc=f"Eval {split_name} E{epoch_num+1 if epoch_num is not None else 'Final'} ({config_name})", leave=False)
    with torch.no_grad():
        for labels, texts, padding_mask in pbar:
            predictions = model(texts, src_padding_mask=padding_mask)
            loss = criterion(predictions, labels)
            predicted_probs = torch.sigmoid(predictions)
            predicted_labels = torch.round(predicted_probs)
            correct = (predicted_labels == labels).float()
            acc = correct.sum() / len(correct)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            all_predictions_probs.extend(predicted_probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            pbar.set_postfix(loss=loss.item(), acc=acc.item())

    avg_loss = epoch_loss / num_batches
    avg_acc = epoch_acc / num_batches # This is batch-averaged accuracy
    epoch_duration = time.time() - epoch_start_time

    # Calculate exact accuracy from all_labels and all_predictions_probs
    overall_acc = sum(1 for i, p_val in enumerate(all_predictions_probs) if (p_val >= 0.5 and all_labels[i] == 1) or (p_val < 0.5 and all_labels[i] == 0)) / len(all_labels) if all_labels else 0


    eval_log_data = {
        f"{split_name}_loss": avg_loss, f"{split_name}_accuracy_avg_batch": avg_acc, f"{split_name}_accuracy_overall": overall_acc,
        "duration_seconds": epoch_duration,
        "samples_per_second": len(dataloader.dataset) / epoch_duration if epoch_duration > 0 else 0,
    }
    if torch.cuda.is_available():
        eval_log_data["peak_memory_eval_gpu_mb"] = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)


    # Update experiment_log
    if epoch_num is not None: # Part of epoch training
        current_epoch_data = None
        for i, ed in enumerate(experiment_log["epochs"]):
            if ed["epoch"] == epoch_num + 1:
                current_epoch_data = experiment_log["epochs"][i]
                break
        if current_epoch_data is not None:
             if f"{split_name}_evaluation" not in current_epoch_data: current_epoch_data[f"{split_name}_evaluation"] = {}
             current_epoch_data[f"{split_name}_evaluation"].update(eval_log_data)
        else: # Should not happen if train_epoch created the entry
            experiment_log["epochs"].append({"epoch": epoch_num + 1, f"{split_name}_evaluation": eval_log_data})
    else: # Final evaluation
        if "final_evaluation" not in experiment_log: experiment_log["final_evaluation"] = {}
        experiment_log["final_evaluation"][split_name] = eval_log_data
    
    update_log_file()
    gc.collect(); 
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return avg_loss, overall_acc # Return overall accuracy


# --- Experiment Runner ---
def run_experiment(config_name, landmark_strategy, k_landmarks_override=None, num_train_samples=256, num_test_samples=128, num_ablation_epochs=2):
    global LOG_FILE, experiment_log, K_LANDMARKS_DEFAULT

    current_k_landmarks = k_landmarks_override if k_landmarks_override is not None else K_LANDMARKS_DEFAULT
    
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    LOG_FILE = os.path.join(LOG_DIR, f"{MODEL_BASE_NAME}_{config_name}_{run_timestamp}.json")
    
    experiment_log = {
        "model_base_name": MODEL_BASE_NAME,
        "config_name": config_name,
        "run_timestamp": run_timestamp,
        "landmark_strategy": landmark_strategy,
        "hyperparameters_run": {
            "embed_dim": EMBED_DIM, "num_heads": NUM_HEADS, "num_layers": NUM_LAYERS,
            "hidden_dim": HIDDEN_DIM, "k_landmarks": current_k_landmarks,
            "kmeans_iters": KMEANS_ITERS, "dropout": DROPOUT, "max_len": MAX_LEN,
            "batch_size": BATCH_SIZE, "learning_rate": LR, "num_epochs": num_ablation_epochs,
            "clip_grad": CLIP_GRAD, "num_train_samples": num_train_samples, "num_test_samples": num_test_samples
        },
        "hardware": {
            "device": str(DEVICE), "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cpu_count": os.cpu_count(),
            "total_system_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        },
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": [],
        "training_summary": {},
        "final_evaluation": {}
    }
    update_log_file()
    print(f"\n--- Running Experiment: {config_name} ({landmark_strategy}, k={current_k_landmarks}) ---")
    print(f"Logging to: {LOG_FILE}")

    # Load full datasets
    full_imdb_train = load_dataset("imdb", split='train')
    full_imdb_test = load_dataset("imdb", split='test')

    # Create subsets
    train_subset_hf = full_imdb_train.select(range(num_train_samples))
    test_subset_hf = full_imdb_test.select(range(num_test_samples))
    
    train_dataset = IMDbDataset(train_subset_hf)
    test_dataset = IMDbDataset(test_subset_hf) # Using test as validation for quick runs

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch_hf)
    val_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch_hf)

    vocab_size = tokenizer.vocab_size # From global tokenizer
    experiment_log["dataset_info"] = {"name": "IMDb_subset", "vocab_size": vocab_size, 
                                      "train_samples_used": num_train_samples, "test_samples_used": num_test_samples,
                                      "max_seq_length_effective": MAX_LEN}
    update_log_file()

    model = TransformerWithGPA(
        vocab_size=vocab_size, d_model=EMBED_DIM, nhead=NUM_HEADS, num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM, k_landmarks=current_k_landmarks,
        kmeans_iters=KMEANS_ITERS, dropout=DROPOUT, max_len=MAX_LEN,
        landmark_selection_strategy=landmark_strategy
    ).to(DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    experiment_log["model_params"] = num_params
    print(f"Model Parameters ({config_name}): {num_params:,}")
    update_log_file()

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    
    overall_train_start_time = time.time()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(DEVICE)
    experiment_log["initial_gpu_memory_mb"] = torch.cuda.memory_allocated(DEVICE) / (1024*1024) if torch.cuda.is_available() else 0
    experiment_log["initial_sys_memory_mb"] = psutil.virtual_memory().used / (1024 * 1024)
    update_log_file()

    best_val_loss_run = float('inf')

    for epoch in range(num_ablation_epochs):
        print(f"\nEpoch {epoch+1}/{num_ablation_epochs}")
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, criterion, CLIP_GRAD, epoch, config_name)
        val_loss, val_acc = evaluate_epoch(model, val_dataloader, criterion, "val", epoch, config_name)
        print(f"  Config: {config_name} | Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")

        if val_loss < best_val_loss_run:
            best_val_loss_run = val_loss
            # torch.save(model.state_dict(), f'{MODEL_BASE_NAME}_{config_name}_best.pt') # Optional save
            current_epoch_data = next(item for item in experiment_log["epochs"] if item["epoch"] == epoch + 1)
            current_epoch_data["is_best_val_loss_so_far"] = True
            update_log_file()


    overall_train_duration = time.time() - overall_train_start_time
    experiment_log["training_summary"] = {
        "total_duration_seconds": overall_train_duration,
        "avg_epoch_time": overall_train_duration / num_ablation_epochs if num_ablation_epochs > 0 else 0
    }
    if torch.cuda.is_available():
        experiment_log["training_summary"]["peak_gpu_memory_train_mb"] = torch.cuda.max_memory_allocated(DEVICE) / (1024*1024)
    experiment_log["training_summary"]["final_sys_memory_mb"] = psutil.virtual_memory().used / (1024 * 1024)


    # Final Test (using the same val_dataloader for this quick test)
    final_test_loss, final_test_acc = evaluate_epoch(model, val_dataloader, criterion, "final_test", epoch_num=None, config_name=config_name) # epoch_num=None for final
    print(f"  Config: {config_name} | Final Test Loss: {final_test_loss:.4f}, Acc: {final_test_acc*100:.2f}%")

    experiment_log["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment_log["total_script_duration_seconds"] = time.time() - datetime.strptime(experiment_log["start_time"], "%Y-%m-%d %H:%M:%S").timestamp()
    update_log_file()
    print(f"--- Experiment {config_name} Finished. Log: {LOG_FILE} ---")
    return LOG_FILE


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure tokenizer is globally accessible for model's padding_idx
    # It's already defined globally above the classes

    # Ablation Scenario Parameters
    NUM_TRAIN_SAMPLES_ABLATION = 2000 # Keep small for quick runs
    NUM_TEST_SAMPLES_ABLATION = 1000
    NUM_EPOCHS_ABLATION = 2 # Very few epochs for quick check

    all_log_files = []
    try:
        log_A = run_experiment(
            config_name="GPA_KMeans_k64", 
            landmark_strategy="kmeans", 
            k_landmarks_override=64,
            num_train_samples=NUM_TRAIN_SAMPLES_ABLATION,
            num_test_samples=NUM_TEST_SAMPLES_ABLATION,
            num_ablation_epochs=NUM_EPOCHS_ABLATION
        )
        all_log_files.append(log_A)

        log_B = run_experiment(
            config_name="GPA_Random_k64", 
            landmark_strategy="random", 
            k_landmarks_override=64,
            num_train_samples=NUM_TRAIN_SAMPLES_ABLATION,
            num_test_samples=NUM_TEST_SAMPLES_ABLATION,
            num_ablation_epochs=NUM_EPOCHS_ABLATION
        )
        all_log_files.append(log_B)

        log_C = run_experiment(
            config_name="GPA_KMeans_k8", 
            landmark_strategy="kmeans", 
            k_landmarks_override=16,
            num_train_samples=NUM_TRAIN_SAMPLES_ABLATION,
            num_test_samples=NUM_TEST_SAMPLES_ABLATION,
            num_ablation_epochs=NUM_EPOCHS_ABLATION
        )
        all_log_files.append(log_C)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        # If experiment_log was initialized for the failing run, log the error
        if experiment_log and LOG_FILE: # Check if LOG_FILE is set
            experiment_log["error_details"] = {"error": str(e), "traceback": traceback.format_exc()}
            update_log_file()
    finally:
        print("\nCompleted all ablation runs.")
        if all_log_files:
            print("Log files created:")
            for lf in all_log_files:
                print(f"- {lf}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
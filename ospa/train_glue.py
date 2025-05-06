# train_ospa.py
import os
import json
import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Assuming 'datasets' library is installed: pip install datasets
try:
    from datasets import load_dataset
except ImportError:
    print("Please install the 'datasets' library: pip install datasets")
    exit(1)

# Assuming these files exist in the same directory or are importable
try:
    from improved_transformer_model import TransformerModel
except ImportError:
    print("Error: improved_transformer_model.py not found.")
    print("Make sure it's in the same directory or your Python path.")
    exit(1)

# --- Data Loading Functions ---
def get_language_modeling_data(args):
    """Prepare data for language modeling task (WikiText-2) using datasets library."""
    from datasets import load_dataset
    from tqdm import tqdm
    import torch

    print("Loading WikiText-2 dataset...")
    try:
        wikitext = load_dataset("wikitext", "wikitext-2-v1")
    except Exception as e:
        print(f"Error loading WikiText-2 dataset: {e}")
        print("Please ensure you have an internet connection and the 'datasets' library is up to date.")
        exit(1)

    # Tokenizer: simple whitespace split (Consider replacing with a better tokenizer later)
    def tokenize(text):
        return text.split()

    # Special tokens and initial vocab
    # <pad> might not be strictly needed for pure LM if not padding batches, but good practice.
    special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    word_count = {}

    # Count token frequencies from training set
    print("Building vocabulary from training data...")
    for text in tqdm(wikitext['train']['text'], desc="Counting Tokens"):
        if text.strip():
            for token in tokenize(text):
                # Ensure token is not empty string
                if token and token not in special_tokens:
                    word_count[token] = word_count.get(token, 0) + 1

    # Add words to vocab based on frequency and cutoff
    print("Constructing final vocabulary...")
    if args.vocab_cutoff > 0:
        print(f"Limiting vocabulary to top {args.vocab_cutoff} tokens (including special tokens)")
        # Ensure special tokens are prioritized
        sorted_tokens = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        for token, _ in sorted_tokens:
             if len(vocab) >= args.vocab_cutoff:
                 break
             if token not in vocab: # Avoid adding duplicates or overwriting special tokens
                vocab[token] = len(vocab)
    else:
        # Add all words if no cutoff
        print("Using full vocabulary (no cutoff).")
        for token in word_count:
            if token not in vocab:
                 vocab[token] = len(vocab)

    unk_idx = vocab.get('<unk>')
    if unk_idx is None or unk_idx != 0:
         # This should not happen if <unk> is in special_tokens[0]
         print("Error: '<unk>' token not found at index 0 in vocabulary!")
         # Fallback: Force <unk> at index 0 if needed, potentially shifting others
         # For simplicity, we assert based on initial setup.
         assert False, "Special token <unk> must be at index 0"

    print(f"Final vocabulary size: {len(vocab)}")

    # Convert text to token IDs
    def data_process(text_iter, split_name=""):
        data = []
        print(f"Tokenizing {split_name} data...")
        for text in tqdm(text_iter, desc=f"Tokenizing {split_name}"):
            if text.strip():
                # Map tokens to IDs, using unk_idx for unknowns
                tokens = [vocab.get(token, unk_idx) for token in tokenize(text) if token] # Handle empty tokens
                if tokens:
                    # Example: Add BOS/EOS if desired:
                    # data.append(torch.tensor([vocab['<bos>']] + tokens + [vocab['<eos>']], dtype=torch.long))
                    data.append(torch.tensor(tokens, dtype=torch.long))
        # Concatenate all tensors into one long sequence
        return torch.cat(data) if data else torch.tensor([], dtype=torch.long)

    print("Processing datasets...")
    train_data = data_process(wikitext['train']['text'], "train")
    val_data = data_process(wikitext['validation']['text'], "validation")
    test_data = data_process(wikitext['test']['text'], "test")

    # Reshape data into batches [SeqLen, BatchSize]
    def batchify(data, batch_size, device):
        if data.numel() == 0:
             print("Warning: Data is empty after processing.")
             # Return an empty tensor with shape [0, batch_size]
             return torch.empty((0, batch_size), dtype=torch.long, device=device)
        num_tokens = data.size(0)
        # Calculate number of full batches
        num_batches = num_tokens // batch_size
        # Trim off remainder tokens that don't fit full batches
        data = data.narrow(0, 0, num_batches * batch_size)
        # Reshape: [SeqLen * BatchSize] -> [BatchSize, SeqLen] -> [SeqLen, BatchSize]
        data = data.view(batch_size, -1).t().contiguous()
        print(f"Batchified data shape: {data.shape}")
        return data.to(device)

    print("Batchifying...")
    train_data = batchify(train_data, args.batch_size, args.device)
    val_data = batchify(val_data, args.batch_size, args.device)
    test_data = batchify(test_data, args.batch_size, args.device)

    # --- Function to get a batch slice for LM training ---
    # This function will be passed to the training/eval loops
    def get_batch_lm(source, i, bptt):
        """
        Args:
            source: Tensor, shape [full_seq_len, batch_size]
            i: int, starting index
            bptt: int, sequence length
        Returns:
            tuple: (data, target) tensors
        """
        # Calculate sequence length, ensuring it doesn't exceed source boundaries
        seq_len = min(bptt, len(source) - 1 - i)
        if seq_len <= 0: # Handle edge case where i is too close to the end
            return torch.empty((0, source.size(1)), dtype=torch.long, device=source.device), \
                   torch.empty((0,), dtype=torch.long, device=source.device)

        data = source[i : i + seq_len]                 # Input sequence [seq_len, batch_size]
        target = source[i + 1 : i + 1 + seq_len]     # Target sequence [seq_len, batch_size]
        # Target needs to be reshaped for CrossEntropyLoss: [seq_len * batch_size]
        return data, target.reshape(-1)

    return train_data, val_data, test_data, vocab, get_batch_lm # Return the batching function


def get_classification_data(args):
    """Prepare data for text classification task (IMDB) using datasets library."""
    import torch
    from torch.utils.data import DataLoader

    print("Loading IMDB dataset...")
    try:
        imdb = load_dataset("imdb")
    except Exception as e:
        print(f"Error loading IMDB dataset: {e}")
        exit(1)

    # Simple tokenizer function (split by whitespace)
    def tokenize(text):
        return text.split()

    # Build vocabulary from tokens
    vocab = {}
    word_count = {}

    # Add special tokens - IMPORTANT: Ensure PAD index is consistent if used by loss
    special_tokens = ['<unk>', '<pad>'] # UNK=0, PAD=1
    for i, token in enumerate(special_tokens):
        vocab[token] = i
    unk_idx = vocab['<unk>']
    pad_idx = vocab['<pad>']

    # Process training tokens and build vocabulary
    print("Building vocabulary from training data...")
    # Consider increasing num_build_samples or processing all data if feasible
    num_build_samples = 10000
    for i, example in enumerate(tqdm(imdb['train'], desc="Counting Tokens")):
        if i >= num_build_samples:
            break
        text = example['text']
        for token in tokenize(text):
            if token: # Avoid empty strings
                 word_count[token] = word_count.get(token, 0) + 1

    # Add words to vocab after counting all frequencies
    print("Constructing final vocabulary...")
    for token in word_count:
        if token not in vocab: # Avoid overwriting special tokens
            vocab[token] = len(vocab)

    # Optionally limit vocabulary size
    if args.vocab_cutoff > 0 and len(vocab) > args.vocab_cutoff:
        print(f"Limiting vocabulary from {len(vocab)} to {args.vocab_cutoff} tokens")
        # Sort by frequency (descending)
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        # Rebuild vocab, keeping special tokens
        new_vocab = {token: i for i, token in enumerate(special_tokens)}
        for token, _ in sorted_words:
             if len(new_vocab) >= args.vocab_cutoff:
                 break
             if token not in new_vocab: # Add if not already special and space permits
                new_vocab[token] = len(new_vocab)
        vocab = new_vocab
        # Re-assign pad/unk indices after potential cutoff
        unk_idx = vocab['<unk>']
        pad_idx = vocab['<pad>']

    print(f"Final vocabulary size: {len(vocab)}")
    print(f"PAD index: {pad_idx}, UNK index: {unk_idx}")

    # Define collate function for DataLoader
    def collate_batch(batch):
        label_list, text_list = [], []
        max_len = args.max_seq_len # Use arg for max length

        for example in batch:
            label_list.append(1 if example['label'] == 1 else 0)
            # Tokenize and map to IDs, handle unknown tokens
            processed_text = [vocab.get(token, unk_idx) for token in tokenize(example['text']) if token]
            processed_text_tensor = torch.tensor(processed_text, dtype=torch.long)

            # Truncate or pad to fixed length
            current_len = len(processed_text_tensor)
            if current_len > max_len:
                processed_text_tensor = processed_text_tensor[:max_len]
            elif current_len < max_len:
                # Pad with pad_idx
                padding = torch.full((max_len - current_len,), pad_idx, dtype=torch.long)
                processed_text_tensor = torch.cat([processed_text_tensor, padding])
            # Else: current_len == max_len, do nothing

            text_list.append(processed_text_tensor)

        # Convert lists to tensors
        label_tensor = torch.tensor(label_list, dtype=torch.long)
        # Stack tensors along batch dimension FIRST [batch_size, seq_len]
        text_tensor = torch.stack(text_list, dim=0)
        # Transpose for Transformer convention [seq_len, batch_size]
        return text_tensor.t().contiguous(), label_tensor

    # Create DataLoaders
    print("Creating dataloaders...")
    # Pin memory helps speed up CPU->GPU transfer if using CUDA
    pin_memory_flag = (args.device != 'cpu')

    train_dataloader = DataLoader(
        imdb['train'],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=pin_memory_flag
    )
    # Use test set for validation in this setup
    test_dataloader = DataLoader(
        imdb['test'],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=pin_memory_flag
    )

    return train_dataloader, test_dataloader, vocab


# --- Training and Evaluation Functions ---

def train_language_model(model, train_data, optimizer, criterion, scheduler, args, epoch, get_batch_func):
    """Train a language model for one epoch with causal masking."""
    model.train() # Set model to training mode
    total_loss = 0.0
    log_loss = 0.0 # Loss accumulator for logging interval
    start_time = time.time()
    ntokens = model.vocab_size # Get vocab size from the model attribute

    # Calculate total number of batches for tqdm
    num_batches = math.ceil((train_data.size(0) - 1) / args.bptt) if train_data.numel() > 0 else 0
    pbar = tqdm(range(0, train_data.size(0) - 1, args.bptt),
                total=num_batches, desc=f"Epoch {epoch}/{args.epochs} Training", leave=False)

    for batch_idx, data_start_index in enumerate(pbar):
        # Fetch batch data and targets using the provided function
        data, targets = get_batch_func(train_data, data_start_index, args.bptt)
        seq_len = data.size(0)
        if seq_len == 0: continue # Skip empty batches

        # --- Gradient Accumulation: Zero Grads ---
        # Clear gradients at the start of each accumulation cycle
        if batch_idx % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True) # More memory efficient

        try:
            # --- Generate Causal Mask ---
            # Use the built-in PyTorch function (requires torch >= 1.9)
            src_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=args.device)

            # --- Forward Pass ---
            output = model(data, src_mask=src_mask) # Pass the generated mask

            # --- Check for NaNs/Infs ---
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"\nWARNING: NaN or Inf detected in model output (Epoch {epoch}, Batch {batch_idx}). Skipping batch.")
                # Consider logging this occurrence more formally
                continue # Skip backprop for this batch

            # --- Loss Calculation ---
            # Reshape output to [SeqLen * BatchSize, VocabSize] for CrossEntropyLoss
            loss = criterion(output.view(-1, ntokens), targets)

            # --- Add Orthogonality Penalty (if applicable) ---
            if model.transformer_type == "ospa" and model.orth_mode == "regularize":
                orth_penalty = model.get_orthogonality_penalty()
                # Check penalty value before adding
                if not (torch.isnan(orth_penalty).any() or torch.isinf(orth_penalty).any()):
                     loss = loss + args.orth_penalty_weight * orth_penalty
                else:
                     print(f"\nWARNING: NaN/Inf in orthogonality penalty (Epoch {epoch}, Batch {batch_idx}). Penalty not added.")

            # --- Scale Loss for Gradient Accumulation ---
            if args.gradient_accumulation_steps > 1:
                # Normalize loss contribution across accumulation steps
                loss = loss / args.gradient_accumulation_steps

            # --- Backward Pass ---
            loss.backward()

            # --- Optimizer Step (after accumulation) ---
            is_accumulation_step = (batch_idx + 1) % args.gradient_accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == num_batches
            if is_accumulation_step or is_last_batch:
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                # Perform optimizer step
                optimizer.step()
                # Step the scheduler IF configured to update per step
                if args.scheduler_update_every_step:
                     if scheduler and hasattr(scheduler, 'step'):
                         scheduler.step()

            # --- Logging ---
            current_unscaled_loss = loss.item() * args.gradient_accumulation_steps # Get loss before scaling
            total_loss += current_unscaled_loss
            log_loss += current_unscaled_loss

            # Log progress periodically
            if (batch_idx + 1) % args.log_interval == 0 and batch_idx > 0:
                lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
                ms_per_batch = (time.time() - start_time) * 1000 / args.log_interval
                cur_avg_loss = log_loss / args.log_interval # Average loss over the interval
                try:
                    ppl = math.exp(min(cur_avg_loss, 700)) # Cap exponent for display stability
                except OverflowError:
                    ppl = float('inf')
                pbar.set_postfix({
                    'loss': f"{cur_avg_loss:.3f}", # More precision for loss
                    'ppl': f"{ppl:.2f}",
                    'lr': f"{lr:.6f}"  # More precision for LR
                })
                log_loss = 0.0 # Reset interval loss accumulator
                start_time = time.time()

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"\nERROR: CUDA OOM encountered in training (Epoch {epoch}, Batch {batch_idx}).")
                print("Try reducing --batch_size or --d_model/--nlayers.")
                # Consider how to handle OOM: exit, skip, try recovery
                raise e # Re-raise to stop training, or implement recovery
            else:
                print(f"\nERROR: Unexpected runtime error during training (Epoch {epoch}, Batch {batch_idx}): {e}")
                # Skip batch and try to continue, zero grads first
                optimizer.zero_grad(set_to_none=True)
                continue

    # Return average loss over the epoch (could be calculated more precisely if needed)
    # Return None for now as validation loss is the main metric tracked
    return None


def evaluate_language_model(model, eval_data, vocab, get_batch_func, criterion, args, eval_type="Validation"):
    """Evaluate the language model using causal mask."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    ntokens = model.vocab_size
    num_evaluated_tokens = 0 # Track number of tokens evaluated for accurate averaging

    with torch.no_grad(): # Disable gradient calculations for evaluation
        pbar_desc = f"{eval_type} Evaluating"
        # Iterate through evaluation data
        for i in tqdm(range(0, eval_data.size(0) - 1, args.bptt), desc=pbar_desc, leave=False):
            data_batch, targets = get_batch_func(eval_data, i, args.bptt)
            seq_len = data_batch.size(0)
            if seq_len == 0: continue # Skip if batch is empty

            try:
                # --- Generate Causal Mask ---
                src_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=args.device)

                # --- Forward Pass ---
                output = model(data_batch, src_mask=src_mask) # Pass generated mask

                # --- Check for NaNs/Infs ---
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"\nWARNING: NaN/Inf detected in {eval_type} output. Skipping batch.")
                    continue

                # --- Loss Calculation ---
                # Calculate loss for the batch, output shape [SeqLen*BatchSize, VocabSize]
                loss = criterion(output.view(-1, ntokens), targets)
                # Accumulate total loss weighted by the number of target tokens in the batch
                total_loss += loss.item() * targets.size(0)
                num_evaluated_tokens += targets.size(0)

            except RuntimeError as e:
                print(f"\nERROR: {eval_type} evaluation error: {e}")
                # Skip problematic batches during evaluation
                continue

    if num_evaluated_tokens == 0:
        print(f"\nWarning: No tokens were successfully evaluated during {eval_type}.")
        return float('inf') # Return infinity if no tokens were evaluated

    # Calculate average loss over all evaluated tokens
    average_loss = total_loss / num_evaluated_tokens
    return average_loss


def train_classifier(model, train_dataloader, vocab, optimizer, criterion, scheduler, args, epoch):
    """Train a text classifier for one epoch with padding mask."""
    model.train() # Set model to training mode
    total_loss = 0.0
    log_loss = 0.0 # Loss accumulator for logging interval
    correct = 0
    total = 0
    start_time = time.time()

    # Get padding index from vocabulary, provide a default if '<pad>' is missing
    pad_idx = vocab.get('<pad>', -1) # Use -1 or another invalid index if pad isn't expected
    if pad_idx == -1:
        print("Warning: '<pad>' token not found in vocabulary. Padding mask may not work correctly.")

    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs} Training", leave=False)
    for batch_idx, (data, target) in enumerate(pbar):
        # data shape: [seq_len, batch_size]
        data, target = data.to(args.device), target.to(args.device)

        # --- Gradient Accumulation: Zero Grads ---
        if batch_idx % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        try:
            # --- Generate Padding Mask ---
            # Mask should be True where values ARE padded (i.e., == pad_idx).
            # Expected shape by MultiheadAttention: [batch_size, seq_len]
            src_key_padding_mask = (data == pad_idx).transpose(0, 1) if pad_idx != -1 else None
            # -------------------------------

            # --- Forward Pass ---
            # Pass the generated padding mask
            output = model(data, src_key_padding_mask=src_key_padding_mask)

            # --- Check for NaNs/Infs ---
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"\nWARNING: NaN or Inf detected in model output (Epoch {epoch}, Batch {batch_idx}). Skipping.")
                continue

            # --- Loss Calculation ---
            # Output shape: [batch_size, num_classes], Target shape: [batch_size]
            loss = criterion(output, target)

            # --- Add Orthogonality Penalty ---
            if model.transformer_type == "ospa" and model.orth_mode == "regularize":
                 orth_penalty = model.get_orthogonality_penalty()
                 if not (torch.isnan(orth_penalty).any() or torch.isinf(orth_penalty).any()):
                     loss = loss + args.orth_penalty_weight * orth_penalty
                 else:
                     print(f"\nWARNING: NaN/Inf in orthogonality penalty (Epoch {epoch}, Batch {batch_idx}). Penalty not added.")

            # --- Scale Loss for Grad Accumulation ---
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # --- Backward Pass ---
            loss.backward()

            # --- Optimizer Step ---
            is_accumulation_step = (batch_idx + 1) % args.gradient_accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == len(train_dataloader)
            if is_accumulation_step or is_last_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                if args.scheduler_update_every_step:
                     if scheduler and hasattr(scheduler, 'step'):
                         scheduler.step()

            # --- Metrics Calculation ---
            current_unscaled_loss = loss.item() * args.gradient_accumulation_steps
            total_loss += current_unscaled_loss
            log_loss += current_unscaled_loss

            with torch.no_grad(): # Accuracy calculation doesn't need gradients
                pred = output.argmax(dim=1) # Get predicted class index
                batch_correct = (pred == target).sum().item()
                batch_total = target.size(0)
                correct += batch_correct
                total += batch_total

            # --- Logging ---
            if (batch_idx + 1) % args.log_interval == 0 and batch_idx > 0:
                 lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
                 # Calculate average loss and accuracy over the interval
                 interval_avg_loss = log_loss / args.log_interval
                 # Accuracy calculation needs interval counts if resetting, or use cumulative counts
                 interval_acc = correct / total if total > 0 else 0.0 # Using cumulative acc here
                 pbar.set_postfix({
                     'loss': f"{interval_avg_loss:.4f}",
                     'acc': f"{interval_acc:.4f}",
                     'lr': f"{lr:.6f}"
                 })
                 log_loss = 0.0 # Reset interval loss accumulator

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"\nERROR: CUDA OOM encountered in training (Epoch {epoch}, Batch {batch_idx}).")
                raise e
            else:
                print(f"\nERROR: Unexpected runtime error during training (Epoch {epoch}, Batch {batch_idx}): {e}")
                optimizer.zero_grad(set_to_none=True)
                continue

    # --- End of Epoch ---
    # Step scheduler if updating per epoch
    if not args.scheduler_update_every_step:
         if scheduler and hasattr(scheduler, 'step'):
             scheduler.step()

    # Calculate final epoch average loss (per sample) and accuracy
    avg_epoch_loss = total_loss / total if total > 0 else float('inf')
    avg_epoch_acc = correct / total if total > 0 else 0.0

    return avg_epoch_loss, avg_epoch_acc


def evaluate_classifier(model, eval_dataloader, vocab, criterion, args, eval_type="Validation"):
    """Evaluate a text classifier on evaluation data with padding mask."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    pad_idx = vocab.get('<pad>', -1) # Get padding index

    with torch.no_grad(): # Disable gradients for evaluation
        pbar_desc = f"{eval_type} Evaluating"
        for data, target in tqdm(eval_dataloader, desc=pbar_desc, leave=False):
             # data shape: [seq_len, batch_size]
             data, target = data.to(args.device), target.to(args.device)

             try:
                 # --- Generate Padding Mask ---
                 src_key_padding_mask = (data == pad_idx).transpose(0, 1) if pad_idx != -1 else None
                 # ---------------------------

                 # --- Forward Pass ---
                 output = model(data, src_key_padding_mask=src_key_padding_mask)

                 # --- Check for NaNs/Infs ---
                 if torch.isnan(output).any() or torch.isinf(output).any():
                     print(f"\nWARNING: NaN/Inf detected in {eval_type} output. Skipping batch.")
                     continue

                 # --- Loss Calculation ---
                 loss = criterion(output, target)
                 # Accumulate total loss, weighted by actual batch size (target size)
                 total_loss += loss.item() * target.size(0)

                 # --- Accuracy Calculation ---
                 pred = output.argmax(dim=1)
                 correct += (pred == target).sum().item()
                 total += target.size(0)

             except RuntimeError as e:
                 print(f"\nERROR: {eval_type} evaluation error: {e}")
                 # Skip problematic batches during evaluation
                 continue

    # Calculate average loss (per sample) and accuracy
    avg_loss = total_loss / total if total > 0 else float('inf')
    avg_acc = correct / total if total > 0 else 0.0

    return avg_loss, avg_acc


# --- Main Training Function ---
def train(args):
    """Main training and evaluation function."""
    # --- Reproducibility: Set Seed ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # For full reproducibility, uncomment these, but they may impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Set random seed to {args.seed}")
    # ---------------------------------

    # --- Output Directory ---
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
    except OSError as e:
        print(f"Error creating output directory {args.output_dir}: {e}")
        exit(1)
    # ----------------------

    # --- Device Setup ---
    if args.device is None: # Auto-detect if not specified
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA device requested but not available, falling back to CPU.")
        args.device = 'cpu'
    print(f"Using device: {args.device}")
    # ------------------

    # --- Data Loading ---
    get_batch_func_lm = None # Placeholder for LM batch function
    if args.task == 'lm':
        print("\n--- Loading Language Modeling Data (WikiText-2) ---")
        train_data, val_data, test_data, vocab, get_batch_func_lm = get_language_modeling_data(args)
        ntokens = len(vocab)
        criterion = nn.CrossEntropyLoss() # Standard loss for LM
        if train_data.numel() == 0:
             print("Error: Training data is empty after loading/batchifying!")
             exit(1)
    elif args.task == 'classification':
        print("\n--- Loading Classification Data (IMDB) ---")
        train_dataloader, test_dataloader, vocab = get_classification_data(args)
        ntokens = len(vocab)
        # Output layer has 2 units for binary classification
        criterion = nn.CrossEntropyLoss()
        if len(train_dataloader) == 0:
            print("Error: Training dataloader is empty!")
            exit(1)
    else:
         raise ValueError(f"Unknown task: {args.task}")
    # --------------------

    # --- Model Initialization ---
    print("\n--- Initializing Model ---")
    try:
        model = TransformerModel(
            transformer_type=args.transformer_type,
            vocab_size=ntokens, # Pass vocab size determined from data
            d_model=args.d_model,
            nhead=args.nhead,
            nlayers=args.nlayers,
            dropout=args.dropout,
            dim_feedforward=args.dim_feedforward,
            orth_mode=args.orth_mode,
            orth_penalty_weight=args.orth_penalty_weight, # Passed for reference/internal use
            task=args.task
        ).to(args.device)
    except Exception as e:
        print(f"Error creating model: {e}")
        raise e

    # Print model summary info
    output_layer_dim = getattr(getattr(model, 'output_layer', None), 'out_features', 'N/A')
    print(f"Model Output Dim: {output_layer_dim} | Vocab Size: {ntokens}")
    try:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {param_count/1000000:.2f}M")
    except Exception as e:
        print(f"Could not calculate parameter count: {e}")
    print(f"Model Type: {args.transformer_type.upper()}")
    print(f"Orthogonality Mode: {args.orth_mode}")
    if model.transformer_type == 'ospa' and args.orth_mode == 'regularize':
        print(f"Orthogonality Penalty Weight (lambda): {args.orth_penalty_weight}")
    # --------------------------

    # --- Optimizer and Scheduler ---
    print("\n--- Setting up Optimizer and Scheduler ---")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Calculate T_max for CosineAnnealingLR scheduler
    if args.task == 'lm':
        # Number of optimizer steps per epoch
        num_update_steps_per_epoch = math.ceil((train_data.size(0) - 1) / args.bptt) // args.gradient_accumulation_steps
    else: # classification
        num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps

    if num_update_steps_per_epoch == 0:
         print("Warning: Calculated zero update steps per epoch. Check batch size, data size, and accumulation steps.")
         t_max_steps = args.epochs # Fallback to epochs if no steps calculated
    else:
         t_max_steps = args.epochs * num_update_steps_per_epoch

    print(f"Scheduler T_max: {t_max_steps} steps")
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, t_max_steps), # Ensure T_max >= 1
        eta_min=args.lr / 100 # Example: eta_min is 1/100th of initial LR
    )
    # -----------------------------

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    best_val_metric = float('inf') # Use validation loss for selecting best model
    history = {'args': vars(args), 'epochs': {}} # Store args and epoch results
    model_save_path = os.path.join(args.output_dir, args.save)
    print(f"Best model checkpoint will be saved to: {model_save_path}")

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            print(f"\n--- Epoch {epoch}/{args.epochs} ---")
            epoch_history = {} # Store results for this epoch

            # --- Training Step ---
            if args.task == 'lm':
                 # Pass the specific LM batching function
                train_language_model(model, train_data, optimizer, criterion, scheduler, args, epoch, get_batch_func_lm)
                epoch_history['train_loss'] = None # Not currently returned
            else: # classification
                train_loss, train_acc = train_classifier(model, train_dataloader, vocab, optimizer, criterion, scheduler, args, epoch)
                epoch_history['train_loss'] = train_loss
                epoch_history['train_acc'] = train_acc * 100 # Store as percentage
                print(f'| Epoch {epoch} Training | Avg Loss: {train_loss:.4f} | Avg Acc: {train_acc*100:.2f}%')

            # --- Validation Step ---
            val_start_time = time.time()
            if args.task == 'lm':
                val_loss = evaluate_language_model(model, val_data, vocab, get_batch_func_lm, criterion, args, eval_type="Validation")
                epoch_history['val_loss'] = val_loss
                try:
                    epoch_history['val_ppl'] = math.exp(min(val_loss, 700)) # Cap exponent
                except OverflowError:
                    epoch_history['val_ppl'] = float('inf')
                print(f'| Epoch {epoch} Validation | Time: {time.time() - val_start_time:.2f}s | Loss: {val_loss:.4f} | PPL: {epoch_history["val_ppl"]:8.2f}')
                current_metric = val_loss

            else: # classification
                val_loss, val_acc = evaluate_classifier(model, test_dataloader, vocab, criterion, args, eval_type="Validation")
                epoch_history['val_loss'] = val_loss
                epoch_history['val_acc'] = val_acc * 100 # Store as percentage
                print(f'| Epoch {epoch} Validation | Time: {time.time() - val_start_time:.2f}s | Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%')
                current_metric = val_loss # Use loss for model saving decision

            history['epochs'][epoch] = epoch_history # Store epoch results

            # --- Save Best Model Checkpoint ---
            if current_metric < best_val_metric:
                best_val_metric = current_metric
                print(f'| New best validation metric: {best_val_metric:.4f}. Saving model checkpoint...')
                try:
                    # Save model state dictionary
                    torch.save(model.state_dict(), model_save_path)
                    print(f'| Model checkpoint saved to {model_save_path}')
                except Exception as e:
                    print(f"\n[ERROR] Failed to save model checkpoint: {e}")

            # Step the scheduler if updating per epoch (and not per step)
            if not args.scheduler_update_every_step:
                if scheduler and hasattr(scheduler, 'step'):
                    scheduler.step() # Pass val_loss if using ReduceLROnPlateau, etc.


    except KeyboardInterrupt:
        print('-' * 89)
        print('| Keyboard interrupt detected - Exiting training loop early.')
        print('-' * 89)
    except Exception as e:
         print(f"\nAn unexpected error occurred during training loop: {e}")
         # Optionally save current state before exiting
         error_save_path = os.path.join(args.output_dir, "error_checkpoint.pt")
         print(f"Saving current model state to {error_save_path}")
         torch.save(model.state_dict(), error_save_path)
         raise e # Re-raise the exception to halt execution

    # --- Final Evaluation on Test Set ---
    print("\n--- Final Evaluation on Test Set ---")
    # Load the best model checkpoint for final evaluation
    if os.path.exists(model_save_path):
        try:
            print(f"Loading best model checkpoint from {model_save_path}...")
            # Load state dict onto the correct device
            model.load_state_dict(torch.load(model_save_path, map_location=args.device))
        except Exception as e:
            print(f"\n[ERROR] Failed to load best model checkpoint from {model_save_path}: {e}.")
            print("Evaluating with the model state at the end of training instead.")
    else:
        print("\n[Warning] No best model checkpoint found. Evaluating with the final model state.")

    # Perform final evaluation
    if args.task == 'lm':
        test_loss = evaluate_language_model(model, test_data, vocab, get_batch_func_lm, criterion, args, eval_type="Test")
        try:
            test_ppl = math.exp(min(test_loss, 700))
        except OverflowError:
            test_ppl = float('inf')
        print(f'=' * 89)
        print(f'| End of Training | Test Loss: {test_loss:.4f} | Test PPL: {test_ppl:8.2f} |')
        print(f'=' * 89)
        # Store final results in history
        history['final_test_loss'] = test_loss
        history['final_test_ppl'] = test_ppl
    else:  # classification
        test_loss, test_acc = evaluate_classifier(model, test_dataloader, vocab, criterion, args, eval_type="Test")
        print(f'=' * 89)
        print(f'| End of Training | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}% |')
        print(f'=' * 89)
        # Store final results in history
        history['final_test_loss'] = test_loss
        history['final_test_acc'] = test_acc * 100 # Store accuracy as percentage

    # --- Save Results History ---
    # Create a descriptive filename using args
    lambda_str = f"lambda{args.orth_penalty_weight}" if model.transformer_type == 'ospa' and args.orth_mode == 'regularize' else "lambdaNA"
    results_filename = (
        f"results_{args.task}_{args.transformer_type}_{args.orth_mode}_{lambda_str}_"
        f"d{args.d_model}_h{args.nhead}_l{args.nlayers}_e{args.epochs}_s{args.seed}.json"
    )
    results_filepath = os.path.join(args.output_dir, results_filename)
    print(f"Saving results history to: {results_filepath}")
    try:
        # Save the entire history dictionary (including args)
        with open(results_filepath, 'w') as f:
            # Use default=str to handle potential non-serializable items like device objects in args
            json.dump(history, f, indent=2, default=str)
    except Exception as e:
        print(f"\n[ERROR] Failed to save results JSON to {results_filepath}: {e}")
    # --------------------------

    print("\n--- Training Script Finished ---")


# --- Argument Parser Setup ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Orthogonal Subspace Projection Attention Transformer Training Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help message
        )

    # --- Task Configuration ---
    parser.add_argument('--task', type=str, default='lm', choices=['lm', 'classification'],
                        help='Task to perform: language modeling (lm) or text classification (classification)')
    # Dataset choice is inferred from task in this script, but could be an arg
    # parser.add_argument('--dataset', type=str, default='wikitext-2', choices=['wikitext-2', 'imdb'], help='Dataset')

    # --- Model Architecture ---
    parser.add_argument('--transformer_type', type=str, default='ospa', choices=['ospa', 'vanilla', 'linformer'],
                        help='Type of transformer architecture to use.')
    parser.add_argument('--d_model', type=int, default=512, # Increased default based on results
                        help='Model embedding dimension.')
    parser.add_argument('--nhead', type=int, default=8, # Increased default based on results
                         help='Number of attention heads (must divide d_model).')
    parser.add_argument('--nlayers', type=int, default=6, # Increased default based on results
                         help='Number of transformer encoder layers.')
    parser.add_argument('--dim_feedforward', type=int, default=2048, # Increased default based on results
                        help='Dimension of the feedforward network hidden layer.')
    parser.add_argument('--dropout', type=float, default=0.1,
                         help='Dropout rate applied in the model.')

    # --- OSPA Specific Parameters ---
    parser.add_argument('--orth_mode', type=str, default='regularize', choices=['init', 'regularize', 'strict'],
                        help="How to enforce orthogonality in OSPA ('init', 'regularize', 'strict').")
    parser.add_argument('--orth_penalty_weight', type=float, default=0.001, # Default based on LM results
                        help='Weight (lambda) for the orthogonality penalty loss term (used in regularize mode).')

    # --- Data Handling ---
    parser.add_argument('--vocab_cutoff', type=int, default=30000, # Default based on results
                        help='Limit vocabulary size by frequency (0 for no limit).')
    parser.add_argument('--bptt', type=int, default=70, # Default based on results
                        help='Sequence length (backpropagation through time) for language modeling.')
    parser.add_argument('--max_seq_len', type=int, default=256,
                        help='Maximum sequence length for classification padding/truncation.')

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=15, # Default based on results
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, # Default based on results
                        help='Batch size per device.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, # Default based on results
                        help='Accumulate gradients over N steps before optimizer update (effective batch size = batch_size * N).')
    parser.add_argument('--lr', type=float, default=5e-4, # Default based on results
                        help='Initial learning rate for the AdamW optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (L2 regularization) for the optimizer.')
    parser.add_argument('--clip', type=float, default=0.25, # Default based on results
                        help='Maximum norm for gradient clipping.')
    parser.add_argument('--scheduler_update_every_step', action='store_true', default=True, # Default based on results
                        help='Update learning rate scheduler every optimizer step (recommended for cosine schedule).')

    # --- Runtime Environment ---
    parser.add_argument('--device', type=str, default=None,
                        help="Device to use ('cuda', 'cpu', or specific GPU like 'cuda:0'). Auto-detects if None.")
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of worker processes for DataLoader.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')

    # --- Logging and Saving ---
    parser.add_argument('--log_interval', type=int, default=50, # Default based on results
                        help='Log training status every N batches.')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save model checkpoints and results JSON.')
    parser.add_argument('--save', type=str, default='best_model.pt',
                        help='Filename for saving the best model checkpoint within the output directory.')


    args = parser.parse_args()

    # --- Post-processing Args ---
    # Auto-detect device if not specified
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Basic validation (optional but good practice)
    if args.d_model % args.nhead != 0:
        parser.error(f"--d_model ({args.d_model}) must be divisible by --nhead ({args.nhead})")

    # --- Start Training ---
    train(args)
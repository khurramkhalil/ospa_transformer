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
from datasets import load_dataset

# Assuming these files exist in the same directory or are importable
from improved_transformer_model import TransformerModel

# Data Loading Functions (Assumed Mostly Correct from User Input)
def get_language_modeling_data(args):
    """Prepare data for language modeling task (WikiText-2) using datasets library."""
    from datasets import load_dataset
    from tqdm import tqdm
    import torch

    print("Loading WikiText-2 dataset...")
    wikitext = load_dataset("wikitext", "wikitext-2-v1")

    # Tokenizer: simple whitespace split (Consider replacing with a better tokenizer later)
    def tokenize(text):
        return text.split()

    # Special tokens and initial vocab
    special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>'] # <pad> might not be needed for pure LM
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    word_count = {}

    # Count token frequencies
    print("Building vocabulary...")
    for text in tqdm(wikitext['train']['text']):
        if text.strip():
            for token in tokenize(text):
                # Ensure token is not empty string
                if token and token not in special_tokens:
                    word_count[token] = word_count.get(token, 0) + 1

    # Limit vocabulary if needed
    if args.vocab_cutoff > 0:
        print(f"Limiting vocabulary to {args.vocab_cutoff} tokens (including special tokens)")
        sorted_tokens = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        for token, _ in sorted_tokens:
             if len(vocab) >= args.vocab_cutoff:
                 break
             if token not in vocab: # Ensure token is not already a special token
                vocab[token] = len(vocab)

    else:
        # Add all words if no cutoff
        for token in word_count:
            if token not in vocab:
                 vocab[token] = len(vocab)

    unk_idx = vocab['<unk>']
    assert '<unk>' in vocab and unk_idx == 0, "Special token <unk> must be at index 0"
    print(f"Final vocabulary size: {len(vocab)}")

    # Convert text to token IDs
    def data_process(text_iter):
        data = []
        for text in tqdm(text_iter):
            if text.strip():
                tokens = [vocab.get(token, unk_idx) for token in tokenize(text) if token] # Handle empty tokens
                if tokens:
                    # Optionally add <bos>/<eos> here if needed by model structure
                    # data.append(torch.tensor([vocab['<bos>']] + tokens + [vocab['<eos>']], dtype=torch.long))
                    data.append(torch.tensor(tokens, dtype=torch.long))
        # Concatenate all tensors into one long sequence
        return torch.cat(data) if data else torch.tensor([], dtype=torch.long)

    print("Processing datasets...")
    train_data = data_process(wikitext['train']['text'])
    val_data = data_process(wikitext['validation']['text'])
    test_data = data_process(wikitext['test']['text'])

    # Reshape data into batches
    def batchify(data, batch_size, device):
        if data.numel() == 0:
            return torch.tensor([[]] * batch_size, device=device).t() # Empty tensor with correct batch dim first
        num_batches = data.size(0) // batch_size
        # Drop last incomplete batch
        data = data.narrow(0, 0, num_batches * batch_size)
        # Reshape: [SeqLen * BatchSize] -> [BatchSize, SeqLen] -> [SeqLen, BatchSize]
        data = data.view(batch_size, -1).t().contiguous()
        return data.to(device)

    print("Batchifying...")
    train_data = batchify(train_data, args.batch_size, args.device)
    val_data = batchify(val_data, args.batch_size, args.device)
    test_data = batchify(test_data, args.batch_size, args.device)

    # Function to get a batch slice for LM training
    def get_batch(source, i, bptt):
        # Calculate sequence length, ensuring it doesn't exceed source boundaries
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len] # Input sequence
        target = source[i+1:i+1+seq_len].reshape(-1) # Target sequence (shifted by one)
        return data, target

    return train_data, val_data, test_data, vocab, get_batch


def get_classification_data(args):
    """Prepare data for text classification task (IMDB) using datasets library."""
    import torch
    from torch.utils.data import DataLoader

    print("Loading IMDB dataset...")
    imdb = load_dataset("imdb")

    # Simple tokenizer function (split by whitespace)
    def tokenize(text):
        return text.split()

    # Build vocabulary from tokens
    vocab = {}
    word_count = {}

    # Add special tokens (PAD must be first if using ignore_index=0 later, but keep UNK=0 for consistency)
    # Let's keep UNK=0, PAD=1 for simplicity unless loss function requires specific index.
    special_tokens = ['<unk>', '<pad>']
    for i, token in enumerate(special_tokens):
        vocab[token] = i
    unk_idx = vocab['<unk>']
    pad_idx = vocab['<pad>']

    # Process training tokens and build vocabulary
    print("Building vocabulary...")
    # Consider processing more/all data if time permits for better vocab
    num_build_samples = 10000
    for i, example in enumerate(tqdm(imdb['train'])):
        if i >= num_build_samples:
            break
        text = example['text']
        for token in tokenize(text):
            if token: # Avoid empty strings
                 word_count[token] = word_count.get(token, 0) + 1

    # Add words to vocab after counting all frequencies
    for token in word_count:
        if token not in vocab:
            vocab[token] = len(vocab)

    # Optionally limit vocabulary size
    if args.vocab_cutoff > 0 and len(vocab) > args.vocab_cutoff:
        print(f"Limiting vocabulary from {len(vocab)} to {args.vocab_cutoff} tokens")
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        new_vocab = {token: i for i, token in enumerate(special_tokens)}
        for token, _ in sorted_words:
             if len(new_vocab) >= args.vocab_cutoff:
                 break
             if token not in new_vocab: # Avoid overwriting special tokens
                new_vocab[token] = len(new_vocab)
        vocab = new_vocab
        # Re-assign pad/unk indices after potential cutoff
        unk_idx = vocab['<unk>']
        pad_idx = vocab['<pad>']


    print(f"Vocabulary size: {len(vocab)}")
    print(f"PAD index: {pad_idx}, UNK index: {unk_idx}")

    # Define collate function for DataLoader
    def collate_batch(batch):
        label_list, text_list = [], []
        max_len = args.max_seq_len # Use arg for max length

        for example in batch:
            label_list.append(1 if example['label'] == 1 else 0)
            # Tokenize and map to IDs, handle unknown tokens
            processed_text = [vocab.get(token, unk_idx) for token in tokenize(example['text']) if token] # Map known, others to UNK
            processed_text_tensor = torch.tensor(processed_text, dtype=torch.long)

            # Truncate or pad to fixed length
            if len(processed_text_tensor) > max_len:
                processed_text_tensor = processed_text_tensor[:max_len]
            else:
                padding = torch.full((max_len - len(processed_text_tensor),), pad_idx, dtype=torch.long)
                processed_text_tensor = torch.cat([processed_text_tensor, padding])

            text_list.append(processed_text_tensor)

        label_tensor = torch.tensor(label_list, dtype=torch.long)
        # Stack tensors along batch dimension FIRST, then transpose for Transformer convention
        text_tensor = torch.stack(text_list, dim=0) # [batch_size, seq_len]
        return text_tensor.t().contiguous(), label_tensor  # Return [seq_len, batch_size], [batch_size]

    # Create DataLoaders
    print("Creating dataloaders...")
    train_dataloader = DataLoader(
        imdb['train'],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() # Improve data transfer speed if using CUDA
    )
    # Use test set for validation in this simple setup
    test_dataloader = DataLoader(
        imdb['test'],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_dataloader, test_dataloader, vocab


# Training and Evaluation Functions

def train_language_model(model, train_data, optimizer, criterion, scheduler, args, epoch, get_batch_func):
    """Train a language model on WikiText-2 with causal masking."""
    model.train()
    total_loss = 0.0
    start_time = time.time()
    ntokens = model.vocab_size # Get from model

    pbar = tqdm(range(0, train_data.size(0) - 1, args.bptt),
                desc=f"Epoch {epoch}/{args.epochs}", leave=False)

    for batch, i in enumerate(pbar):
        # Use the get_batch function defined in get_language_modeling_data
        data, targets = get_batch_func(train_data, i, args.bptt)
        seq_len = data.size(0) # Get actual sequence length

        # Clear gradients only at the beginning of accumulation steps
        if batch % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True) # More memory efficient

        try:
            # Generate causal mask using built-in method
            src_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=args.device)

            # Forward pass with causal mask
            output = model(data, src_mask=src_mask) # Pass the generated mask

            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"\nWARNING: NaN or inf in model output at epoch {epoch}, batch {batch}. Skipping.")
                # Consider reducing LR or checking gradients if this persists
                continue

            loss = criterion(output.view(-1, ntokens), targets)

            # Add orthogonality penalty if using OSPA with regularize mode
            if model.transformer_type == "ospa" and model.orth_mode == "regularize":
                # Ensure penalty is computed only when regularization is active
                orth_penalty = model.get_orthogonality_penalty()
                if torch.isnan(orth_penalty).any() or torch.isinf(orth_penalty).any():
                     print(f"\nWARNING: NaN/Inf in orth penalty. Skipping penalty term for batch {batch}.")
                else:
                     loss = loss + args.orth_penalty_weight * orth_penalty


            # Scale loss for gradient accumulation
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            # Gradient accumulation step
            if (batch + 1) % args.gradient_accumulation_steps == 0 or (batch + 1) * args.bptt >= train_data.size(0) -1 : # Also step on last batch
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                # Update scheduler per step if configured
                if args.scheduler_update_every_step:
                     # Check if scheduler exists and has 'step' method
                     if scheduler and hasattr(scheduler, 'step'):
                         scheduler.step()


            current_loss = loss.item() * args.gradient_accumulation_steps # Log unscaled loss
            total_loss += current_loss

            # Log progress periodically
            if batch % args.log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
                ms_per_batch = (time.time() - start_time) * 1000 / args.log_interval
                cur_avg_loss = total_loss / args.log_interval
                try: # Avoid potential overflow with high loss
                    ppl = math.exp(min(cur_avg_loss, 700)) # Cap exponent for display
                except OverflowError:
                    ppl = float('inf')
                pbar.set_postfix({
                    'loss': f"{cur_avg_loss:.2f}",
                    'ppl': f"{ppl:.2f}",
                    'lr': f"{lr:.5f}",
                    #'ms/batch': f"{ms_per_batch:.1f}"
                })
                total_loss = 0.0 # Reset loss accumulator
                start_time = time.time()

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"\nERROR: CUDA OOM at epoch {epoch}, batch {batch}. Try reducing batch size or model size.")
                # Optionally: attempt to clear cache and skip batch, or exit
                # torch.cuda.empty_cache()
                # optimizer.zero_grad(set_to_none=True) # Clear potentially corrupted gradients
                # continue
                raise e # Re-raise for handling in main loop if desired
            else:
                print(f"\nERROR: {e} at epoch {epoch}, batch {batch}. Skipping batch.")
                optimizer.zero_grad(set_to_none=True) # Clear potentially corrupted gradients
                continue # Skip to next batch

    # Return average loss over the epoch (approximate if log_interval doesn't divide perfectly)
    # A more accurate way would be to sum total loss and divide by total batches
    # For now, return None as train_loss is logged within the loop
    return None # Or calculate and return average epoch loss if needed


def evaluate_language_model(model, eval_data, vocab, get_batch_func, criterion, args, eval_type="Validation"):
    """Evaluate the language model using causal mask."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    ntokens = model.vocab_size
    num_evaluated_tokens = 0 # Track number of tokens evaluated

    with torch.no_grad():
        pbar_desc = f"{eval_type} Evaluating"
        for i in tqdm(range(0, eval_data.size(0) - 1, args.bptt), desc=pbar_desc, leave=False):
            data_batch, targets = get_batch_func(eval_data, i, args.bptt)
            seq_len = data_batch.size(0)
            if seq_len == 0: continue # Skip if batch is empty

            try:
                # Generate causal mask using built-in method
                src_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=args.device)

                output = model(data_batch, src_mask=src_mask) # Pass generated mask

                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"WARNING: NaN/Inf in {eval_type} output. Skipping batch.")
                    continue

                loss = criterion(output.view(-1, ntokens), targets) # Calculate loss for the batch
                total_loss += loss.item() * targets.size(0) # Accumulate total loss weighted by target length
                num_evaluated_tokens += targets.size(0)

            except RuntimeError as e:
                print(f"{eval_type} evaluation error: {e}")
                # Skip problematic batches during evaluation
                continue

    if num_evaluated_tokens == 0:
        print(f"Warning: No tokens were evaluated during {eval_type}.")
        return float('inf')

    average_loss = total_loss / num_evaluated_tokens # Calculate average loss over all evaluated tokens
    return average_loss


def train_classifier(model, train_dataloader, vocab, optimizer, criterion, scheduler, args, epoch):
    """Train a text classifier on IMDB with padding mask."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    pad_idx = vocab.get('<pad>', 1) # Get padding index, default to 1 if somehow missing

    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs} Training", leave=False)
    for batch_idx, (data, target) in enumerate(pbar):
        # data shape: [seq_len, batch_size]
        data, target = data.to(args.device), target.to(args.device)

        # Clear gradients
        if batch_idx % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        try:
            # --- PADDING MASK GENERATION ---
            # Mask should be True where values ARE padded.
            # Expected shape: [batch_size, seq_len]
            src_key_padding_mask = (data == pad_idx).transpose(0, 1)
            # -------------------------------

            # Forward pass with padding mask
            output = model(data, src_key_padding_mask=src_key_padding_mask) # Pass generated mask

            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"\nWARNING: NaN or inf in model output at epoch {epoch}, batch {batch_idx}. Skipping.")
                continue

            loss = criterion(output, target) # Output should be [batch_size, num_classes]

            # Add orthogonality penalty
            if model.transformer_type == "ospa" and model.orth_mode == "regularize":
                 orth_penalty = model.get_orthogonality_penalty()
                 if torch.isnan(orth_penalty).any() or torch.isinf(orth_penalty).any():
                     print(f"\nWARNING: NaN/Inf in orth penalty. Skipping penalty term for batch {batch_idx}.")
                 else:
                     loss = loss + args.orth_penalty_weight * orth_penalty

            # Scale loss for gradient accumulation
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            # Accumulate gradients and step
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                # Step scheduler if needed
                if args.scheduler_update_every_step:
                     if scheduler and hasattr(scheduler, 'step'):
                         scheduler.step()

            # --- Metrics ---
            current_loss = loss.item() * args.gradient_accumulation_steps # Log unscaled loss
            total_loss += current_loss

            with torch.no_grad(): # Accuracy calculation doesn't need gradients
                pred = output.argmax(dim=1)
                batch_correct = (pred == target).sum().item()
                batch_total = target.size(0)
                correct += batch_correct
                total += batch_total

            # Update progress bar periodically
            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                 lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
                 current_avg_loss = total_loss / (args.log_interval * args.gradient_accumulation_steps) # Avg loss over interval
                 current_acc = correct / total if total > 0 else 0.0
                 pbar.set_postfix({
                     'loss': f"{current_avg_loss:.4f}",
                     'acc': f"{current_acc:.4f}",
                     'lr': f"{lr:.5f}"
                 })
                 # Reset counters for the next interval averaging if desired, or keep cumulative
                 # total_loss = 0.0
                 # correct = 0
                 # total = 0


        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"\nERROR: CUDA OOM at epoch {epoch}, batch {batch_idx}.")
                raise e
            else:
                print(f"\nERROR: {e} at epoch {epoch}, batch {batch_idx}. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

    # Update learning rate scheduler at the end of epoch if not step-wise
    if not args.scheduler_update_every_step:
         if scheduler and hasattr(scheduler, 'step'):
             scheduler.step()


    # Calculate final epoch average loss and accuracy
    avg_loss = total_loss / len(train_dataloader.dataset) if len(train_dataloader.dataset) > 0 else 0.0 # Avg loss per sample
    avg_acc = correct / total if total > 0 else 0.0

    return avg_loss, avg_acc


def evaluate_classifier(model, eval_dataloader, vocab, criterion, args, eval_type="Validation"):
    """Evaluate a text classifier on evaluation data with padding mask."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    pad_idx = vocab.get('<pad>', 1) # Get padding index

    with torch.no_grad():
        pbar_desc = f"{eval_type} Evaluating"
        for data, target in tqdm(eval_dataloader, desc=pbar_desc, leave=False):
             # data shape: [seq_len, batch_size]
             data, target = data.to(args.device), target.to(args.device)

             try:
                 # --- PADDING MASK GENERATION ---
                 src_key_padding_mask = (data == pad_idx).transpose(0, 1)
                 # -------------------------------

                 # Forward pass with padding mask
                 output = model(data, src_key_padding_mask=src_key_padding_mask) # Pass generated mask

                 if torch.isnan(output).any() or torch.isinf(output).any():
                     print(f"WARNING: NaN/Inf in {eval_type} output. Skipping batch.")
                     continue

                 # Calculate loss
                 loss = criterion(output, target)
                 total_loss += loss.item() * target.size(0) # Accumulate total loss weighted by batch size

                 # Calculate accuracy
                 pred = output.argmax(dim=1)
                 correct += (pred == target).sum().item()
                 total += target.size(0)

             except RuntimeError as e:
                 print(f"{eval_type} evaluation error: {e}")
                 # Skip problematic batches during evaluation
                 continue

    # Calculate average loss and accuracy
    avg_loss = total_loss / total if total > 0 else float('inf')
    avg_acc = correct / total if total > 0 else 0.0

    return avg_loss, avg_acc


# Main Training Function
def train(args):
    """Main training and evaluation function."""
    # --- SET SEED ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # Making CUDA operations deterministic can slightly slow down training
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Set random seed to {args.seed}")
    # ----------------

    # --- Create output directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    # -----------------------------

    # --- Device Setup ---
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA device requested but not available, using CPU.")
        args.device = 'cpu'
    print(f"Using device: {args.device}")
    # ------------------

    # --- Data Loading ---
    if args.task == 'lm':
        print("\n--- Loading Language Modeling Data (WikiText-2) ---")
        train_data, val_data, test_data, vocab, get_batch_func = get_language_modeling_data(args)
        ntokens = len(vocab)
        criterion = nn.CrossEntropyLoss()
    elif args.task == 'classification':
        print("\n--- Loading Classification Data (IMDB) ---")
        train_dataloader, test_dataloader, vocab = get_classification_data(args)
        ntokens = len(vocab)
        # Using standard CrossEntropyLoss as model outputs logits for classes
        criterion = nn.CrossEntropyLoss()
    else:
         raise ValueError(f"Unknown task: {args.task}")
    # --------------------

    # --- Model Initialization ---
    print("\n--- Initializing Model ---")
    try:
        model = TransformerModel(
            transformer_type=args.transformer_type,
            vocab_size=ntokens, # Pass vocab size to model
            d_model=args.d_model,
            nhead=args.nhead,
            nlayers=args.nlayers,
            dropout=args.dropout,
            dim_feedforward=args.dim_feedforward,
            orth_mode=args.orth_mode,
            # Pass the penalty weight, though it's only used internally by OSPA if mode is 'regularize'
            orth_penalty_weight=args.orth_penalty_weight,
            task=args.task
        ).to(args.device)
    except Exception as e:
        print(f"Error creating model: {e}")
        raise e

    # Print model summary info
    output_layer_dim = getattr(getattr(model, 'output_layer', None), 'out_features', 'N/A')
    print(f"Model output dim: {output_layer_dim} | Vocab size: {ntokens}")
    try:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {param_count/1000000:.2f}M")
    except Exception as e:
        print(f"Could not calculate parameter count: {e}")
    print(f"Model Type: {args.transformer_type.upper()} Transformer")
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

    # Calculate T_max for scheduler correctly
    if args.task == 'lm':
        num_batches_per_epoch = math.ceil((train_data.size(0) - 1) / args.bptt) if train_data.numel() > 0 else 0
    else: # classification
        num_batches_per_epoch = len(train_dataloader)

    if num_batches_per_epoch == 0:
         print("Warning: No batches found for training. Check data loading and batch size.")
         t_max_steps = 1 # Prevent division by zero
    else:
         t_max_steps = args.epochs * (num_batches_per_epoch // args.gradient_accumulation_steps)

    print(f"Scheduler T_max: {t_max_steps} steps")
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, t_max_steps), # Ensure T_max is at least 1
        eta_min=args.lr / 100 # Avoid eta_min=0
    )
    # -----------------------------

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    best_val_metric = float('inf') # Use loss for comparison
    # Store history including args for better tracking
    history = {'args': vars(args), 'epochs': {}}
    model_save_path = os.path.join(args.output_dir, args.save)
    print(f"Best model checkpoint will be saved to: {model_save_path}")

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            print(f"\n--- Epoch {epoch}/{args.epochs} ---")
            epoch_history = {}

            # --- Training Step ---
            if args.task == 'lm':
                train_language_model(model, train_data, optimizer, criterion, scheduler, args, epoch, get_batch_func)
                # We don't capture train loss directly from the function currently
                epoch_history['train_loss'] = None # Placeholder
            else: # classification
                train_loss, train_acc = train_classifier(model, train_dataloader, vocab, optimizer, criterion, scheduler, args, epoch)
                epoch_history['train_loss'] = train_loss
                epoch_history['train_acc'] = train_acc
                print(f'| Epoch {epoch} Training | Avg Loss: {train_loss:.4f} | Avg Acc: {train_acc*100:.2f}%')

            # --- Validation Step ---
            if args.task == 'lm':
                val_loss = evaluate_language_model(model, val_data, vocab, get_batch_func, criterion, args, eval_type="Validation")
                epoch_history['val_loss'] = val_loss
                try:
                    epoch_history['val_ppl'] = math.exp(min(val_loss, 700))
                except OverflowError:
                    epoch_history['val_ppl'] = float('inf')
                print(f'| Epoch {epoch} Validation | Time: {time.time() - epoch_start_time:.2f}s | Loss: {val_loss:5.2f} | PPL: {epoch_history["val_ppl"]:8.2f}')
                current_metric = val_loss # Use validation loss to determine best model

            else: # classification
                val_loss, val_acc = evaluate_classifier(model, test_dataloader, vocab, criterion, args, eval_type="Validation")
                epoch_history['val_loss'] = val_loss
                epoch_history['val_acc'] = val_acc
                print(f'| Epoch {epoch} Validation | Time: {time.time() - epoch_start_time:.2f}s | Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%')
                current_metric = val_loss # Use validation loss

            history['epochs'][epoch] = epoch_history

            # --- Save Best Model ---
            if current_metric < best_val_metric:
                best_val_metric = current_metric
                print(f'| New best validation metric: {best_val_metric:.4f}. Saving model...')
                try:
                    torch.save(model.state_dict(), model_save_path)
                    print(f'| Model saved to {model_save_path}')
                except Exception as e:
                    print(f"[ERROR] Failed to save model checkpoint due to: {e}")
                    # Decide whether to continue or exit

    except KeyboardInterrupt:
        print('-' * 89)
        print('| Keyboard interrupt detected - Exiting training early')
        print('-' * 89)
    except Exception as e:
         print(f"\nAn unexpected error occurred during training: {e}")
         # Optionally save current state before exiting
         # torch.save(model.state_dict(), os.path.join(args.output_dir, "error_checkpoint.pt"))
         raise e # Re-raise the exception

    # --- Final Evaluation ---
    print("\n--- Final Evaluation on Test Set ---")
    # Load the best model saved during training
    if os.path.exists(model_save_path):
        try:
            print(f"Loading best model from {model_save_path}...")
            model.load_state_dict(torch.load(model_save_path, map_location=args.device))
        except Exception as e:
            print(f"[ERROR] Failed to load best model from {model_save_path}: {e}. Evaluating with final model state.")
    else:
        print("[Warning] No best model checkpoint found. Evaluating with final model state.")

    if args.task == 'lm':
        test_loss = evaluate_language_model(model, test_data, vocab, get_batch_func, criterion, args, eval_type="Test")
        try:
            test_ppl = math.exp(min(test_loss, 700))
        except OverflowError:
            test_ppl = float('inf')
        print(f'=' * 89)
        print(f'| End of Training | Test Loss: {test_loss:.4f} | Test PPL: {test_ppl:8.2f} |')
        print(f'=' * 89)
        history['final_test_loss'] = test_loss
        history['final_test_ppl'] = test_ppl
    else:  # classification
        test_loss, test_acc = evaluate_classifier(model, test_dataloader, vocab, criterion, args, eval_type="Test")
        print(f'=' * 89)
        print(f'| End of Training | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}% |')
        print(f'=' * 89)
        history['final_test_loss'] = test_loss
        history['final_test_acc'] = test_acc * 100 # Store accuracy as percentage

    # --- Save Results History ---
    # Create a descriptive filename
    lambda_str = f"lambda{args.orth_penalty_weight}" if model.transformer_type == 'ospa' and args.orth_mode == 'regularize' else "lambdaNA"
    results_filename = (
        f"results_{args.task}_{args.transformer_type}_{args.orth_mode}_{lambda_str}_"
        f"d{args.d_model}_h{args.nhead}_l{args.nlayers}_e{args.epochs}_s{args.seed}.json"
    )
    results_filepath = os.path.join(args.output_dir, results_filename)
    print(f"Saving results history to: {results_filepath}")
    try:
        # Save the entire history dictionary including args
        with open(results_filepath, 'w') as f:
            # Use a custom encoder for potential non-serializable args (like device) if needed
            # json.dump(history, f, indent=2, cls=CustomEncoder)
            # For simplicity, assuming args are basic types:
            json.dump(history, f, indent=2, default=str) # Use default=str to handle non-serializable types gracefully
    except Exception as e:
        print(f"[ERROR] Failed to save results JSON to {results_filepath}: {e}")
    # --------------------------

    print("\n--- Training Complete ---")


# --- Argument Parser Setup ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Orthogonal Subspace Projection Attention Transformer Training')

    # Task Configuration
    parser.add_argument('--task', type=str, default='lm', choices=['lm', 'classification'],
                        help='Task: language modeling (lm) or text classification (classification)')
    parser.add_argument('--dataset', type=str, default='wikitext-2', choices=['wikitext-2', 'imdb'],
                        help='Dataset to use (derived from task, but explicit is clearer)') # Added for clarity

    # Model Configuration
    parser.add_argument('--transformer_type', type=str, default='ospa', choices=['ospa', 'vanilla', 'linformer'],
                        help='Type of transformer architecture (default: ospa)')
    parser.add_argument('--d_model', type=int, default=256, help='Model embedding dimension (default: 256)')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads (default: 4)')
    parser.add_argument('--nlayers', type=int, default=3, help='Number of transformer encoder layers (default: 3)')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='Dimension of feedforward network (default: 1024)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (default: 0.1)')

    # OSPA Specific Parameters
    parser.add_argument('--orth_mode', type=str, default='regularize', choices=['init', 'regularize', 'strict'],
                        help="How to enforce orthogonality in OSPA ('init', 'regularize', 'strict') (default: regularize)")
    parser.add_argument('--orth_penalty_weight', type=float, default=0.0001,
                        help='Weight (lambda) for orthogonality penalty (used in regularize mode) (default: 0.0001)')

    # Data Parameters
    parser.add_argument('--vocab_cutoff', type=int, default=10000, help='Limit vocabulary size (0 for no limit) (default: 10000)')
    parser.add_argument('--bptt', type=int, default=35, help='Sequence length (backprop through time) for language modeling (default: 35)')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Maximum sequence length for classification padding/truncation (default: 256)')

    # Training Parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per device (default: 32)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients before optimizer step (default: 1)')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate (default: 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (L2 penalty) (default: 0.01)')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping value (default: 1.0)')
    parser.add_argument('--scheduler_update_every_step', action='store_true',
                        help='Update learning rate scheduler every step instead of every epoch')

    # Runtime Parameters
    parser.add_argument('--device', type=str, default=None, #'cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to use ('cuda', 'cpu', or specific e.g. 'cuda:0'). Auto-detects if None. (default: auto)")
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of dataloader worker processes (default: 2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')

    # Logging and Saving
    parser.add_argument('--log_interval', type=int, default=100, help='Log training status every N batches (default: 100)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save models and results (default: outputs)')
    parser.add_argument('--save', type=str, default='best_model.pt',
                        help='Filename for saving the best model checkpoint (default: best_model.pt)')


    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set dataset based on task if not explicitly given (optional, task implies dataset here)
    if args.task == 'lm' and args.dataset != 'wikitext-2':
        print("Warning: Task is 'lm', setting dataset to 'wikitext-2'")
        args.dataset = 'wikitext-2'
    elif args.task == 'classification' and args.dataset != 'imdb':
        print("Warning: Task is 'classification', setting dataset to 'imdb'")
        args.dataset = 'imdb'


    # Start Training
    train(args)
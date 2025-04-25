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

from datasets import load_dataset

from improved_transformer_model import TransformerModel

def get_language_modeling_data(args):
    """Prepare data for language modeling task (WikiText-2) using datasets library."""
    from datasets import load_dataset
    from tqdm import tqdm
    import torch

    # Load WikiText-2 dataset
    wikitext = load_dataset("wikitext", "wikitext-2-v1")

    # Tokenizer: simple whitespace split
    def tokenize(text):
        return text.split()

    # Special tokens and initial vocab
    special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    word_count = {}

    # Count token frequencies
    print("Building vocabulary...")
    for text in tqdm(wikitext['train']['text']):
        if text.strip():
            for token in tokenize(text):
                if token not in special_tokens:
                    word_count[token] = word_count.get(token, 0) + 1

    # Limit vocabulary if needed
    if args.vocab_cutoff > 0:
        print(f"Limiting vocabulary to {args.vocab_cutoff} tokens (including special tokens)")
        sorted_tokens = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        for token, _ in sorted_tokens[:args.vocab_cutoff - len(special_tokens)]:
            vocab[token] = len(vocab)
    else:
        for token in word_count:
            vocab[token] = len(vocab)

    assert '<unk>' in vocab and vocab['<unk>'] == 0, "Special token <unk> must be at index 0"
    print(f"Final vocabulary size: {len(vocab)}")

    # Convert text to token IDs
    def data_process(text_iter):
        data = []
        for text in tqdm(text_iter):
            if text.strip():
                tokens = [vocab.get(token, vocab['<unk>']) for token in tokenize(text)]
                if tokens:
                    data.append(torch.tensor(tokens, dtype=torch.long))
        return torch.cat(data)

    print("Processing datasets...")
    train_data = data_process(wikitext['train']['text'])
    val_data = data_process(wikitext['validation']['text'])
    test_data = data_process(wikitext['test']['text'])

    # Reshape data into batches
    def batchify(data, batch_size):
        nbatch = data.size(0) // batch_size
        data = data[:nbatch * batch_size].view(batch_size, -1).t().contiguous()
        return data.to(args.device)

    print("Batchifying...")
    train_data = batchify(train_data, args.batch_size)
    val_data = batchify(val_data, args.batch_size)
    test_data = batchify(test_data, args.batch_size)

    # Create batched slices
    def get_batch(source, i, bptt):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target

    return train_data, val_data, test_data, vocab, get_batch

# Re-implementing for clarity if needed:
def generate_square_subsequent_mask(sz, device):
    """Generates a square causal mask for attending to previous tokens."""
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# --- Modified train_language_model ---
def train_language_model(model, train_data, val_data, vocab, get_batch, optimizer, criterion, scheduler, args, epoch):
    """Train a language model on WikiText-2."""
    model.train()
    total_loss = 0.0
    start_time = time.time()
    ntokens = len(vocab)

    # Use gradient accumulation if batch size is very large
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps

    pbar = tqdm(range(0, train_data.size(0) - 1, args.bptt),
               desc=f"Epoch {epoch}/{args.epochs}")

    # Initialize mask - it might change size slightly at the end
    src_mask = None

    for batch, i in enumerate(pbar):
        data, targets = get_batch(train_data, i, args.bptt)
        seq_len = data.size(0) # Get actual sequence length for this batch

        # Clear gradients only at the beginning of accumulation steps
        if batch % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        try:
            # --- MASK GENERATION ---
            # Generate causal mask if not created or if seq_len changes
            if src_mask is None or src_mask.size(0) != seq_len:
                src_mask = generate_square_subsequent_mask(seq_len, args.device)
            # ---------------------

            # Forward pass - PASS THE MASK
            output = model(data, src_mask=src_mask) # <--- Pass src_mask here

            # Check for NaN or inf in output
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"WARNING: NaN or inf in model output. Skipping batch.")
                continue

            # Safety check: are all targets in vocab?
            if targets.max().item() >= ntokens or targets.min().item() < 0:
                print(f"[ERROR] Invalid target indices detected:")
                print(f"  → Max target: {targets.max().item()} (Vocab size: {ntokens})")
                print(f"  → Min target: {targets.min().item()}")
                print(f"  → Batch index: {batch}")
                print("  → Sample target values:", targets[:20])
                exit(1)

            assert targets.max() < ntokens, f"Invalid target index {targets.max().item()} >= vocab size {ntokens}"

            # Calculate loss
            loss = criterion(output.view(-1, ntokens), targets) # Reshape was slightly off before

            # Add orthogonality penalty if using OSPA with regularize mode
            if model.transformer_type == "ospa" and model.orth_mode == "regularize":
                orth_penalty = model.get_orthogonality_penalty()
                loss = loss + args.orth_penalty_weight * orth_penalty

            # Scale loss for gradient accumulation
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Accumulate gradients for multiple steps
            if (batch + 1) % args.gradient_accumulation_steps == 0 or (batch + 1) == len(pbar):
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

                # Update weights
                optimizer.step()

                # Update scheduler
                if args.scheduler_update_every_step:
                    scheduler.step()

            # Update metrics
            total_loss += loss.item() * args.gradient_accumulation_steps

            # Log progress
            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time

                # Calculate perplexity safely
                try:
                    # Cap loss to prevent overflow
                    safe_loss = min(cur_loss, 20)
                    ppl = math.exp(safe_loss)
                    ppl_str = f"{ppl:.2f}"
                except OverflowError:
                    ppl_str = "N/A"

                pbar.set_postfix({
                    'loss': f"{cur_loss:.2f}",
                    'ppl': ppl_str,
                    'ms/batch': f"{elapsed * 1000 / args.log_interval:.2f}"
                })

                total_loss = 0
                start_time = time.time()

        except RuntimeError as e:
            print(f"ERROR: {e}")
            if "CUDA out of memory" in str(e): # Handle OOM specifically if needed
                 print("OOM error encountered. Try reducing batch size or model size.")
                 # Potentially break or implement retry logic
            print(f"Skipping problematic batch")
            optimizer.zero_grad() # Ensure gradients are cleared if an error occurs mid-accumulation
            continue

    # Update learning rate scheduler at the end of epoch if not updated per step
    if not args.scheduler_update_every_step:
        scheduler.step()

    # Validate after each epoch
    val_loss = evaluate_language_model(model, val_data, vocab, get_batch, criterion, args) # Pass vocab
    print(f'| End of epoch {epoch:3d} | valid loss {val_loss:5.2f} | valid ppl {math.exp(min(val_loss, 20)):8.2f}')

    return val_loss


# --- Modified evaluate_language_model ---
def evaluate_language_model(model, data, vocab, get_batch, criterion, args):
    """Evaluate a language model on validation or test data."""
    model.eval()
    total_loss = 0.0
    ntokens = len(vocab)

    # Initialize mask
    src_mask = None

    with torch.no_grad():
        for i in tqdm(range(0, data.size(0) - 1, args.bptt),
                     desc="Evaluating"):
            data_batch, targets = get_batch(data, i, args.bptt)
            seq_len = data_batch.size(0)

            try:
                # --- MASK GENERATION ---
                if src_mask is None or src_mask.size(0) != seq_len:
                    src_mask = generate_square_subsequent_mask(seq_len, args.device)
                # ---------------------

                # Forward pass - PASS THE MASK
                output = model(data_batch, src_mask=src_mask) # <--- Pass src_mask here

                # Skip batches with NaN or inf values
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"WARNING: NaN or inf in evaluation output. Skipping.")
                    continue

                loss = criterion(output.view(-1, ntokens), targets).item() # Reshape was slightly off
                total_loss += loss * targets.size(0) # Use target size for accurate loss aggregation
            except RuntimeError as e:
                print(f"Evaluation error: {e}")
                # Skip problematic batches during evaluation
                continue

    # Avoid division by zero if data size is less than bptt
    num_evaluated_tokens = (data.size(0) - 1)
    if num_evaluated_tokens == 0:
        return float('inf') # Or handle appropriately

    return total_loss / num_evaluated_tokens


def get_classification_data(args):
    """Prepare data for text classification task (IMDB) using datasets library."""
    import torch
    from torch.utils.data import DataLoader
    
    # Load IMDB dataset
    imdb = load_dataset("imdb")
    
    # Simple tokenizer function (split by whitespace)
    def tokenize(text):
        return text.split()
    
    # Build vocabulary from tokens
    vocab = {}
    word_count = {}
    
    # Add special tokens
    special_tokens = ['<unk>', '<pad>']
    for i, token in enumerate(special_tokens):
        vocab[token] = i
    
    # Process a subset of training tokens and build vocabulary (for efficiency)
    print("Building vocabulary...")
    for i, example in enumerate(tqdm(imdb['train'])):
        if i >= 5000:  # Limit processing for speed
            break
        text = example['text']
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
            word_count[token] = word_count.get(token, 0) + 1
    
    # Optionally limit vocabulary size (can help with stability)
    if args.vocab_cutoff > 0 and len(vocab) > args.vocab_cutoff:
        print(f"Limiting vocabulary from {len(vocab)} to {args.vocab_cutoff} tokens")
        # Sort by frequency
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        # Keep only most frequent words
        new_vocab = {token: i for i, token in enumerate(special_tokens)}
        for i, (token, _) in enumerate(sorted_words[:args.vocab_cutoff - len(special_tokens)]):
            new_vocab[token] = i + len(special_tokens)
        vocab = new_vocab
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Define collate function for DataLoader
    def collate_batch(batch):
        label_list, text_list = [], []
        for example in batch:
            label_list.append(1 if example['label'] == 1 else 0)
            processed_text = torch.tensor([vocab.get(token, vocab['<unk>']) for token in tokenize(example['text'])], 
                                        dtype=torch.long)
            # Truncate or pad to fixed length
            if len(processed_text) > args.max_seq_len:
                processed_text = processed_text[:args.max_seq_len]
            else:
                processed_text = torch.cat([
                    processed_text, 
                    torch.ones(args.max_seq_len - len(processed_text), dtype=torch.long) * vocab['<pad>']
                ])
            text_list.append(processed_text)
            
        label_tensor = torch.tensor(label_list, dtype=torch.long)
        text_tensor = torch.stack(text_list)
        return text_tensor.t(), label_tensor  # [seq_len, batch_size], [batch_size]
    
    # Create DataLoaders
    print("Creating dataloaders...")
    train_dataloader = DataLoader(
        imdb['train'], 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.num_workers
    )
    test_dataloader = DataLoader(
        imdb['test'], 
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers
    )
    
    return train_dataloader, test_dataloader, vocab

# --- Modified train_classifier ---
def train_classifier(model, train_dataloader, vocab, optimizer, criterion, scheduler, args, epoch): # Added vocab
    """Train a text classifier on IMDB."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    pad_idx = vocab['<pad>'] # Get padding index

    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs}")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(args.device), target.to(args.device) #[seq_len, batch_size]

        # Clear gradients only at the beginning of accumulation steps
        if batch_idx % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        try:
            # --- MASK GENERATION ---
            # Create padding mask: True where pads are, False elsewhere
            # Shape: [batch_size, seq_len]
            src_key_padding_mask = (data == pad_idx).transpose(0, 1)
            # ---------------------

            # Forward pass - PASS THE MASK
            output = model(data, src_key_padding_mask=src_key_padding_mask) # <--- Pass padding mask here

            # Check for NaN or inf in output
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"WARNING: NaN or inf in model output. Skipping batch.")
                continue

            # Calculate loss
            loss = criterion(output, target)

            # Add orthogonality penalty if using OSPA with regularize mode
            if model.transformer_type == "ospa" and model.orth_mode == "regularize":
                orth_penalty = model.get_orthogonality_penalty()
                loss = loss + args.orth_penalty_weight * orth_penalty

            # Scale loss for gradient accumulation
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Accumulate gradients for multiple steps
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(pbar):
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

                # Update weights
                optimizer.step()

                # Update scheduler
                if args.scheduler_update_every_step:
                    scheduler.step()

            # Calculate accuracy
            pred = output.argmax(dim=1)
            batch_correct = (pred == target).sum().item()
            batch_total = target.size(0)

            correct += batch_correct
            total += batch_total

            # Update metrics
            total_loss += loss.item() * args.gradient_accumulation_steps

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{batch_correct/batch_total:.4f}"
            })

        except RuntimeError as e:
            print(f"ERROR: {e}")
            if "CUDA out of memory" in str(e): # Handle OOM specifically if needed
                 print("OOM error encountered. Try reducing batch size or model size.")
            print(f"Skipping problematic batch")
            optimizer.zero_grad() # Ensure gradients are cleared if an error occurs mid-accumulation
            continue

    # Update learning rate scheduler at the end of epoch if not updated per step
    if not args.scheduler_update_every_step:
        scheduler.step()

    avg_loss = total_loss / len(train_dataloader) if len(train_dataloader) > 0 else float('inf')
    avg_acc = correct / total if total > 0 else 0.0

    return avg_loss, avg_acc


# --- Modified evaluate_classifier ---
def evaluate_classifier(model, test_dataloader, vocab, criterion, args): # Added vocab
    """Evaluate a text classifier on test data."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pad_idx = vocab['<pad>'] # Get padding index

    with torch.no_grad():
        for data, target in tqdm(test_dataloader, desc="Evaluating"):
            data, target = data.to(args.device), target.to(args.device) # [seq_len, batch_size]

            try:
                # --- MASK GENERATION ---
                src_key_padding_mask = (data == pad_idx).transpose(0, 1) # [batch_size, seq_len]
                # ---------------------

                # Forward pass - PASS THE MASK
                output = model(data, src_key_padding_mask=src_key_padding_mask) # <--- Pass padding mask here

                # Skip batches with NaN or inf values
                if torch.isnan(output).any() or torch.isinf(output).any():
                     print(f"WARNING: NaN or inf in evaluation output. Skipping.")
                     continue

                # Calculate loss
                loss = criterion(output, target)
                total_loss += loss.item() * target.size(0) # Use target size for accurate aggregation

                # Calculate accuracy
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

            except RuntimeError as e:
                print(f"Evaluation error: {e}")
                # Skip problematic batches during evaluation
                continue

    # Print test results
    test_loss = total_loss / total if total > 0 else float('inf')
    test_acc = 100 * correct / total if total > 0 else 0
    print(f'| Test loss {test_loss:.2f} | Test accuracy {test_acc:.2f}%')

    return test_loss, test_acc

# --- Modified train function signature calls ---
def train(args):
    """Main training function."""
    # --- SET SEED ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # ----------------------------

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True) # Changed from args.output to args.output_dir

    # Print device info
    print(f"Using device: {args.device}")

    # Set up data
    if args.task == 'lm':
        print("Loading language modeling data...")
        train_data, val_data, test_data, vocab, get_batch = get_language_modeling_data(args)
        ntokens = len(vocab)
        criterion = nn.CrossEntropyLoss() # Default ignore_index=-100 might be fine if no padding here
    else:  # classification
        print("Loading classification data...")
        train_dataloader, test_dataloader, vocab = get_classification_data(args)
        ntokens = len(vocab)
        pad_idx = vocab['<pad>']
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx) # Ignore padding index in loss
        # Note: You might NOT want to ignore padding if your model output is only based on [CLS] or first token
        # Since your model uses output[0], ignoring padding in the loss is likely NOT necessary
        # Revert to: criterion = nn.CrossEntropyLoss() # IF output is only based on first token repr.
        # Let's keep it simple for now, assuming the target `criterion(output, target)` doesn't see padding:
        criterion = nn.CrossEntropyLoss()

    # Create model
    print("Creating model...")
    model = TransformerModel(
        transformer_type=args.transformer_type,
        vocab_size=ntokens,
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
        orth_mode=args.orth_mode,
        orth_penalty_weight=args.orth_penalty_weight,
        task=args.task
    ).to(args.device)

    print("Model output dim:", model.output_layer.out_features if hasattr(model, 'output_layer') else 'N/A', "| Vocab size:", ntokens)

    # Set up optimizer and scheduler
    print("Setting up optimizer and scheduler...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Use Cosine Annealing scheduler
    if args.task == 'lm':
        num_batches = len(range(0, train_data.size(0) - 1, args.bptt))
    else:
        num_batches = len(train_dataloader)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * (num_batches // args.gradient_accumulation_steps),
        eta_min=args.lr / 100 # Avoid eta_min=0
    )

    # Print model info
    print(f"Model: {args.transformer_type.upper()} Transformer")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1000000:.2f}M")
    print(f"Orthogonality Mode: {args.orth_mode}")
    if args.orth_mode == 'regularize':
        print(f"Orthogonality Penalty Weight: {args.orth_penalty_weight}")

    # Training loop
    best_val_metric = float('inf') # Use loss for LM, maybe -accuracy for classification? Let's stick to loss.
    history_data = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []} # More structured history


    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time() # Time epochs
            print(f"\n--- Epoch {epoch}/{args.epochs} ---")

            # Train for one epoch
            if args.task == 'lm':
                train_loss = train_language_model(model, train_data, val_data, vocab, get_batch, optimizer, criterion, scheduler, args, epoch) # This function doesn't return train loss currently
                val_loss = evaluate_language_model(model, val_data, vocab, get_batch, criterion, args)
                print(f'| Epoch {epoch} Time: {time.time() - epoch_start_time:.2f}s | Valid Loss: {val_loss:5.2f} | Valid PPL: {math.exp(min(val_loss, 700)):8.2f}') # Cap PPL exponent
                current_metric = val_loss
                history_data['val_loss'].append(val_loss)

            else:  # classification
                train_loss, train_acc = train_classifier(model, train_dataloader, vocab, optimizer, criterion, scheduler, args, epoch) # Pass vocab
                print(f'| Epoch {epoch} Time: {time.time() - epoch_start_time:.2f}s | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
                val_loss, val_acc = evaluate_classifier(model, test_dataloader, vocab, criterion, args) # Pass vocab
                # Use test set as validation here - common for IMDB, but consider a separate val split ideally
                print(f'| Epoch {epoch} Evaluation | Valid Loss: {val_loss:.3f} | Valid Acc: {val_acc:.2f}%')
                current_metric = val_loss # Or use -val_acc if optimizing for accuracy
                history_data['train_loss'].append(train_loss)
                history_data['train_acc'].append(train_acc)
                history_data['val_loss'].append(val_loss)
                history_data['val_acc'].append(val_acc)


            # Save model if validation metric improved
            if current_metric < best_val_metric:
                best_val_metric = current_metric
                save_path = os.path.join(args.output_dir, args.save) # Use output_dir
                try:
                    torch.save(model.state_dict(), save_path)
                    print(f'| Model saved to {save_path} (Validation Metric: {best_val_metric:.4f})')
                except Exception as e: # Catch broader exceptions
                    print(f"[ERROR] Failed to save model due to: {e}")
                    # Decide whether to exit or continue
                    # exit(1)

    except KeyboardInterrupt:
        print('-' * 89)
        print('| Keyboard interrupt detected - Exiting training early')
        print('-' * 89)

    # --- Load BEST model for final evaluation ---
    # Construct filename using args for uniqueness
    output_filename = f"{args.task}_{args.transformer_type}_{args.orth_mode}_lambda{args.orth_penalty_weight if args.orth_mode == 'regularize' else 0}_d{args.d_model}_h{args.nhead}_l{args.nlayers}.json"
    model_path = os.path.join(args.output_dir, args.save) # Use output_dir
    if os.path.exists(model_path):
        try:
            print(f"Loading best model from {model_path} for final evaluation...")
            model.load_state_dict(torch.load(model_path, map_location=args.device))
            torch.save(model.state_dict(), model_path)
        except Exception as e:
            print(f"[ERROR] Failed to load best model from {model_path}: {e}")
            print("Proceeding with the current model state for final evaluation.")
    else:
        print("[Warning] No saved model found. Evaluating with the final model state.")


    # Final evaluation
    print("\n--- Final Evaluation ---")
    if args.task == 'lm':
        test_loss = evaluate_language_model(model, test_data, vocab, get_batch, criterion, args) # Pass vocab
        print(f'=' * 89)
        print(f'| End of training | Test Loss {test_loss:5.2f} | Test PPL {math.exp(min(test_loss, 700)):8.2f}') # Cap PPL exponent
        print(f'=' * 89)
        history_data['test_loss'] = test_loss
        history_data['test_ppl'] = math.exp(min(test_loss, 700))
    else:  # classification
        # Re-evaluate on the test set using the loaded best model
        test_loss, test_acc = evaluate_classifier(model, test_dataloader, vocab, criterion, args) # Pass vocab
        print(f'=' * 89)
        print(f'| End of training | Test Loss {test_loss:5.2f} | Test Accuracy {test_acc:5.2f}%')
        print(f'=' * 89)
        history_data['test_loss'] = test_loss
        history_data['test_acc'] = test_acc

    # --- Corrected JSON saving ---
    filepath = os.path.join(args.output_dir, output_filename)
    print(f"Saving results history to: {filepath}")
    try:
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2) # Save history_data directly
    except Exception as e:
        print(f"[ERROR] Failed to save results JSON: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Orthogonal Subspace Projection Attention')

    # Model configuration
    parser.add_argument('--transformer_type', type=str, default='ospa', choices=['ospa', 'vanilla', 'linformer'],
                        help='Type of transformer architecture')
    parser.add_argument('--task', type=str, default='lm', choices=['lm', 'classification'],
                        help='Task: language modeling (lm) or text classification')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--nlayers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='Dimension of feedforward network')

    # Orthogonality parameters
    parser.add_argument('--orth_mode', type=str, default='regularize', choices=['init', 'regularize', 'strict'], # Changed default
                        help='How to enforce orthogonality')
    parser.add_argument('--orth_penalty_weight', type=float, default=0.0001, # Changed default based on IWSLT results
                        help='Weight for orthogonality penalty (used in regularize mode)')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size') # Slightly increased default
    parser.add_argument('--bptt', type=int, default=35, help='Sequence length for language modeling')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Max sequence length for classification')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate') # Reduced default
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs') # Reduced default for quicker tests
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Steps for gradient accumulation')
    parser.add_argument('--scheduler_update_every_step', action='store_true', help='Update scheduler every step')
    parser.add_argument('--vocab_cutoff', type=int, default=10000, help='Limit vocabulary size (0 for no limit)')
    parser.add_argument('--log_interval', type=int, default=100, help='Report interval') # Reduced default
    parser.add_argument('--save', type=str, default='best_model.pt', help='Model save filename') # Changed default
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for models and results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--num_workers', type=int, default=2, help='Data loader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # --- Removed automatic output filename generation ---
    # We now generate filename before saving JSON
    # args.output = ... (removed)

    train(args)

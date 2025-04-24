import os
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
    import torch
    
    # Load WikiText-2 dataset
    wikitext = load_dataset("wikitext", "wikitext-2-v1")
    
    # Simple tokenizer function (split by whitespace)
    def tokenize(text):
        return text.split()
    
    # Build vocabulary from tokens
    vocab = {}
    word_count = {}
    
    # Add special tokens
    special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']
    for i, token in enumerate(special_tokens):
        vocab[token] = i
    
    # Process all training tokens and build vocabulary
    print("Building vocabulary...")
    for text in tqdm(wikitext['train']['text']):
        if text.strip():  # Skip empty lines
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
    
    # Process datasets
    def data_process(raw_text_iter):
        data = []
        for text in tqdm(raw_text_iter):
            if text.strip():  # Skip empty lines
                tokens = [vocab.get(token, vocab['<unk>']) for token in tokenize(text)]
                if tokens:
                    data.append(torch.tensor(tokens, dtype=torch.long))
        return torch.cat(data)
    
    print("Processing train data...")
    train_data = data_process(wikitext['train']['text'])
    print("Processing validation data...")
    val_data = data_process(wikitext['validation']['text'])
    print("Processing test data...")
    test_data = data_process(wikitext['test']['text'])
    
    # Batch data
    def batchify(data, batch_size):
        # Work out how cleanly we can divide the dataset into batch_size parts
        nbatch = data.size(0) // batch_size
        if nbatch == 0:
            raise ValueError(f"Dataset too small for batch size {batch_size}")
        # Trim off any extra elements that wouldn't cleanly fit
        data = data.narrow(0, 0, nbatch * batch_size)
        # Evenly divide the data across the batch_size batches
        data = data.view(batch_size, -1).t().contiguous()
        return data.to(args.device)
    
    print("Batchifying data...")
    train_data = batchify(train_data, args.batch_size)
    val_data = batchify(val_data, args.batch_size)
    test_data = batchify(test_data, args.batch_size)
    
    # Create batches for training
    def get_batch(source, i, bptt):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target
    
    return train_data, val_data, test_data, vocab, get_batch


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
    
    for batch, i in enumerate(pbar):
        data, targets = get_batch(train_data, i, args.bptt)
        
        # Clear gradients only at the beginning of accumulation steps
        if batch % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        try:
            # Forward pass
            output = model(data)
            
            # Check for NaN or inf in output
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"WARNING: NaN or inf in model output. Skipping batch.")
                continue
                
            # Calculate loss
            loss = criterion(output.view(-1, ntokens), targets)
            
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
            print(f"Skipping problematic batch")
            optimizer.zero_grad()
            continue
    
    # Update learning rate scheduler at the end of epoch if not updated per step
    if not args.scheduler_update_every_step:
        scheduler.step()
    
    # Validate after each epoch
    val_loss = evaluate_language_model(model, val_data, vocab, get_batch, criterion, args)
    print(f'| End of epoch {epoch:3d} | valid loss {val_loss:5.2f} | valid ppl {math.exp(min(val_loss, 20)):8.2f}')
    
    return val_loss


def evaluate_language_model(model, data, vocab, get_batch, criterion, args):
    """Evaluate a language model on validation or test data."""
    model.eval()
    total_loss = 0.0
    ntokens = len(vocab)
    
    with torch.no_grad():
        for i in tqdm(range(0, data.size(0) - 1, args.bptt), 
                     desc="Evaluating"):
            data_batch, targets = get_batch(data, i, args.bptt)
            
            try:
                output = model(data_batch)
                
                # Skip batches with NaN or inf values
                if torch.isnan(output).any() or torch.isinf(output).any():
                    continue
                    
                loss = criterion(output.view(-1, ntokens), targets).item()
                total_loss += loss * targets.size(0)
            except RuntimeError:
                # Skip problematic batches
                continue
    
    return total_loss / (data.size(0) - 1)


def train_classifier(model, train_dataloader, optimizer, criterion, scheduler, args, epoch):
    """Train a text classifier on IMDB."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs}")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(args.device), target.to(args.device)
        
        # Clear gradients only at the beginning of accumulation steps
        if batch_idx % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        try:
            # Forward pass
            output = model(data)
            
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
            print(f"Skipping problematic batch")
            optimizer.zero_grad()
            continue
    
    # Update learning rate scheduler at the end of epoch if not updated per step
    if not args.scheduler_update_every_step:
        scheduler.step()
    
    avg_loss = total_loss / len(train_dataloader)
    avg_acc = correct / total
    
    return avg_loss, avg_acc


def evaluate_classifier(model, test_dataloader, criterion, args):
    """Evaluate a text classifier on test data."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_dataloader, desc="Evaluating"):
            data, target = data.to(args.device), target.to(args.device)
            
            try:
                # Forward pass
                output = model(data)
                
                # Skip batches with NaN or inf values
                if torch.isnan(output).any() or torch.isinf(output).any():
                    continue
                    
                # Calculate loss
                loss = criterion(output, target)
                total_loss += loss.item() * target.size(0)
                
                # Calculate accuracy
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                
            except RuntimeError:
                # Skip problematic batches
                continue
    
    # Print test results
    test_loss = total_loss / total if total > 0 else float('inf')
    test_acc = 100 * correct / total if total > 0 else 0
    print(f'| Test loss {test_loss:.2f} | Test accuracy {test_acc:.2f}%')
    
    return test_loss, test_acc


def train(args):
    """Main training function."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print device info
    print(f"Using device: {args.device}")
    
    # Set up data
    if args.task == 'lm':
        print("Loading language modeling data...")
        train_data, val_data, test_data, vocab, get_batch = get_language_modeling_data(args)
        ntokens = len(vocab)
        criterion = nn.CrossEntropyLoss()
    else:  # classification
        print("Loading classification data...")
        train_dataloader, test_dataloader, vocab = get_classification_data(args)
        ntokens = len(vocab)
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
    
    # Set up optimizer and scheduler
    print("Setting up optimizer and scheduler...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Use Cosine Annealing scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * (len(train_data) // args.bptt // args.gradient_accumulation_steps if args.task == 'lm' 
                            else len(train_dataloader) // args.gradient_accumulation_steps),
        eta_min=args.lr / 100
    )
    
    # Print model info
    print(f"Model: {args.transformer_type.upper()} Transformer")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1000000:.2f}M")
    print(f"Orthogonality Mode: {args.orth_mode}")
    if args.orth_mode == 'regularize':
        print(f"Orthogonality Penalty Weight: {args.orth_penalty_weight}")
    
    # Training loop
    best_val_loss = float('inf')
    
    try:
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            
            # Train for one epoch
            if args.task == 'lm':
                train_language_model(model, train_data, val_data, vocab, get_batch, optimizer, criterion, scheduler, args, epoch)
                val_loss = evaluate_language_model(model, val_data, vocab, get_batch, criterion, args)
            else:  # classification
                train_classifier(model, train_dataloader, optimizer, criterion, scheduler, args, epoch)
                val_loss, _ = evaluate_classifier(model, test_dataloader, criterion, args)
            
            # Save model if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.output_dir, args.save))
                print(f'| Saving model to {os.path.join(args.output_dir, args.save)}')
    
    except KeyboardInterrupt:
        print('| Keyboard interrupt - stopping training')
    
    # Load best model
    model_path = os.path.join(args.output_dir, args.save)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    
    # Final evaluation
    if args.task == 'lm':
        test_loss = evaluate_language_model(model, test_data, vocab, get_batch, criterion, args)
        print(f'| End of training | test loss {test_loss:5.2f} | test ppl {math.exp(min(test_loss, 20)):8.2f}')
    else:  # classification
        test_loss, test_acc = evaluate_classifier(model, test_dataloader, criterion, args)
        print(f'| End of training | test loss {test_loss:5.2f} | test accuracy {test_acc:5.2f}%')


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
    parser.add_argument('--orth_mode', type=str, default='init', choices=['init', 'regularize', 'strict'],
                        help='How to enforce orthogonality')
    parser.add_argument('--orth_penalty_weight', type=float, default=0.001, 
                        help='Weight for orthogonality penalty (used in regularize mode)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--bptt', type=int, default=35, help='Sequence length for language modeling')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Max sequence length for classification')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Steps for gradient accumulation')
    parser.add_argument('--scheduler_update_every_step', action='store_true', help='Update scheduler every step')
    parser.add_argument('--vocab_cutoff', type=int, default=10000, help='Limit vocabulary size (0 for no limit)')
    parser.add_argument('--log_interval', type=int, default=200, help='Report interval')
    parser.add_argument('--save', type=str, default='model.pt', help='Model save filename')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--num_workers', type=int, default=2, help='Data loader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    train(args)

import os
import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import torchtext
# from torchtext.datasets import WikiText2, IMDB
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
import matplotlib.pyplot as plt
from tqdm import tqdm

from orthogonal_linear import OrthogonalLinear
from ospa_attention import OSPAMultiHeadAttention
from ospa_transformer import OSPATransformer
from baseline_models import VanillaTransformer, LinformerTransformer


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models."""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container for a Transformer model with token embeddings and task-specific heads."""
    
    def __init__(self, transformer_type, vocab_size, d_model, nhead, nlayers, 
                 dropout=0.5, dim_feedforward=2048, orth_mode="regularize", 
                 orth_penalty_weight=0.01, task="lm"):
        super(TransformerModel, self).__init__()
        self.transformer_type = transformer_type
        self.d_model = d_model
        self.orth_mode = orth_mode
        self.orth_penalty_weight = orth_penalty_weight
        self.task = task
        
        # Token embedding and positional encoding
        self.encoder = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Initialize the appropriate transformer architecture
        if transformer_type == "ospa":
            self.transformer = OSPATransformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=nlayers,
                num_decoder_layers=0,  # Use only encoder for these tasks
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                orth_mode=orth_mode,
                orth_penalty_weight=orth_penalty_weight
            )
        elif transformer_type == "vanilla":
            self.transformer = VanillaTransformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=nlayers,
                num_decoder_layers=0,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
        elif transformer_type == "linformer":
            self.transformer = LinformerTransformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=nlayers,
                num_decoder_layers=0,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                k=256  # Projection dimension for Linformer
            )
        else:
            raise ValueError(f"Unknown transformer type: {transformer_type}")
        
        # Task-specific output heads
        if task == "lm":  # Language modeling
            self.output_layer = nn.Linear(d_model, vocab_size)
        elif task == "classification":  # Text classification
            self.output_layer = nn.Linear(d_model, 2)  # Binary classification (e.g., IMDB)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: Token indices [seq_len, batch_size]
            src_mask: Mask for self-attention [seq_len, seq_len]
            src_key_padding_mask: Mask for padding tokens [batch_size, seq_len]
        """
        # Token embedding and positional encoding
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Pass through transformer (only using encoder part)
        if hasattr(self.transformer, 'encoder'):
            output = self.transformer.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        else:
            output = self.transformer(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        # Apply output layer based on task
        if self.task == "lm":
            output = self.output_layer(output)
        elif self.task == "classification":
            # For classification, use the representation of the first token
            output = self.output_layer(output[0])
        
        return output
    
    def get_orthogonality_penalty(self):
        """Get the orthogonality penalty if using OSPA."""
        if self.transformer_type == "ospa":
            return self.transformer.get_orthogonality_penalty()
        return 0.0


def get_language_modeling_data(args):
    """Prepare data for language modeling task (WikiText-2) using only the datasets library."""
    import torch
    from datasets import load_dataset
    
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
    for text in wikitext['train']['text']:
        if text.strip():  # Skip empty lines
            for token in tokenize(text):
                if token not in vocab:
                    vocab[token] = len(vocab)
                word_count[token] = word_count.get(token, 0) + 1
    
    # Process datasets
    def data_process(raw_text_iter):
        data = []
        for text in raw_text_iter:
            if text.strip():  # Skip empty lines
                tokens = torch.tensor([vocab.get(token, vocab['<unk>']) for token in tokenize(text)], 
                                    dtype=torch.long)
                if len(tokens) > 0:
                    data.append(tokens)
        return torch.cat(data)
    
    train_data = data_process(wikitext['train']['text'])
    val_data = data_process(wikitext['validation']['text'])
    test_data = data_process(wikitext['test']['text'])
    
    # Batch data
    def batchify(data, batch_size):
        # Work out how cleanly we can divide the dataset into batch_size parts
        nbatch = data.size(0) // batch_size
        # Trim off any extra elements that wouldn't cleanly fit
        data = data.narrow(0, 0, nbatch * batch_size)
        # Evenly divide the data across the batch_size batches
        data = data.view(batch_size, -1).t().contiguous()
        return data.to(args.device)
    
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
    """Prepare data for text classification task (IMDB) using only the datasets library."""
    import torch
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    
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
    for i, example in enumerate(imdb['train']):
        if i >= 5000:  # Limit processing for speed
            break
        text = example['text']
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
            word_count[token] = word_count.get(token, 0) + 1
    
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
    train_dataloader = DataLoader(
        imdb['train'], 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )
    test_dataloader = DataLoader(
        imdb['test'], 
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )
    
    return train_dataloader, test_dataloader, vocab


def train_language_model(model, train_data, val_data, vocab, get_batch, optimizer, criterion, scheduler, args, epoch):
    """Train a language model on WikiText-2."""
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(vocab)
    
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i, args.bptt)
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        
        # Add orthogonality penalty if using OSPA with regularize mode
        if model.transformer_type == "ospa" and model.orth_mode == "regularize":
            orth_penalty = model.get_orthogonality_penalty()
            loss = loss + args.orth_penalty_weight * orth_penalty
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log progress
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_data) // args.bptt:5d} batches | '
                  f'lr {scheduler.get_last_lr()[0]:02.6f} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}')
            total_loss = 0
            start_time = time.time()
    
    # Validate after each epoch
    val_loss = evaluate_language_model(model, val_data, vocab, get_batch, criterion, args)
    print(f'| End of epoch {epoch:3d} | valid loss {val_loss:5.2f} | valid ppl {math.exp(val_loss):8.2f}')
    
    return val_loss


def evaluate_language_model(model, data, vocab, get_batch, criterion, args):
    """Evaluate a language model on validation or test data."""
    model.eval()
    total_loss = 0.
    ntokens = len(vocab)
    
    with torch.no_grad():
        for i in range(0, data.size(0) - 1, args.bptt):
            data_batch, targets = get_batch(data, i, args.bptt)
            output = model(data_batch)
            total_loss += criterion(output.view(-1, ntokens), targets).item() * targets.size(0)
    
    return total_loss / (data.size(0) - 1)


def train_classifier(model, train_dataloader, optimizer, criterion, scheduler, args, epoch):
    """Train a text classifier on IMDB."""
    model.train()
    total_loss = 0.
    correct = 0
    total = 0
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(args.device), target.to(args.device)
        
        # Create padding mask for transformer
        padding_mask = (data.t() == vocab['<pad>']).to(args.device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data, src_key_padding_mask=padding_mask)
        loss = criterion(output, target)
        
        # Add orthogonality penalty if using OSPA with regularize mode
        if model.transformer_type == "ospa" and model.orth_mode == "regularize":
            orth_penalty = model.get_orthogonality_penalty()
            loss = loss + args.orth_penalty_weight * orth_penalty
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
        
        total_loss += loss.item()
        
        # Log progress
        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print(f'| epoch {epoch:3d} | {batch_idx:5d}/{len(train_dataloader):5d} batches | '
                  f'lr {scheduler.get_last_lr()[0]:02.6f} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | '
                  f'loss {cur_loss:5.2f} | acc {100 * correct / total:.2f}%')
            total_loss = 0
            correct = 0
            total = 0
            start_time = time.time()
    
    return total_loss / len(train_dataloader)


def evaluate_classifier(model, test_dataloader, criterion, args):
    """Evaluate a text classifier on test data."""
    model.eval()
    total_loss = 0.
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(args.device), target.to(args.device)
            
            # Create padding mask for transformer
            padding_mask = (data.t() == vocab['<pad>']).to(args.device)
            
            output = model(data, src_key_padding_mask=padding_mask)
            total_loss += criterion(output, target).item() * target.size(0)
            
            # Calculate accuracy
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    # Print test results
    test_loss = total_loss / total
    test_acc = 100 * correct / total
    print(f'| Test loss {test_loss:.2f} | Test accuracy {test_acc:.2f}%')
    
    return test_loss, test_acc


def analyze_head_similarity(model, dataloader, args):
    """Analyze the similarity between attention heads."""
    model.eval()
    
    # Get sample batch
    for data, _ in dataloader:
        data = data.to(args.device)
        break
    
    # Extract attention weights from model
    attentions = []
    
    def hook_fn(module, input, output):
        attn_output, attn_weights = output
        attentions.append(attn_weights)
    
    # Register hooks for all attention modules
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, OSPAMultiHeadAttention):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass to get attention weights
    with torch.no_grad():
        model(data)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate cosine similarity between heads
    for layer_idx, attn_weights in enumerate(attentions):
        # attn_weights shape: [batch_size, num_heads, seq_len, seq_len]
        attn_weights = attn_weights.mean(dim=0)  # Average over batch
        
        # Calculate cosine similarity
        num_heads = attn_weights.size(0)
        cosine_sim = torch.zeros((num_heads, num_heads), device=args.device)
        
        for i in range(num_heads):
            for j in range(num_heads):
                vec_i = attn_weights[i].reshape(-1)
                vec_j = attn_weights[j].reshape(-1)
                cosine_sim[i, j] = F.cosine_similarity(vec_i.unsqueeze(0), vec_j.unsqueeze(0))
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(cosine_sim.cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(f'Layer {layer_idx+1} Attention Head Similarity')
        plt.xlabel('Head Index')
        plt.ylabel('Head Index')
        plt.savefig(f'head_similarity_layer{layer_idx+1}.png')
        plt.close()


def track_orthogonality(model, train_dataloader, args):
    """Track orthogonality deviation during training."""
    # Initialize arrays to store orthogonality metrics
    num_steps = min(100, len(train_dataloader))
    steps = np.zeros(num_steps)
    orth_deviations = np.zeros(num_steps)
    
    # Get orthogonal linear layers
    orth_layers = []
    for name, module in model.named_modules():
        if isinstance(module, OrthogonalLinear):
            orth_layers.append(module)
    
    # Training loop with tracking
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    for step, (data, target) in enumerate(train_dataloader):
        if step >= num_steps:
            break
            
        data, target = data.to(args.device), target.to(args.device)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Calculate orthogonality deviation
        orth_dev = 0.0
        for layer in orth_layers:
            if layer.is_transposed:
                prod = layer.weight @ layer.weight.t()
                identity = torch.eye(layer.out_features, device=layer.weight.device)
            else:
                prod = layer.weight.t() @ layer.weight
                identity = torch.eye(layer.in_features, device=layer.weight.device)
            orth_dev += torch.norm(prod - identity, p='fro').item()
        orth_dev /= len(orth_layers)
        
        # Store metrics
        steps[step] = step
        orth_deviations[step] = orth_dev
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Plot orthogonality deviation
    plt.figure(figsize=(10, 6))
    plt.plot(steps, orth_deviations)
    plt.xlabel('Training Step')
    plt.ylabel('Orthogonality Deviation')
    plt.title('Orthogonality Deviation During Training')
    plt.savefig('orthogonality_deviation.png')
    plt.close()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Orthogonal Subspace Projection Attention')
    
#     # Model configuration
#     parser.add_argument('--transformer_type', type=str, default='ospa', choices=['ospa', 'vanilla', 'linformer'],
#                         help='Type of transformer architecture')
#     parser.add_argument('--task', type=str, default='lm', choices=['lm', 'classification'],
#                         help='Task: language modeling (lm) or text classification')
#     parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
#     parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
#     parser.add_argument('--nlayers', type=int, default=6, help='Number of transformer layers')
#     parser.add_argument('--dim_feedforward', type=int, default=2048, help='Dimension of feedforward network')
    
#     # Orthogonality parameters
#     parser.add_argument('--orth_mode', type=str, default='regularize', choices=['init', 'regularize', 'strict'],
#                         help='How to enforce orthogonality')
#     parser.add_argument('--orth_penalty_weight', type=float, default=0.01, 
#                         help='Weight for orthogonality penalty (used in regularize mode)')
    
#     # Training parameters
#     parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
#     parser.add_argument('--bptt', type=int, default=35, help='Sequence length for language modeling')
#     parser.add_argument('--max_seq_len', type=int, default=256, help='Max sequence length for classification')
#     parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
#     parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
#     parser.add_argument('--lr', type=float, default=5.0, help='Initial learning rate')
#     parser.add_argument('--clip', type=float, default=0.25, help='Gradient clipping')
#     parser.add_argument('--log_interval', type=int, default=200, help='Report interval')
    
#     # Other parameters
#     parser.add_argument('--seed', type=int, default=1111, help='Random seed')
#     parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
#     parser.add_argument('--save', type=str, default='model.pt', help='Path to save model')
#     parser.add_argument('--analyze', action='store_true', help='Run analysis after training')
    
#     args = parser.parse_args()
    
#     # Set device
#     args.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    
#     # Set random seed
#     torch.manual_seed(args.seed)
    
#     # Set up data
#     if args.task == 'lm':
#         train_data, val_data, test_data, vocab, get_batch = get_language_modeling_data(args)
#         ntokens = len(vocab)
#         criterion = nn.CrossEntropyLoss()
#     else:  # classification
#         train_dataloader, test_dataloader, vocab = get_classification_data(args)
#         ntokens = len(vocab)
#         criterion = nn.CrossEntropyLoss()
    
#     # Create model
#     model = TransformerModel(
#         transformer_type=args.transformer_type,
#         vocab_size=ntokens,
#         d_model=args.d_model,
#         nhead=args.nhead,
#         nlayers=args.nlayers,
#         dropout=args.dropout,
#         dim_feedforward=args.dim_feedforward,
#         orth_mode=args.orth_mode,
#         orth_penalty_weight=args.orth_penalty_weight,
#         task=args.task
#     ).to(args.device)
    
#     # Set up optimizer and scheduler
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1.0, gamma=0.95)
    
def train():
    """Main training function."""
    # Training loop
    best_val_loss = float('inf')
    
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            
            # Train for one epoch
            if args.task == 'lm':
                train_language_model(model, train_data, val_data, vocab, get_batch, optimizer, criterion, scheduler, args, epoch)
                val_loss = evaluate_language_model(model, val_data, vocab, get_batch, criterion, args)
            else:  # classification
                train_classifier(model, train_dataloader, optimizer, criterion, scheduler, args, epoch)
                val_loss, _ = evaluate_classifier(model, test_dataloader, criterion, args)
            
            # Update learning rate
            scheduler.step()
            
            # Save model if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), args.save)
                print(f'| Saving model to {args.save}')
    
    except KeyboardInterrupt:
        print('| Keyboard interrupt - stopping training')
    
    # Load best model
    model.load_state_dict(torch.load(args.save))
    
    # Final evaluation
    if args.task == 'lm':
        test_loss = evaluate_language_model(model, test_data, vocab, get_batch, criterion, args)
        print(f'| End of training | test loss {test_loss:5.2f} | test ppl {math.exp(test_loss):8.2f}')
    else:  # classification
        test_loss, test_acc = evaluate_classifier(model, test_dataloader, criterion, args)
        print(f'| End of training | test loss {test_loss:5.2f} | test accuracy {test_acc:5.2f}%')
    
    # Run analysis if requested
    if args.analyze and args.transformer_type == 'ospa':
        print('| Running head similarity analysis...')
        if args.task == 'classification':
            analyze_head_similarity(model, test_dataloader, args)
            track_orthogonality(model, train_dataloader, args)
        else:
            # For language modeling, create a dummy dataloader
            dummy_data = torch.randint(0, ntokens, (args.bptt, args.batch_size)).to(args.device)
            dummy_target = torch.randint(0, ntokens, (args.batch_size,)).to(args.device)
            dummy_dataloader = [(dummy_data, dummy_target) for _ in range(10)]
            analyze_head_similarity(model, dummy_dataloader, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Orthogonal Subspace Projection Attention')
    
    # Model configuration
    parser.add_argument('--transformer_type', type=str, default='ospa', choices=['ospa', 'vanilla', 'linformer'],
                        help='Type of transformer architecture')
    parser.add_argument('--task', type=str, default='lm', choices=['lm', 'classification'],
                        help='Task: language modeling (lm) or text classification')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--nlayers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Dimension of feedforward network')
    
    # Orthogonality parameters
    parser.add_argument('--orth_mode', type=str, default='regularize', choices=['init', 'regularize', 'strict'],
                        help='How to enforce orthogonality')
    parser.add_argument('--orth_penalty_weight', type=float, default=0.01, 
                        help='Weight for orthogonality penalty (used in regularize mode)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--bptt', type=int, default=35, help='Sequence length for language modeling')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Max sequence length for classification')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5.0, help='Initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25, help='Gradient clipping')
    parser.add_argument('--log_interval', type=int, default=200, help='Report interval')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=1111, help='Random seed')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--save', type=str, default='model.pt', help='Path to save model')
    parser.add_argument('--analyze', action='store_true', help='Run analysis after training')
    
    args = parser.parse_args()
    
    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Set up data
    if args.task == 'lm':
        train_data, val_data, test_data, vocab, get_batch = get_language_modeling_data(args)
        ntokens = len(vocab)
        criterion = nn.CrossEntropyLoss()
    else:  # classification
        train_dataloader, test_dataloader, vocab = get_classification_data(args)
        ntokens = len(vocab)
        criterion = nn.CrossEntropyLoss()
    
    # Create model
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1.0, gamma=0.95)
    
    # Print model
    print(f"Model: {args.transformer_type.upper()} Transformer")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1000000:.2f}M")
    print(f"Orthogonality Mode: {args.orth_mode}")
    if args.orth_mode == 'regularize':
        print(f"Orthogonality Penalty Weight: {args.orth_penalty_weight}")
    
    # Train the model
    train()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Orthogonal Subspace Projection Attention')
    
#     # Model configuration
#     parser.add_argument('--transformer_type', type=str, default='ospa', choices=['ospa', 'vanilla', 'linformer'],
#                         help='Type of transformer architecture')
#     parser.add_argument('--task', type=str, default='lm', choices=['lm', 'classification'],
#                         help='Task: language modeling (lm) or text classification')
#     parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
#     parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
#     parser.add_argument('--nlayers', type=int, default=6, help='Number of transformer layers')
#     parser.add_argument('--dim_feedforward', type=int, default=2048, help='Dimension of feedforward network')
    
#     # Orthogonality parameters
#     parser.add_argument('--orth_mode', type=str, default='regularize', choices=['init', 'regularize', 'strict'],
#                         help='How to enforce orthogonality')
#     parser.add_argument('--orth_penalty_weight', type=float, default=0.01, 
#                         help='Weight for orthogonality penalty (used in regularize mode)')
    
#     # Training parameters
#     parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
#     parser.add_argument('--bptt', type=int, default=35, help='Sequence length for language modeling')
#     parser.add_argument('--max_seq_len', type=int, default=256, help='Max sequence length for classification')
#     parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
#     parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
#     parser.add_argument('--lr', type=float, default=5.0, help='Initial learning rate')
#     parser.add_argument('--clip', type=float, default=0.25, help='Gradient clipping')
#     parser.add_argument('--log_interval', type=int, default=200, help='Report interval')
    
#     # Other parameters
#     parser.add_argument('--seed', type=int, default=1111, help='Random seed')
#     parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
#     parser.add_argument('--save', type=str, default='model.pt', help='Path to save model')
#     parser.add_argument('--analyze', action='store_true', help='Run analysis after training')
    
#     args = parser.parse_args()
    
#     # Set device
#     args.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    
#     # Set random seed
#     torch.manual_seed(args.seed)
    
#     # Set up data
#     if args.task == 'lm':
#         train_data, val_data, test_data, vocab, get_batch = get_language_modeling_data(args)
#         ntokens = len(vocab)
#         criterion = nn.CrossEntropyLoss()
#     else:  # classification
#         train_dataloader, test_dataloader, vocab = get_classification_data(args)
#         ntokens = len(vocab)
#         criterion = nn.CrossEntropyLoss()
    
#     # Create model
#     model = TransformerModel(
#         transformer_type=args.transformer_type,
#         vocab_size=ntokens,
#         d_model=args.d_model,
#         nhead=args.nhead,
#         nlayers=args.nlayers,
#         dropout=args.dropout,
#         dim_feedforward=args.dim_feedforward,
#         orth_mode=args.orth_mode,
#         orth_penalty_weight=args.orth_penalty_weight,
#         task=args.task
#     ).to(args.device)
    
#     # Set up optimizer and scheduler
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1.0, gamma=0.95)
    
#     # Print model
#     print(f"Model: {args.transformer_type.upper()} Transformer")
#     print(f"Parameters: {sum(p.numel() for p in model.parameters())/1000000:.2f}M")
#     print(f"Orthogonality Mode: {args.orth_mode}")
#     if args.orth_mode == 'regularize':
#         print(f"Orthogonality Penalty Weight: {args.orth_penalty_weight}")
    
#     # Train the model
#     train()
# train_ospa.py
import os
import sys
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
from tqdm.auto import tqdm # Use auto version for better notebook compatibility
import logging

# --- Standard Libraries ---
try:
    from transformers import AutoTokenizer, get_scheduler
    from datasets import load_dataset
    import evaluate # Hugging Face Evaluate library for metrics
except ImportError:
    print("Error: Required libraries not found.")
    print("Please install: pip install torch transformers datasets evaluate scikit-learn tqdm")
    exit(1)

# --- Local Imports ---
try:
    # Assuming improved_transformer_model.py defines TransformerModel compatible with HF inputs
    from improved_transformer_model import TransformerModel
except ImportError:
    print("Error: improved_transformer_model.py not found.")
    print("Make sure it's in the same directory or your Python path.")
    print("It needs to be adapted to handle 'input_ids' and 'attention_mask'.")
    exit(1)

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Log to console
)
logger = logging.getLogger(__name__)

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(args):
    """Loads and preprocesses data for LM or GLUE tasks using transformers."""
    logger.info(f"--- Loading Tokenizer: {args.tokenizer_name} ---")
    try:
        # Using use_fast=True is generally recommended for performance
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
        # Add pad token if tokenizer doesn't have one (needed for padding)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                logger.warning(f"Tokenizer {args.tokenizer_name} missing pad token, using eos token {tokenizer.eos_token} as pad token.")
                tokenizer.pad_token = tokenizer.eos_token
            else:
                 # Add a new pad token if no eos token either
                 logger.warning(f"Tokenizer {args.tokenizer_name} missing pad token and eos token. Adding new [PAD] token.")
                 tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                 # Important: The model embedding layer needs resizing after adding tokens
                 args.resize_embedding = True # Flag to resize model embeddings later

    except Exception as e:
        logger.error(f"Failed to load tokenizer '{args.tokenizer_name}': {e}")
        exit(1)

    # Standard GLUE task keys
    task_to_keys = {
        "cola": ("sentence", None), "sst2": ("sentence", None),
        "mrpc": ("sentence1", "sentence2"), "qqp": ("question1", "question2"),
        "stsb": ("sentence1", "sentence2"), "mnli": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"), "rte": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    if args.task == 'glue':
        # --- GLUE Task Processing ---
        glue_task = args.glue_task.lower()
        if glue_task not in task_to_keys:
             logger.error(f"Invalid GLUE task: {glue_task}. Valid: {list(task_to_keys.keys())}")
             exit(1)

        logger.info(f"--- Loading and Preprocessing GLUE task: {glue_task} ---")
        try:
            raw_datasets = load_dataset("glue", glue_task)
        except Exception as e:
            logger.error(f"Failed to load GLUE dataset '{glue_task}': {e}")
            exit(1)

        sentence1_key, sentence2_key = task_to_keys[glue_task]

        # Determine label info
        args.is_regression = glue_task == "stsb"
        if not args.is_regression:
            label_list = raw_datasets["train"].features["label"].names
            args.num_labels = len(label_list)
            logger.info(f"Task: {glue_task} (Classification), Num Labels: {args.num_labels}")
        else:
            args.num_labels = 1
            logger.info(f"Task: {glue_task} (Regression), Num Labels: {args.num_labels}")

        # Preprocessing function for tokenizer
        def preprocess_glue(examples):
            if sentence2_key is None: # Single sentence task
                return tokenizer(examples[sentence1_key], truncation=True, padding="max_length", max_length=args.max_seq_len)
            else: # Sentence pair task
                return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding="max_length", max_length=args.max_seq_len)

        # Columns to remove after tokenization
        remove_cols = ["idx"] + ([sentence1_key] if sentence1_key else []) + ([sentence2_key] if sentence2_key else [])

        # Apply tokenization
        try:
            processed_datasets = raw_datasets.map(
                preprocess_glue,
                batched=True,
                remove_columns=remove_cols,
                desc=f"Tokenizing GLUE/{glue_task}"
            )
        except Exception as e:
            logger.error(f"Error during tokenization: {e}")
            exit(1)

        # Rename label column for consistency
        processed_datasets = processed_datasets.rename_column("label", "labels")
        processed_datasets.set_format("torch")

        # Select splits (handle MNLI special case for validation)
        train_dataset = processed_datasets["train"]
        validation_key = "validation_matched" if glue_task == "mnli" else "validation"
        eval_dataset = processed_datasets[validation_key]
        # Use validation set as test set (no public test labels for GLUE)
        test_dataset = eval_dataset
        # MNLI also has a mismatched validation set you might want to evaluate separately
        if glue_task == "mnli":
            test_mismatched_dataset = processed_datasets["validation_mismatched"]
            logger.info("Using MNLI validation_matched for validation, validation_mismatched available for testing.")
            return {"train": train_dataset, "validation": eval_dataset, "test": test_dataset, "test_mismatched": test_mismatched_dataset}, tokenizer
        else:
            logger.info(f"Using validation split '{validation_key}' for evaluation and testing.")
            return {"train": train_dataset, "validation": eval_dataset, "test": test_dataset}, tokenizer


    elif args.task == 'lm':
        # --- Language Modeling Task Processing ---
        logger.info(f"--- Loading and Preprocessing LM dataset: {args.lm_dataset_name} ({args.lm_dataset_config}) ---")
        try:
            # Load dataset (e.g., wikitext, wikitext-103-raw-v1)
            raw_datasets = load_dataset(args.lm_dataset_name, args.lm_dataset_config)
        except Exception as e:
             logger.error(f"Failed to load LM dataset '{args.lm_dataset_name}' config '{args.lm_dataset_config}': {e}")
             exit(1)

        # Assume the main text is in the 'text' column
        text_column_name = "text"
        if text_column_name not in raw_datasets["train"].column_names:
            logger.error(f"Expected column '{text_column_name}' not found in LM dataset. Available columns: {raw_datasets['train'].column_names}")
            exit(1)

        # Set model output size to vocab size
        args.num_labels = tokenizer.vocab_size
        args.is_regression = False
        logger.info(f"Task: Language Modeling, Vocab Size: {args.num_labels}")

        # Tokenization function for LM
        def tokenize_lm(examples):
            # Tokenize without padding initially
            return tokenizer(examples[text_column_name], truncation=False)

        logger.info("Tokenizing LM dataset...")
        try:
            tokenized_datasets = raw_datasets.map(
                tokenize_lm,
                batched=True,
                remove_columns=[text_column_name], # Remove original text column
                desc="Tokenizing LM data"
            )
        except Exception as e:
            logger.error(f"Error during LM tokenization: {e}")
            exit(1)

        # Group texts into blocks of fixed size (e.g., max_seq_len)
        block_size = args.max_seq_len
        if block_size is None:
            # Determine block_size if not set (e.g., tokenizer model_max_length)
            block_size = tokenizer.model_max_length
            if block_size > 1024: # Cap block size for memory reasons
                logger.warning(f"Tokenizer max length {block_size} is large, capping block size to 1024.")
                block_size = 1024
            args.max_seq_len = block_size # Update args if derived
            logger.info(f"Using derived block size for LM: {block_size}")


        logger.info(f"Grouping LM texts into blocks of size {block_size}...")
        def group_texts(examples):
            # Concatenate all texts
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # Drop the small remainder that is smaller than block_size
            total_length = (total_length // block_size) * block_size
            # Split by chunks of block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            # Create labels by shifting input_ids
            result["labels"] = result["input_ids"].copy()
            return result

        try:
            lm_datasets = tokenized_datasets.map(group_texts, batched=True, desc="Grouping LM texts")
        except Exception as e:
             logger.error(f"Error grouping LM texts: {e}")
             exit(1)

        # Prepare dataset dictionary
        train_dataset = lm_datasets["train"]
        # Use appropriate validation/test splits
        eval_dataset = lm_datasets.get("validation", lm_datasets.get("test"))
        test_dataset = lm_datasets.get("test", lm_datasets.get("validation"))

        if eval_dataset is None or test_dataset is None:
             logger.error("Could not find suitable validation/test splits in LM dataset.")
             exit(1)

        logger.info("LM Data Processing Complete.")
        return {"train": train_dataset, "validation": eval_dataset, "test": test_dataset}, tokenizer

    else:
        raise ValueError(f"Unsupported task type specified: {args.task}")

# --- Training Epoch Function ---
def train_epoch(model, dataloader, optimizer, criterion, scheduler, args, epoch, tokenizer):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs} Training", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        try:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            labels = batch["labels"]
        except Exception as e:
            logger.warning(f"Could not move batch {batch_idx} to device {args.device}: {e}. Skipping batch.")
            continue

        # --- Gradient Accumulation: Zero Grads ---
        if batch_idx % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        try:
            # --- Forward Pass ---
            # Model expects input_ids, attention_mask (implicitly handled by padding), maybe token_type_ids
            # Crucially, pass attention_mask to model if it uses it internally for padding!
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch.get('attention_mask')) # Pass mask if model uses it

            # --- Check for NaNs/Infs ---
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                logger.warning(f"NaN or Inf detected in model output (Epoch {epoch}, Batch {batch_idx}). Skipping.")
                continue

            # --- Loss Calculation ---
            loss = None
            if args.task == 'lm':
                # Reshape for CrossEntropyLoss: [Batch*SeqLen, VocabSize] vs [Batch*SeqLen]
                loss = criterion(outputs.view(-1, model.vocab_size), labels.view(-1))
            elif args.task == 'glue':
                if args.is_regression: # STSB
                    loss = criterion(outputs.squeeze(), labels.float()) # MSELoss expects [Batch], [Batch]
                else: # Classification
                    loss = criterion(outputs, labels) # CrossEntropyLoss expects [Batch, N_Labels], [Batch]
            else:
                 # Should be caught earlier
                 raise ValueError("Invalid task type during loss calculation")


            if loss is None: continue # Skip if loss couldn't be calculated

            # --- Add Orthogonality Penalty ---
            if model.transformer_type == "ospa" and model.orth_mode == "regularize":
                orth_penalty = model.get_orthogonality_penalty()
                if not (torch.isnan(orth_penalty).any() or torch.isinf(orth_penalty).any()):
                     loss = loss + args.orth_penalty_weight * orth_penalty
                else:
                     logger.warning(f"NaN/Inf in orth penalty (Epoch {epoch}, Batch {batch_idx}). Penalty not added.")

            # --- Scale Loss for Accumulation ---
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # --- Backward Pass ---
            loss.backward()

            # --- Optimizer Step ---
            is_accumulation_step = (batch_idx + 1) % args.gradient_accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == num_batches
            if is_accumulation_step or is_last_batch:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                # Optimizer step
                optimizer.step()
                # LR Scheduler step (if per-step)
                if args.scheduler_update_every_step:
                     if scheduler and hasattr(scheduler, 'step'):
                         scheduler.step()

            # --- Logging ---
            batch_loss = loss.item() * args.gradient_accumulation_steps # Log unscaled loss for the batch
            total_loss += batch_loss
            # Update progress bar
            progress_bar.set_postfix(loss=f"{batch_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}" if scheduler else f"{optimizer.param_groups[0]['lr']:.6f}")


        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error(f"CUDA OOM encountered in training (Epoch {epoch}, Batch {batch_idx}). Try reducing batch size or model size.")
                # Attempt to clear cache (may or may not help)
                # torch.cuda.empty_cache()
                raise e # Stop training on OOM
            else:
                logger.error(f"Runtime error during training (Epoch {epoch}, Batch {batch_idx}): {e}")
                # Skip batch and try to continue
                optimizer.zero_grad(set_to_none=True)
                continue

    # --- End of Epoch ---
    # Step scheduler if updating per epoch
    if not args.scheduler_update_every_step:
         if scheduler and hasattr(scheduler, 'step'):
             scheduler.step()

    avg_epoch_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    logger.info(f"Epoch {epoch} Training Average Loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss


# --- Evaluation Function ---
def evaluate(model, dataloader, criterion, args, eval_metric=None, eval_type="Validation"):
    """Evaluates the model on a given dataloader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_refs = []
    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, desc=f"{eval_type} Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            try:
                batch = {k: v.to(args.device) for k, v in batch.items()}
                labels = batch["labels"]
            except Exception as e:
                logger.warning(f"Could not move batch to device {args.device} during {eval_type}: {e}. Skipping.")
                continue

            try:
                # --- Forward Pass ---
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch.get('attention_mask'))

                # --- Check for NaNs/Infs ---
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    logger.warning(f"NaN or Inf detected in {eval_type} output. Skipping batch.")
                    continue

                # --- Loss Calculation ---
                loss = None
                if args.task == 'lm':
                    loss = criterion(outputs.view(-1, model.vocab_size), labels.view(-1))
                    # For LM, we typically don't compute accuracy/F1 etc. during training eval
                elif args.task == 'glue':
                    if args.is_regression:
                        loss = criterion(outputs.squeeze(), labels.float())
                        predictions = outputs.squeeze() # Regression predictions
                    else:
                        loss = criterion(outputs, labels)
                        predictions = torch.argmax(outputs, dim=-1) # Classification predictions
                    # Collect predictions and references for metric computation
                    all_preds.extend(predictions.cpu().numpy())
                    all_refs.extend(labels.cpu().numpy())
                else:
                    raise ValueError("Invalid task type")

                if loss is not None:
                    total_loss += loss.item()

            except RuntimeError as e:
                logger.error(f"Runtime error during {eval_type}: {e}")
                continue # Skip batch on error

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    results = {"loss": avg_loss}

    # Compute specific metrics for GLUE tasks
    if args.task == 'glue' and eval_metric is not None and all_preds:
        try:
            metric_results = eval_metric.compute(predictions=all_preds, references=all_refs)
            # Make metric names more descriptive (e.g., "glue_accuracy")
            metric_results = {f"glue_{k}": v for k,v in metric_results.items()}
            results.update(metric_results)
            logger.info(f"{eval_type} Metrics: {metric_results}")
        except Exception as e:
            logger.error(f"Failed to compute GLUE metrics for {args.glue_task}: {e}")

    # Compute perplexity for LM task
    if args.task == 'lm':
        try:
            perplexity = math.exp(min(avg_loss, 700)) # Cap loss for stability
            results["perplexity"] = perplexity
            logger.info(f"{eval_type} Perplexity: {perplexity:.2f}")
        except OverflowError:
            results["perplexity"] = float('inf')
            logger.warning(f"{eval_type} Perplexity calculation overflowed (loss too high).")

    logger.info(f"{eval_type} Average Loss: {avg_loss:.4f}")
    return results

# --- Main Training Script ---
def main(args):
    """Main training and evaluation function."""
    # --- Reproducibility ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info(f"Set random seed to {args.seed}")

    # --- Output Directory ---
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")
    except OSError as e:
        logger.error(f"Error creating output directory {args.output_dir}: {e}")
        exit(1)

    # --- Device Setup ---
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA device requested but not available, falling back to CPU.")
        args.device = 'cpu'
    logger.info(f"Using device: {args.device}")

    # --- Data Loading & Preprocessing ---
    # This function now also sets args.num_labels, args.is_regression based on task
    processed_datasets, tokenizer = load_and_preprocess_data(args)
    vocab_size = tokenizer.vocab_size

    # --- Create DataLoaders ---
    pin_memory_flag = (args.device != 'cpu')
    train_dataloader = DataLoader(
        processed_datasets["train"],
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory_flag
    )
    eval_dataloader = DataLoader(
        processed_datasets["validation"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory_flag
    )
    test_dataloader = DataLoader(
        processed_datasets["test"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory_flag
    )
    # Handle MNLI extra test set if present
    test_mismatched_dataloader = None
    if args.task == 'glue' and args.glue_task == 'mnli' and 'test_mismatched' in processed_datasets:
         test_mismatched_dataloader = DataLoader(
             processed_datasets["test_mismatched"],
             batch_size=args.batch_size,
             num_workers=args.num_workers,
             pin_memory=pin_memory_flag
         )


    # --- Model Initialization ---
    logger.info("\n--- Initializing Model ---")
    try:
        model = TransformerModel(
            transformer_type=args.transformer_type,
            vocab_size=vocab_size, # Use vocab size from tokenizer
            d_model=args.d_model,
            nhead=args.nhead,
            nlayers=args.nlayers,
            dropout=args.dropout,
            dim_feedforward=args.dim_feedforward,
            orth_mode=args.orth_mode,
            orth_penalty_weight=args.orth_penalty_weight,
            task=args.task,
            num_labels=args.num_labels # Pass num_labels determined from data
        ).to(args.device)

        # Resize embeddings if new tokens were added (e.g., [PAD])
        if getattr(args, 'resize_embedding', False):
             logger.info("Resizing model token embeddings to match tokenizer vocab size.")
             model.resize_token_embeddings(len(tokenizer))


    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise e

    # Log model info
    output_layer_dim = getattr(getattr(model, 'output_layer', None), 'out_features', 'N/A')
    logger.info(f"Model Output Dim: {output_layer_dim} | Tokenizer Vocab Size: {vocab_size}")
    try:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable Parameters: {param_count/1000000:.2f}M")
    except Exception as e:
        logger.info(f"Could not calculate parameter count: {e}") # Changed to info level
    logger.info(f"Model Type: {args.transformer_type.upper()}")
    logger.info(f"Orthogonality Mode: {args.orth_mode}")
    if model.transformer_type == 'ospa' and args.orth_mode == 'regularize':
        logger.info(f"Orthogonality Penalty Weight (lambda): {args.orth_penalty_weight}")

    # --- Loss Function ---
    if args.task == 'lm':
        criterion = nn.CrossEntropyLoss() # Ignore index handled by data prep if needed
    elif args.task == 'glue':
        if args.is_regression:
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
    else:
         raise ValueError("Criterion setup failed: Unknown task.")

    # --- Optimizer ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # --- LR Scheduler ---
    num_training_steps = args.epochs * len(train_dataloader) // args.gradient_accumulation_steps
    if num_training_steps == 0:
        logger.warning("Calculated zero training steps. Check dataset size, batch size, epochs, accumulation.")
        num_training_steps = 1 # Prevent error

    # Option 1: CosineAnnealingLR (requires T_max in steps if updating per step)
    # scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=args.lr / 100)

    # Option 2: Standard HF Scheduler (e.g., linear warmup then decay)
    num_warmup_steps = int(num_training_steps * 0.1) # Example: 10% warmup
    scheduler = get_scheduler(
        name="linear", # Or "cosine"
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    logger.info(f"Using LR scheduler: {type(scheduler).__name__}, Total Steps: {num_training_steps}, Warmup Steps: {num_warmup_steps}")
    # Ensure scheduler update logic matches choice (linear usually per step)
    if not args.scheduler_update_every_step and type(scheduler).__name__ != "cosine":
         logger.warning("Linear/other HF schedulers typically update per step. Consider setting --scheduler_update_every_step")
    # For CosineAnnealingLR, update per epoch if scheduler_update_every_step is False


    # --- Metrics (for GLUE) ---
    eval_metric = None
    metric_name_to_report = "loss" # Default metric to track for best model saving
    if args.task == 'glue':
        try:
            eval_metric = evaluate.load("glue", args.glue_task)
            # Determine the primary metric for this GLUE task
            if args.glue_task == "cola": metric_name_to_report = "glue_matthews_correlation"
            elif args.glue_task == "sst2": metric_name_to_report = "glue_accuracy"
            elif args.glue_task == "mrpc": metric_name_to_report = "glue_f1" # Or accuracy
            elif args.glue_task == "qqp": metric_name_to_report = "glue_f1" # Or accuracy
            elif args.glue_task == "stsb": metric_name_to_report = "glue_spearmanr" # Or pearsonr, lower loss is better here too
            elif args.glue_task == "mnli": metric_name_to_report = "glue_accuracy"
            elif args.glue_task == "qnli": metric_name_to_report = "glue_accuracy"
            elif args.glue_task == "rte": metric_name_to_report = "glue_accuracy"
            elif args.glue_task == "wnli": metric_name_to_report = "glue_accuracy" # Note: WNLI is tricky
            logger.info(f"Loaded GLUE metric for {args.glue_task}. Tracking: {metric_name_to_report}")
        except Exception as e:
            logger.error(f"Failed to load GLUE metric for task {args.glue_task}: {e}. Evaluation will only report loss.")
    elif args.task == 'lm':
        metric_name_to_report = "perplexity" # Or loss


    # --- Training Loop ---
    logger.info("\n--- Starting Training Loop ---")
    best_val_metric_value = float('inf') if metric_name_to_report == "loss" or metric_name_to_report == "perplexity" else float('-inf')
    higher_is_better = not (metric_name_to_report == "loss" or metric_name_to_report == "perplexity")

    history = {'args': vars(args), 'epochs': {}}
    model_save_path = os.path.join(args.output_dir, args.save)
    logger.info(f"Best model checkpoint will be saved to: {model_save_path}")

    training_completed_normally = False
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            logger.info(f"\n--- Epoch {epoch}/{args.epochs} ---")

            # --- Train ---
            avg_train_loss = train_epoch(model, train_dataloader, optimizer, criterion, scheduler, args, epoch, tokenizer)

            # --- Validate ---
            logger.info(f"--- Starting Epoch {epoch} Validation ---")
            val_results = evaluate(model, eval_dataloader, criterion, args, eval_metric, eval_type="Validation")

            # Store epoch results
            history['epochs'][epoch] = {"train_loss": avg_train_loss, "validation": val_results}
            logger.info(f"Epoch {epoch} Validation Results: {val_results}")
            logger.info(f"Epoch {epoch} Duration: {time.time() - epoch_start_time:.2f}s")

            # --- Save Best Model ---
            current_val_metric = val_results.get(metric_name_to_report, None)
            if current_val_metric is None:
                 logger.warning(f"Tracking metric '{metric_name_to_report}' not found in validation results. Using validation loss for checkpointing.")
                 current_val_metric = val_results['loss']
                 temp_higher_is_better = False # Loss, lower is better
            else:
                 temp_higher_is_better = higher_is_better


            # Check if current metric is better than best found so far
            is_better = False
            if temp_higher_is_better:
                if current_val_metric > best_val_metric_value:
                    is_better = True
            else: # Lower is better
                if current_val_metric < best_val_metric_value:
                    is_better = True

            if is_better:
                best_val_metric_value = current_val_metric
                logger.info(f'| New best validation {metric_name_to_report}: {best_val_metric_value:.4f}. Saving model checkpoint...')
                try:
                    torch.save(model.state_dict(), model_save_path)
                    logger.info(f'| Model checkpoint saved to {model_save_path}')
                except Exception as e:
                    logger.error(f"Failed to save model checkpoint: {e}")

        training_completed_normally = True # Mark training as complete if loop finishes

    except KeyboardInterrupt:
        logger.warning('-' * 89)
        logger.warning('| Keyboard interrupt detected - Exiting training loop early.')
        logger.warning('-' * 89)
    except Exception as e:
         logger.error(f"An unexpected error occurred during training loop: {e}", exc_info=True) # Log traceback
         # Optionally save current state before exiting
         error_save_path = os.path.join(args.output_dir, "error_checkpoint.pt")
         logger.info(f"Saving current model state due to error to {error_save_path}")
         try:
             torch.save(model.state_dict(), error_save_path)
         except Exception as save_e:
             logger.error(f"Failed to save error checkpoint: {save_e}")
         # Decide whether to re-raise or exit
         # raise e

    # --- Final Evaluation on Test Set ---
    logger.info("\n--- Final Evaluation on Test Set ---")
    # Load the best model checkpoint IF training completed normally AND checkpoint exists
    if training_completed_normally and os.path.exists(model_save_path):
        try:
            logger.info(f"Loading best model checkpoint from {model_save_path}...")
            model.load_state_dict(torch.load(model_save_path, map_location=args.device))
        except Exception as e:
            logger.error(f"Failed to load best model checkpoint from {model_save_path}: {e}. Evaluating with final model state.")
    elif not training_completed_normally:
         logger.warning("Training did not complete normally. Evaluating with model state at interruption/error.")
    else: # Training completed but no checkpoint saved (e.g., validation never improved)
        logger.warning("No best model checkpoint found (validation metric may not have improved). Evaluating with final model state.")

    # Perform final evaluation on the standard test set
    test_results = evaluate(model, test_dataloader, criterion, args, eval_metric, eval_type="Test")
    logger.info(f"Final Test Results: {test_results}")
    history['final_test_results'] = test_results

    # Evaluate on MNLI mismatched if applicable
    if args.task == 'glue' and args.glue_task == 'mnli' and test_mismatched_dataloader:
        logger.info("\n--- Final Evaluation on MNLI Mismatched Test Set ---")
        test_mm_results = evaluate(model, test_mismatched_dataloader, criterion, args, eval_metric, eval_type="Test_Mismatched")
        logger.info(f"Final Test Mismatched Results: {test_mm_results}")
        history['final_test_mismatched_results'] = test_mm_results


    # --- Save Results History ---
    # Create a descriptive filename
    lambda_str = f"lambda{args.orth_penalty_weight}" if model.transformer_type == 'ospa' and args.orth_mode == 'regularize' else "lambdaNA"
    task_name = args.glue_task if args.task == 'glue' else args.lm_dataset_name
    results_filename = (
        f"results_{args.task}_{task_name}_{args.transformer_type}_{args.orth_mode}_{lambda_str}_"
        f"d{args.d_model}_h{args.nhead}_l{args.nlayers}_s{args.seed}.json"
    )
    results_filepath = os.path.join(args.output_dir, results_filename)
    logger.info(f"Saving results history to: {results_filepath}")
    try:
        with open(results_filepath, 'w') as f:
            # Use default=str to handle potential non-serializable items like device objects
            json.dump(history, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save results JSON to {results_filepath}: {e}")

    logger.info("\n--- Training Script Finished ---")


# --- Argument Parser Setup ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='OSPA Transformer Training Script with Standard Practices',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    # --- Task Configuration ---
    parser.add_argument('--task', type=str, default='lm', choices=['lm', 'glue'],
                        help='Task type: language modeling or GLUE benchmark.')
    parser.add_argument('--glue_task', type=str, default='sst2',
                        choices=['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli'],
                        help='Specific GLUE task to run (ignored if task is lm).')
    parser.add_argument('--lm_dataset_name', type=str, default='wikitext',
                        help='Hugging Face dataset name for language modeling (e.g., wikitext, ptb_text_only).')
    parser.add_argument('--lm_dataset_config', type=str, default='wikitext-103-raw-v1', # WT103 is larger/more standard
                        help='Dataset configuration name (e.g., wikitext-2-raw-v1, wikitext-103-raw-v1).')

    # --- Model Architecture ---
    parser.add_argument('--transformer_type', type=str, default='ospa', choices=['ospa', 'vanilla'], # Removed linformer for now
                        help='Type of transformer architecture.')
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased',
                        help='Hugging Face tokenizer name (e.g., bert-base-uncased, roberta-base). Defines vocab.')
    parser.add_argument('--d_model', type=int, default=768, # BERT-base size often used
                        help='Model embedding dimension.')
    parser.add_argument('--nhead', type=int, default=12, # BERT-base heads
                         help='Number of attention heads (must divide d_model).')
    parser.add_argument('--nlayers', type=int, default=6, # Reduced layers for faster scratch training
                         help='Number of transformer encoder layers.')
    parser.add_argument('--dim_feedforward', type=int, default=3072, # BERT-base FFN dim
                        help='Dimension of the feedforward network hidden layer.')
    parser.add_argument('--dropout', type=float, default=0.1,
                         help='Dropout rate applied in the model.')

    # --- OSPA Specific Parameters ---
    parser.add_argument('--orth_mode', type=str, default='regularize', choices=['init', 'regularize', 'strict'],
                        help="How to enforce orthogonality in OSPA.")
    parser.add_argument('--orth_penalty_weight', type=float, default=0.001,
                        help='Weight (lambda) for the orthogonality penalty loss term (used in regularize mode).')

    # --- Data Handling ---
    parser.add_argument('--max_seq_len', type=int, default=128, # Common default for GLUE
                        help='Maximum sequence length for tokenizer padding/truncation and LM blocking.')

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=50, # Fewer epochs for faster iteration
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per device.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Accumulate gradients over N steps before optimizer update.')
    parser.add_argument('--lr', type=float, default=2e-5, # Common fine-tuning LR for AdamW
                        help='Initial learning rate for the AdamW optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (L2 regularization) for the optimizer.')
    parser.add_argument('--clip', type=float, default=1.0, # Common default clipping
                        help='Maximum norm for gradient clipping.')
    parser.add_argument('--scheduler_update_every_step', action=argparse.BooleanOptionalAction, default=True,
                        help='Update learning rate scheduler every optimizer step (recommended for linear/cosine).')

    # --- Runtime Environment ---
    parser.add_argument('--device', type=str, default=None,
                        help="Device to use ('cuda', 'cpu', or specific GPU like 'cuda:0'). Auto-detects if None.")
    parser.add_argument('--num_workers', type=int, default=4, # Often higher helps
                        help='Number of worker processes for DataLoader.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')

    # --- Logging and Saving ---
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Log training status every N batches.')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save model checkpoints and results JSON.')
    parser.add_argument('--save', type=str, default='best_model.pt',
                        help='Filename for saving the best model checkpoint within the output directory.')


    args = parser.parse_args()

    # --- Post-processing and Validation ---
    # Auto-detect device if not specified
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Validate model dimensions
    if args.d_model % args.nhead != 0:
        parser.error(f"--d_model ({args.d_model}) must be divisible by --nhead ({args.nhead})")

    # Ensure task-specific args are logical
    if args.task == 'lm':
         # Maybe override glue_task if set incorrectly
         args.glue_task = None
    elif args.task == 'glue':
        if not args.glue_task:
            parser.error("--glue_task must be specified when --task is 'glue'")
        # Override LM dataset args if set incorrectly
        args.lm_dataset_name = None
        args.lm_dataset_config = None
    else:
         # Should be caught by choices, but defensive check
         parser.error(f"Invalid task specified: {args.task}")


    # --- Start Training ---
    main(args)
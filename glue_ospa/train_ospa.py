# train_ospa.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    from datasets import load_dataset, ClassLabel
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
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                 # Add a new pad token if no eos token either
                 logger.warning(f"Tokenizer {args.tokenizer_name} missing pad token and eos token. Adding new [PAD] token.")
                 tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                 # Important: The model embedding layer needs resizing after adding tokens
                 args.resize_embedding = True # Flag to resize model embeddings later

        # Ensure pad_token_id is set if pad_token exists
        if tokenizer.pad_token is not None and tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)


    except Exception as e:
        logger.error(f"Failed to load tokenizer '{args.tokenizer_name}': {e}", exc_info=True)
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
            # Load with a specific cache directory if desired
            # raw_datasets = load_dataset("glue", glue_task, cache_dir=args.cache_dir)
            raw_datasets = load_dataset("glue", glue_task)
        except Exception as e:
            logger.error(f"Failed to load GLUE dataset '{glue_task}': {e}", exc_info=True)
            exit(1)

        sentence1_key, sentence2_key = task_to_keys[glue_task]

        # Determine label info from dataset features
        args.is_regression = glue_task == "stsb"
        label_column_name = "label" # Standard label column in GLUE
        if not args.is_regression:
            # Ensure label is ClassLabel type
            if not isinstance(raw_datasets["train"].features[label_column_name], ClassLabel):
                # Attempt to cast if it's just int (sometimes happens with local data)
                logger.warning(f"Label column for {glue_task} is not ClassLabel. Attempting to cast.")
                # This is tricky; usually features are set by `datasets` library.
                # If loading custom data, ensure Features are defined correctly.
                # For GLUE from HF Hub, this should be correct.
                pass # Assume it's okay or will be handled by label_list below
            try:
                label_list = raw_datasets["train"].features[label_column_name].names
                args.num_labels = len(label_list)
                logger.info(f"Task: {glue_task} (Classification), Num Labels: {args.num_labels}, Labels: {label_list}")
            except Exception as e:
                logger.error(f"Could not infer labels for GLUE task {glue_task}: {e}. Features: {raw_datasets['train'].features}")
                # Fallback for tasks like WNLI which might have issues or if features are minimal
                unique_labels = sorted(list(set(raw_datasets["train"][label_column_name])))
                args.num_labels = len(unique_labels)
                logger.warning(f"Falling back to inferring num_labels from unique values: {args.num_labels} ({unique_labels})")
                if args.num_labels <= 1:
                    logger.error("Inferred only one label or invalid label set. Check dataset.")
                    exit(1)
        else:
            args.num_labels = 1
            logger.info(f"Task: {glue_task} (Regression), Num Labels: {args.num_labels}")


        # Preprocessing function for tokenizer
        def preprocess_glue(examples):
            if sentence2_key is None: # Single sentence task
                texts = (examples[sentence1_key],)
            else: # Sentence pair task
                texts = (examples[sentence1_key], examples[sentence2_key])
            # Padding to max_length, truncation if longer
            result = tokenizer(*texts, padding="max_length", max_length=args.max_seq_len, truncation=True)
            # For STSB, labels are float. Ensure they are handled if present in examples.
            # The `map` function will keep other columns if not in remove_columns.
            return result


        # Columns to remove after tokenization (original text columns)
        remove_cols = [key for key in [sentence1_key, sentence2_key, "idx"] if key is not None]


        # Apply tokenization
        logger.info(f"Tokenizing GLUE/{glue_task} with max_seq_len={args.max_seq_len}...")
        try:
            processed_datasets = raw_datasets.map(
                preprocess_glue,
                batched=True,
                remove_columns=remove_cols, # Remove original text columns after tokenization
                desc=f"Tokenizing GLUE/{glue_task}"
            )
        except Exception as e:
            logger.error(f"Error during GLUE tokenization: {e}", exc_info=True)
            exit(1)

        # Rename label column to 'labels' for consistency with some HF models/trainers
        if label_column_name != "labels":
            processed_datasets = processed_datasets.rename_column(label_column_name, "labels")

        processed_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels", "token_type_ids"] if sentence2_key else ["input_ids", "attention_mask", "labels"])


        # Select splits (handle MNLI special case for validation)
        train_dataset = processed_datasets["train"]
        validation_key = "validation_matched" if glue_task == "mnli" else "validation"
        # Handle tasks like WNLI that might not have a standard validation split
        eval_dataset = processed_datasets.get(validation_key)
        if eval_dataset is None:
            logger.warning(f"Split '{validation_key}' not found for {glue_task}. Attempting to use 'test' split if available, or 'train' split for evaluation.")
            eval_dataset = processed_datasets.get("test", processed_datasets["train"]) # Fallback

        # Use validation set as test set (no public test labels for GLUE)
        test_dataset = eval_dataset

        # MNLI also has a mismatched validation set
        final_datasets = {"train": train_dataset, "validation": eval_dataset, "test": test_dataset}
        if glue_task == "mnli":
            # Check if mismatched split exists
            mismatched_split = "validation_mismatched" # Or "test_mismatched" if using that for submission
            if mismatched_split in processed_datasets:
                final_datasets["test_mismatched"] = processed_datasets[mismatched_split]
                logger.info("Using MNLI validation_matched for validation, and validation_mismatched also available.")
            else:
                logger.warning(f"MNLI mismatched split '{mismatched_split}' not found.")

        logger.info(f"GLUE Data Processing Complete. Train size: {len(final_datasets['train'])}, Eval size: {len(final_datasets['validation'])}")
        return final_datasets, tokenizer


    elif args.task == 'lm':
        logger.info(f"--- Loading and Preprocessing LM dataset: {args.lm_dataset_name} (config: {args.lm_dataset_config}) ---")
        try:
            raw_datasets = load_dataset(args.lm_dataset_name, args.lm_dataset_config)
        except Exception as e:
             logger.error(f"Failed to load LM dataset '{args.lm_dataset_name}' config '{args.lm_dataset_config}': {e}", exc_info=True)
             exit(1)

        text_column_name = "text" # Common for datasets like wikitext
        if text_column_name not in raw_datasets["train"].column_names:
            logger.error(f"Expected column '{text_column_name}' not found in LM dataset. Available columns: {raw_datasets['train'].column_names}")
            # Attempt to find another likely text column if 'text' is not present
            potential_text_cols = [col for col in raw_datasets["train"].column_names if isinstance(raw_datasets["train"].features[col], Value) and raw_datasets["train"].features[col].dtype == 'string']
            if potential_text_cols:
                text_column_name = potential_text_cols[0]
                logger.warning(f"Using first string column '{text_column_name}' as text input for LM.")
            else:
                logger.error("No suitable string column found for LM text input.")
                exit(1)


        args.num_labels = tokenizer.vocab_size # For LM, output layer predicts vocab tokens
        args.is_regression = False
        logger.info(f"Task: Language Modeling, Vocab Size: {args.num_labels}")

        # Tokenization function for LM
        def tokenize_lm(examples):
            # Tokenize without padding or truncation initially; grouping will handle fixed lengths.
            return tokenizer(examples[text_column_name], truncation=False)

        logger.info("Tokenizing LM dataset...")
        try:
            # `tokenized_datasets` will typically have 'input_ids' and 'attention_mask' (if tokenizer adds it)
            tokenized_datasets = raw_datasets.map(
                tokenize_lm,
                batched=True,
                remove_columns=[text_column_name], # Remove original text column after tokenization
                desc="Tokenizing LM data"
            )
        except Exception as e:
            logger.error(f"Error during LM tokenization: {e}", exc_info=True)
            exit(1)

        # Determine block_size for grouping texts
        block_size = args.max_seq_len
        if block_size is None: # If not specified, try to use tokenizer's model_max_length
            block_size = tokenizer.model_max_length
            # Cap block_size to avoid excessive memory usage if tokenizer's max_len is very large
            if block_size > 1024 and block_size is not None:
                logger.warning(f"Tokenizer model_max_length ({tokenizer.model_max_length}) is large, capping block_size to 1024 for LM.")
                block_size = 1024
            elif block_size is None: # Fallback if model_max_length is also None
                 logger.warning("max_seq_len and tokenizer.model_max_length are None. Defaulting LM block_size to 128.")
                 block_size = 128
            args.max_seq_len = block_size # Update args if block_size was derived
        logger.info(f"Grouping LM texts into blocks of size {block_size}...")


        def group_texts(examples):
            # Concatenate all texts for 'input_ids' and other keys from tokenizer output
            # (e.g. 'attention_mask' if tokenizer adds it for special tokens even without padding)
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # Drop the small remainder that is smaller than block_size
            if total_length < block_size : # Handle case where total_length is less than block_size
                 return {k: [] for k in examples.keys()} # Return empty dict for this batch
            total_length = (total_length // block_size) * block_size

            # Split by chunks of block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            # Create labels by shifting input_ids (common for causal LM)
            result["labels"] = result["input_ids"].copy()
            return result

        try:
            # After grouping, lm_datasets contains lists of lists of integers for input_ids, etc.
            lm_datasets = tokenized_datasets.map(group_texts, batched=True, desc="Grouping LM texts")
        except Exception as e:
             logger.error(f"Error grouping LM texts: {e}", exc_info=True)
             exit(1)

        # CRITICAL: Set format to torch AFTER all mapping operations
        try:
            # Ensure all columns are convertible to torch tensors
            # Relevant columns are 'input_ids', 'attention_mask' (if present), 'labels'
            columns_to_set_format = ["input_ids", "labels"]
            if "attention_mask" in lm_datasets["train"].column_names:
                columns_to_set_format.append("attention_mask")
            if "token_type_ids" in lm_datasets["train"].column_names: # Less common for scratch LM
                columns_to_set_format.append("token_type_ids")

            lm_datasets.set_format("torch", columns=columns_to_set_format)
        except Exception as e:
            logger.error(f"Error setting torch format for LM datasets: {e}", exc_info=True)
            logger.info(f"Columns in LM dataset 'train' split before set_format: {lm_datasets['train'].column_names}")
            exit(1)

        train_dataset = lm_datasets["train"]
        # Determine validation and test splits, handling missing splits
        eval_dataset = lm_datasets.get("validation")
        if eval_dataset is None:
            logger.warning("LM dataset 'validation' split not found. Using 'test' split for validation.")
            eval_dataset = lm_datasets.get("test")
        if eval_dataset is None: # If still none, use train split for eval (not ideal)
            logger.error("LM dataset 'validation' and 'test' splits not found. Using 'train' for validation - CHECK DATASET.")
            eval_dataset = lm_datasets["train"]


        test_dataset = lm_datasets.get("test")
        if test_dataset is None:
            logger.warning("LM dataset 'test' split not found. Using 'validation' split for testing.")
            test_dataset = eval_dataset # Fallback to eval_dataset if no specific test set
        if test_dataset is None: # If still none, error
            logger.error("Could not find suitable test split for LM.")
            exit(1)


        logger.info("LM Data Processing Complete.")
        logger.info(f"LM Dataset sizes: Train={len(train_dataset)}, Validation={len(eval_dataset)}, Test={len(test_dataset)}")
        return {"train": train_dataset, "validation": eval_dataset, "test": test_dataset}, tokenizer

    else:
        # This case should be caught by argument parsing, but defensive
        raise ValueError(f"Unsupported task type specified: {args.task}")

# --- Training Epoch Function ---
def train_epoch(model, dataloader, optimizer, criterion, scheduler, args, epoch, tokenizer): # tokenizer might not be needed here
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs} Training", leave=False)

    for batch_idx, batch_data in enumerate(progress_bar): # Rename batch to batch_data
        # --- Move batch to device based on task ---
        input_ids, attention_mask, labels = None, None, None
        try:
            if args.task == 'lm':
                # LM batches are dictionaries from datasets.map(group_texts)
                input_ids = batch_data['input_ids'].to(args.device)
                # attention_mask is usually all 1s for LM blocks from group_texts, but good to have
                attention_mask = batch_data.get('attention_mask', torch.ones_like(input_ids)).to(args.device)
                labels = batch_data['labels'].to(args.device)
            elif args.task == 'glue':
                # GLUE batches are tuples (input_features, labels) from your collate_fn
                # input_features is already [SeqLen, BatchSize]
                # labels is [BatchSize]
                # The tokenizer in preprocess_glue already creates input_ids and attention_mask
                # So, your collate_fn for GLUE should ideally return a dict like the LM one
                # OR you unpack the tuple here
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    # Assuming batch_data[0] contains tokenized inputs (potentially a dict itself or a tensor)
                    # And batch_data[1] contains labels
                    
                    # If your GLUE collate_fn returns a dict from tokenizer:
                    # input_ids = batch_data[0]['input_ids'].to(args.device)
                    # attention_mask = batch_data[0]['attention_mask'].to(args.device)
                    # labels = batch_data[1].to(args.device)

                    # *** If your GLUE collate_fn returns (text_tensor, label_tensor) as in your code: ***
                    text_tensor, label_tensor = batch_data
                    input_ids = text_tensor.to(args.device) # text_tensor is already [SeqLen, BatchSize]
                    labels = label_tensor.to(args.device)
                    # We need attention_mask for GLUE based on padding.
                    # This should be generated in collate_fn OR derived here if pad_idx is known
                    pad_idx = tokenizer.pad_token_id # Get pad_token_id from the HF tokenizer
                    if pad_idx is None:
                        logger.warning("PAD token ID not found in tokenizer. Padding mask will not be effective.")
                        attention_mask = torch.ones_like(input_ids, device=args.device, dtype=torch.long) # Assume no padding
                    else:
                        # Create attention_mask: 1 for non-pad, 0 for pad
                        # input_ids is [SeqLen, BatchSize]
                        attention_mask = (input_ids != pad_idx).long()
                        # The model's forward pass expects key_padding_mask as [BatchSize, SeqLen] with True for PAD
                        # We pass attention_mask which is [SeqLen, BatchSize] with 1 for real token
                        # The model's forward pass needs to handle this conversion if it expects key_padding_mask
                else:
                    logger.error(f"Unexpected batch format for GLUE task: {type(batch_data)}. Expected tuple of (inputs, labels) or dict.")
                    continue
            else:
                logger.error(f"Unknown task '{args.task}' for batch processing.")
                continue
            
            if input_ids is None: # Check if processing failed
                logger.warning(f"Batch {batch_idx} could not be processed for device transfer. Skipping.")
                continue

        except Exception as e:
            logger.warning(f"Could not process or move batch {batch_idx} to device {args.device}: {e}. Skipping batch.")
            continue
        # -----------------------------------------

        # --- Gradient Accumulation: Zero Grads ---
        if batch_idx % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        try:
            # --- Forward Pass ---
            # Model now expects input_ids and attention_mask (HF style)
            # The model's forward should internally derive src_key_padding_mask from attention_mask
            # and generate causal_mask if task is 'lm'
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # ... (rest of your train_epoch loop: NaN check, loss, penalty, backward, step, logging) ...
            # Make sure loss calculation uses the correct labels and output shapes:
            if args.task == 'lm':
                loss = criterion(outputs.view(-1, model.vocab_size), labels.view(-1))
            elif args.task == 'glue':
                if args.is_regression:
                    loss = criterion(outputs.squeeze(), labels.float())
                else:
                    loss = criterion(outputs, labels) # labels are already [BatchSize]
            else:
                raise ValueError("Invalid task for loss calculation")


            if model.transformer_type == "ospa" and model.orth_mode == "regularize":
                orth_penalty = model.get_orthogonality_penalty()
                if not (torch.isnan(orth_penalty).any() or torch.isinf(orth_penalty).any()):
                     loss = loss + args.orth_penalty_weight * orth_penalty
                else:
                     logger.warning(f"NaN/Inf in orth penalty (Epoch {epoch}, Batch {batch_idx}). Penalty not added.")

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            is_accumulation_step = (batch_idx + 1) % args.gradient_accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == num_batches
            if is_accumulation_step or is_last_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                if args.scheduler_update_every_step:
                     if scheduler and hasattr(scheduler, 'step'):
                         scheduler.step()

            current_unscaled_loss = loss.item() * args.gradient_accumulation_steps
            total_loss += current_unscaled_loss
            progress_bar.set_postfix(loss=f"{current_unscaled_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}" if scheduler else f"{optimizer.param_groups[0]['lr']:.6f}")


        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error(f"CUDA OOM encountered in training (Epoch {epoch}, Batch {batch_idx}). Try reducing batch size or model size.")
                raise e
            else:
                logger.error(f"Runtime error during training (Epoch {epoch}, Batch {batch_idx}): {e}")
                optimizer.zero_grad(set_to_none=True)
                continue

    if not args.scheduler_update_every_step:
         if scheduler and hasattr(scheduler, 'step'):
             scheduler.step()

    avg_epoch_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    logger.info(f"Epoch {epoch} Training Average Loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss


# --- Evaluation Function ---
def evaluate(model, dataloader, criterion, args, tokenizer, eval_metric=None, eval_type="Validation"): # Added tokenizer
    """Evaluates the model on a given dataloader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_refs = []
    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, desc=f"{eval_type} Evaluating", leave=False)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(progress_bar): # Rename batch
            # --- Move batch to device based on task ---
            input_ids, attention_mask, labels = None, None, None
            try:
                if args.task == 'lm':
                    input_ids = batch_data['input_ids'].to(args.device)
                    attention_mask = batch_data.get('attention_mask', torch.ones_like(input_ids)).to(args.device)
                    labels = batch_data['labels'].to(args.device)
                elif args.task == 'glue':
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        text_tensor, label_tensor = batch_data
                        input_ids = text_tensor.to(args.device)
                        labels = label_tensor.to(args.device)
                        pad_idx = tokenizer.pad_token_id
                        if pad_idx is None:
                            attention_mask = torch.ones_like(input_ids, device=args.device, dtype=torch.long)
                        else:
                            attention_mask = (input_ids != pad_idx).long()
                    else:
                        logger.error(f"Unexpected batch format for GLUE task: {type(batch_data)} during eval. Expected tuple.")
                        continue
                else:
                    logger.error(f"Unknown task '{args.task}' for batch processing during eval.")
                    continue
                
                if input_ids is None:
                    logger.warning(f"Eval Batch {batch_idx} could not be processed for device transfer. Skipping.")
                    continue

            except Exception as e:
                logger.warning(f"Could not process or move eval batch {batch_idx} to device {args.device}: {e}. Skipping.")
                continue
            # -----------------------------------------

            try:
                # --- Forward Pass ---
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                # ... (rest of your evaluate function: NaN check, loss, metric calculation) ...
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    logger.warning(f"NaN or Inf detected in {eval_type} output. Skipping batch.")
                    continue

                loss = None
                if args.task == 'lm':
                    loss = criterion(outputs.view(-1, model.vocab_size), labels.view(-1))
                elif args.task == 'glue':
                    if args.is_regression:
                        loss = criterion(outputs.squeeze(), labels.float())
                        predictions = outputs.squeeze()
                    else:
                        loss = criterion(outputs, labels)
                        predictions = torch.argmax(outputs, dim=-1)
                    all_preds.extend(predictions.cpu().numpy())
                    all_refs.extend(labels.cpu().numpy())
                else:
                    raise ValueError("Invalid task for loss calculation")

                if loss is not None:
                    total_loss += loss.item()

            except RuntimeError as e:
                logger.error(f"Runtime error during {eval_type}: {e}")
                continue

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    results = {"loss": avg_loss}

    if args.task == 'glue' and eval_metric is not None and all_preds:
        try:
            metric_results = eval_metric.compute(predictions=all_preds, references=all_refs)
            metric_results = {f"glue_{k}": v for k,v in metric_results.items()}
            results.update(metric_results)
            logger.info(f"{eval_type} Metrics: {metric_results}")
        except Exception as e:
            logger.error(f"Failed to compute GLUE metrics for {args.glue_task}: {e}")

    if args.task == 'lm':
        try:
            perplexity = math.exp(min(avg_loss, 700))
            results["perplexity"] = perplexity
            logger.info(f"{eval_type} Perplexity: {perplexity:.2f}")
        except OverflowError:
            results["perplexity"] = float('inf')
            logger.warning(f"{eval_type} Perplexity calculation overflowed (loss too high).")

    logger.info(f"{eval_type} Average Loss: {avg_loss:.4f}")
    return results

# --- Main Training Script ---
def main(args):
    # (Your main function remains largely the same, but ensure 'tokenizer'
    # is passed to train_epoch and evaluate if they need it for padding mask derivation.
    # The evaluate function was updated to accept it.)

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
    processed_datasets, tokenizer = load_and_preprocess_data(args) # tokenizer is returned
    vocab_size = tokenizer.vocab_size

    # --- Create DataLoaders ---
    pin_memory_flag = (args.device != 'cpu')
    train_dataloader = DataLoader(
        processed_datasets["train"],
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers, # Set to 0 if "Too many open files" error persists
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
            vocab_size=vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            nlayers=args.nlayers,
            dropout=args.dropout,
            dim_feedforward=args.dim_feedforward,
            orth_mode=args.orth_mode,
            orth_penalty_weight=args.orth_penalty_weight,
            task=args.task,
            num_labels=args.num_labels
        ).to(args.device)

        if getattr(args, 'resize_embedding', False):
             logger.info("Resizing model token embeddings to match tokenizer vocab size after potential [PAD] addition.")
             model.resize_token_embeddings(len(tokenizer)) # len(tokenizer) is new vocab size

    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise e

    logger.info(f"Model Output Dim: {getattr(getattr(model, 'output_layer', None), 'out_features', 'N/A')} | Tokenizer Vocab Size: {vocab_size}")
    try:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable Parameters: {param_count/1000000:.2f}M")
    except: pass
    logger.info(f"Model Type: {args.transformer_type.upper()}")
    logger.info(f"Orthogonality Mode: {args.orth_mode}")
    if model.transformer_type == 'ospa' and args.orth_mode == 'regularize':
        logger.info(f"Orthogonality Penalty Weight (lambda): {args.orth_penalty_weight}")

    # --- Loss Function ---
    if args.task == 'lm':
        criterion = nn.CrossEntropyLoss()
    elif args.task == 'glue':
        if args.is_regression:
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
    else:
         raise ValueError("Criterion setup failed: Unknown task.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = args.epochs * len(train_dataloader) // args.gradient_accumulation_steps
    if num_training_steps == 0: num_training_steps = 1
    num_warmup_steps = int(num_training_steps * 0.1)
    scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    logger.info(f"Using LR scheduler: {scheduler.name if hasattr(scheduler, 'name') else type(scheduler).__name__}, Total Steps: {num_training_steps}, Warmup Steps: {num_warmup_steps}")

    eval_metric = None
    metric_name_to_report = "loss"
    if args.task == 'glue':
        try:
            eval_metric = evaluate.load("glue", args.glue_task)
            # Metric selection logic ... (same as before)
            if args.glue_task == "cola": metric_name_to_report = "glue_matthews_correlation"
            elif args.glue_task == "sst2": metric_name_to_report = "glue_accuracy"
            elif args.glue_task == "mrpc": metric_name_to_report = "glue_f1"
            elif args.glue_task == "qqp": metric_name_to_report = "glue_f1"
            elif args.glue_task == "stsb": metric_name_to_report = "glue_spearmanr"
            elif args.glue_task == "mnli": metric_name_to_report = "glue_accuracy"
            elif args.glue_task == "qnli": metric_name_to_report = "glue_accuracy"
            elif args.glue_task == "rte": metric_name_to_report = "glue_accuracy"
            elif args.glue_task == "wnli": metric_name_to_report = "glue_accuracy"
            logger.info(f"Loaded GLUE metric for {args.glue_task}. Tracking: {metric_name_to_report}")
        except Exception as e:
            logger.error(f"Failed to load GLUE metric for task {args.glue_task}: {e}.")
    elif args.task == 'lm':
        metric_name_to_report = "perplexity"

    logger.info("\n--- Starting Training Loop ---")
    best_val_metric_value = float('inf') if metric_name_to_report in ["loss", "perplexity"] else float('-inf')
    higher_is_better = not (metric_name_to_report in ["loss", "perplexity"])
    history = {'args': vars(args), 'epochs': {}}
    model_save_path = os.path.join(args.output_dir, args.save)
    logger.info(f"Best model checkpoint will be saved to: {model_save_path}")

    training_completed_normally = False
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            logger.info(f"\n--- Epoch {epoch}/{args.epochs} ---")
            avg_train_loss = train_epoch(model, train_dataloader, optimizer, criterion, scheduler, args, epoch, tokenizer) # Pass tokenizer
            val_results = evaluate(model, eval_dataloader, criterion, args, tokenizer, eval_metric, eval_type="Validation") # Pass tokenizer

            history['epochs'][epoch] = {"train_loss": avg_train_loss, "validation": val_results}
            logger.info(f"Epoch {epoch} Validation Results: {val_results}")
            logger.info(f"Epoch {epoch} Duration: {time.time() - epoch_start_time:.2f}s")

            current_val_metric = val_results.get(metric_name_to_report, val_results['loss']) # Fallback to loss
            temp_higher_is_better = higher_is_better
            if metric_name_to_report not in val_results and metric_name_to_report != "loss": # If metric not found, used loss
                temp_higher_is_better = False


            is_better = False
            if temp_higher_is_better:
                if current_val_metric > best_val_metric_value: is_better = True
            else:
                if current_val_metric < best_val_metric_value: is_better = True

            if is_better:
                best_val_metric_value = current_val_metric
                logger.info(f'| New best validation {metric_name_to_report}: {best_val_metric_value:.4f}. Saving model...')
                try:
                    torch.save(model.state_dict(), model_save_path)
                    logger.info(f'| Model checkpoint saved to {model_save_path}')
                except Exception as e: logger.error(f"Failed to save model checkpoint: {e}")
        training_completed_normally = True
    except KeyboardInterrupt: logger.warning("Keyboard interrupt - Exiting training early.")
    except Exception as e: logger.error(f"Unexpected error in training loop: {e}", exc_info=True)
    # ... (rest of main function: final eval, saving results) ...

    # --- Final Evaluation on Test Set ---
    logger.info("\n--- Final Evaluation on Test Set ---")
    if training_completed_normally and os.path.exists(model_save_path):
        try:
            logger.info(f"Loading best model checkpoint from {model_save_path}...")
            model.load_state_dict(torch.load(model_save_path, map_location=args.device))
        except Exception as e:
            logger.error(f"Failed to load best model checkpoint: {e}. Evaluating with final model state.")
    elif not training_completed_normally:
         logger.warning("Training did not complete. Evaluating with model state at interruption/error.")
    else:
        logger.warning("No best model checkpoint saved. Evaluating with final model state.")

    test_results = evaluate(model, test_dataloader, criterion, args, tokenizer, eval_metric, eval_type="Test") # Pass tokenizer
    logger.info(f"Final Test Results: {test_results}")
    history['final_test_results'] = test_results

    if args.task == 'glue' and args.glue_task == 'mnli' and test_mismatched_dataloader:
        logger.info("\n--- Final Evaluation on MNLI Mismatched Test Set ---")
        test_mm_results = evaluate(model, test_mismatched_dataloader, criterion, args, tokenizer, eval_metric, eval_type="Test_Mismatched") # Pass tokenizer
        logger.info(f"Final Test Mismatched Results: {test_mm_results}")
        history['final_test_mismatched_results'] = test_mm_results

    # --- Save Results History ---
    lambda_str = f"lambda{args.orth_penalty_weight}" if model.transformer_type == 'ospa' and args.orth_mode == 'regularize' else "lambdaNA"
    task_name = args.glue_task if args.task == 'glue' else args.lm_dataset_name
    results_filename = (
        f"results_{args.task}_{task_name.replace('/', '-')}_{args.transformer_type}_{args.orth_mode}_{lambda_str}_"
        f"d{args.d_model}_h{args.nhead}_l{args.nlayers}_s{args.seed}.json"
    )
    results_filepath = os.path.join(args.output_dir, results_filename)
    logger.info(f"Saving results history to: {results_filepath}")
    try:
        with open(results_filepath, 'w') as f:
            json.dump(history, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save results JSON: {e}")

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
    parser.add_argument('--orth_penalty_weight', type=float, default=0.001, # Default based on LM results
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
    parser.add_argument('--num_workers', type=int, default=0, # Default to 0 to avoid "Too many open files" initially
                        help='Number of worker processes for DataLoader. Set to 0 if encountering file descriptor issues.')
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
    main(args)
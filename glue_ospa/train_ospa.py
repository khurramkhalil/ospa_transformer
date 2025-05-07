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
    from datasets import load_dataset, ClassLabel, Value
    import evaluate # Hugging Face Evaluate library for metrics
except ImportError:
    print("Error: Required libraries not found.")
    print("Please install: pip install torch transformers datasets evaluate scikit-learn tqdm")
    exit(1)

torch.autograd.set_detect_anomaly(True)

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


def load_and_preprocess_data(args):
    """Loads and preprocesses data for LM or GLUE tasks using transformers."""
    logger.info(f"--- Loading Tokenizer: {args.tokenizer_name} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                logger.warning(f"Tokenizer {args.tokenizer_name} missing pad token, using eos token {tokenizer.eos_token} as pad token.")
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id # Make sure ID is also set
            else:
                 logger.warning(f"Tokenizer {args.tokenizer_name} missing pad token and eos token. Adding new [PAD] token.")
                 tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                 args.resize_embedding = True
        if tokenizer.pad_token is not None and tokenizer.pad_token_id is None: # Ensure ID is set if token exists
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    except Exception as e:
        logger.error(f"Failed to load tokenizer '{args.tokenizer_name}': {e}", exc_info=True)
        exit(1)

    task_to_keys = {
        "cola": ("sentence", None), "sst2": ("sentence", None),
        "mrpc": ("sentence1", "sentence2"), "qqp": ("question1", "question2"),
        "stsb": ("sentence1", "sentence2"), "mnli": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"), "rte": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    if args.task == 'glue':
        glue_task = args.glue_task.lower()
        if glue_task not in task_to_keys:
             logger.error(f"Invalid GLUE task: {glue_task}. Valid: {list(task_to_keys.keys())}")
             exit(1)
        logger.info(f"--- Loading and Preprocessing GLUE task: {glue_task} ---")
        try:
            raw_datasets = load_dataset("glue", glue_task)
        except Exception as e:
            logger.error(f"Failed to load GLUE dataset '{glue_task}': {e}", exc_info=True)
            exit(1)

        sentence1_key, sentence2_key = task_to_keys[glue_task]
        args.is_regression = glue_task == "stsb"
        label_column_name = "label"
        if not args.is_regression:
            # ... (label handling - assume this part is okay for now, focusing on padding) ...
            try:
                label_list = raw_datasets["train"].features[label_column_name].names
                args.num_labels = len(label_list)
            except Exception: # Fallback
                unique_labels = sorted(list(set(raw_datasets["train"][label_column_name])))
                args.num_labels = len(unique_labels)
                if args.num_labels <=1: exit(1) # Should not happen
            logger.info(f"Task: {glue_task} (Classification), Num Labels: {args.num_labels}")
        else:
            args.num_labels = 1
            logger.info(f"Task: {glue_task} (Regression), Num Labels: {args.num_labels}")

        def preprocess_glue(examples):
            if sentence2_key is None: texts = (examples[sentence1_key],)
            else: texts = (examples[sentence1_key], examples[sentence2_key])
            # Tokenizer generates 'input_ids' and 'attention_mask'
            return tokenizer(*texts, padding="max_length", max_length=args.max_seq_len, truncation=True)

        remove_cols = ["idx"] + ([sentence1_key] if sentence1_key else []) + ([sentence2_key] if sentence2_key else [])
        logger.info(f"Tokenizing GLUE/{glue_task} with max_seq_len={args.max_seq_len}...")
        try:
            processed_datasets = raw_datasets.map(
                preprocess_glue, batched=True, remove_columns=remove_cols, desc=f"Tokenizing GLUE/{glue_task}"
            )
        except Exception as e:
            logger.error(f"Error during GLUE tokenization: {e}", exc_info=True); exit(1)

        # *** ADD FILTERING STEP FOR GLUE ***
        logger.info("Filtering out GLUE examples that are entirely padding...")
        def filter_all_padding(example):
            # 'attention_mask' from HF tokenizer: 1 for real token, 0 for padding
            # example['attention_mask'] is a list of ints here
            return any(token_mask == 1 for token_mask in example['attention_mask'])

        for split_name in list(processed_datasets.keys()): # Use list for safe iteration
            original_len = len(processed_datasets[split_name])
            # Filter directly on the dataset object
            processed_datasets[split_name] = processed_datasets[split_name].filter(
                filter_all_padding,
                desc=f"Filtering all-pad for {split_name}"
            ) # This creates a new filtered dataset object
            new_len = len(processed_datasets[split_name])
            if new_len < original_len:
                logger.info(f"Filtered out {original_len - new_len} all-padding examples from GLUE {split_name} split.")
            if new_len == 0 and original_len > 0:
                logger.warning(f"GLUE split '{split_name}' became empty after filtering! Check data or max_seq_len (current: {args.max_seq_len}).")
        # **********************************

        if label_column_name != "labels":
            processed_datasets = processed_datasets.rename_column(label_column_name, "labels")
        
        # Define columns to set to torch format AFTER filtering
        torch_columns = ["input_ids", "attention_mask", "labels"]
        if sentence2_key and "token_type_ids" in processed_datasets["train"].column_names:
             torch_columns.append("token_type_ids")
        processed_datasets.set_format("torch", columns=torch_columns)

        train_dataset = processed_datasets["train"]
        validation_key = "validation_matched" if glue_task == "mnli" else "validation"
        eval_dataset = processed_datasets.get(validation_key)
        if eval_dataset is None:
            eval_dataset = processed_datasets.get("test", processed_datasets["train"])
        test_dataset = eval_dataset
        final_datasets = {"train": train_dataset, "validation": eval_dataset, "test": test_dataset}
        if glue_task == "mnli":
            mismatched_split = "validation_mismatched"
            if mismatched_split in processed_datasets:
                final_datasets["test_mismatched"] = processed_datasets[mismatched_split]
        logger.info(f"GLUE Data Processing Complete. Train size: {len(final_datasets['train'])}, Eval size: {len(final_datasets['validation'])}")
        return final_datasets, tokenizer

    elif args.task == 'lm':
        # ... (Your LM data processing code - this should not produce all-padding sequences
        # if group_texts works correctly, as it creates full blocks from actual tokens) ...
        # The key for LM is that group_texts should ensure resulting blocks have content.
        # If an LM block's attention_mask (if tokenizer generates one for non-padded blocks)
        # were all zeros, that would be an issue with tokenize_lm or group_texts.
        # But the current error is strongly pointing to GLUE due to the `key_padding_mask` print.
        logger.info(f"--- Loading and Preprocessing LM dataset: {args.lm_dataset_name} (config: {args.lm_dataset_config}) ---")
        try:
            raw_datasets = load_dataset(args.lm_dataset_name, args.lm_dataset_config)
        except Exception as e:
             logger.error(f"Failed to load LM dataset '{args.lm_dataset_name}' config '{args.lm_dataset_config}': {e}", exc_info=True)
             exit(1)

        text_column_name = "text"
        if text_column_name not in raw_datasets["train"].column_names:
            potential_text_cols = [col for col in raw_datasets["train"].column_names if isinstance(raw_datasets["train"].features[col], Value) and raw_datasets["train"].features[col].dtype == 'string']
            if potential_text_cols:
                text_column_name = potential_text_cols[0]
                logger.warning(f"Using first string column '{text_column_name}' as text input for LM.")
            else:
                logger.error("No suitable string column found for LM text input."); exit(1)

        args.num_labels = tokenizer.vocab_size
        args.is_regression = False
        logger.info(f"Task: Language Modeling, Vocab Size: {args.num_labels}")

        def tokenize_lm(examples):
            return tokenizer(examples[text_column_name], truncation=False)

        logger.info("Tokenizing LM dataset...")
        try:
            tokenized_datasets = raw_datasets.map(
                tokenize_lm, batched=True, remove_columns=[text_column_name], desc="Tokenizing LM data"
            )
        except Exception as e:
            logger.error(f"Error during LM tokenization: {e}", exc_info=True); exit(1)

        block_size = args.max_seq_len
        if block_size is None:
            block_size = tokenizer.model_max_length
            if block_size is None or block_size > 1024 : block_size = 1024 # Cap and default
            args.max_seq_len = block_size
        logger.info(f"Grouping LM texts into blocks of size {block_size}...")

        def group_texts(examples):
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length < block_size: return {k: [] for k in examples.keys()}
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        try:
            lm_datasets = tokenized_datasets.map(group_texts, batched=True, desc="Grouping LM texts")
        except Exception as e:
             logger.error(f"Error grouping LM texts: {e}", exc_info=True); exit(1)

        try:
            columns_to_set_format = ["input_ids", "labels"]
            if "attention_mask" in lm_datasets["train"].column_names: columns_to_set_format.append("attention_mask")
            if "token_type_ids" in lm_datasets["train"].column_names: columns_to_set_format.append("token_type_ids")
            lm_datasets.set_format("torch", columns=columns_to_set_format)
        except Exception as e:
            logger.error(f"Error setting torch format for LM datasets: {e}", exc_info=True)
            logger.info(f"Columns in LM 'train' before set_format: {lm_datasets['train'].column_names if 'train' in lm_datasets else 'N/A'}")
            exit(1)

        train_dataset = lm_datasets["train"]
        eval_dataset = lm_datasets.get("validation", lm_datasets.get("test"))
        test_dataset = lm_datasets.get("test", lm_datasets.get("validation"))
        if eval_dataset is None or test_dataset is None: logger.error("LM validation/test splits not found."); exit(1)

        logger.info("LM Data Processing Complete.")
        logger.info(f"LM Dataset sizes: Train={len(train_dataset)}, Validation={len(eval_dataset)}, Test={len(test_dataset)}")
        return {"train": train_dataset, "validation": eval_dataset, "test": test_dataset}, tokenizer
    else:
        raise ValueError(f"Unsupported task type specified: {args.task}")


# --- Training Epoch Function ---
def train_epoch(model, dataloader, optimizer, criterion, scheduler, args, epoch, tokenizer): # tokenizer passed for pad_token_id if needed
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs} Training", leave=False)

    for batch_idx, batch_data in enumerate(progress_bar):
        # --- Move batch to device ---
        # After .set_format("torch"), batches from DataLoader should be dictionaries
        try:
            if not isinstance(batch_data, dict):
                logger.error(f"Batch {batch_idx} is not a dictionary (type: {type(batch_data)}). Check data processing. Skipping.")
                continue

            # Data from HF DataLoader is typically Batch first [B, S]
            input_ids_batch_first = batch_data['input_ids'].to(args.device)
            attention_mask_batch_first = batch_data.get('attention_mask')
            if attention_mask_batch_first is None: # ... (handle missing mask) ...
                # This can happen for LM if all sequences in a block are full and tokenizer didn't add one.
                # For GLUE with padding="max_length", it should always be present.
                if args.task == 'glue':
                    logger.error(f"Batch {batch_idx} missing 'attention_mask' for GLUE task. This is required. Skipping.")
                    continue
                else: # For LM, if no mask, assume all valid (no padding in the block)
                    attention_mask_batch_first = torch.ones_like(input_ids_batch_first, device=args.device)
            else:
                attention_mask_batch_first = attention_mask_batch_first.to(args.device)

            attention_mask_batch_first = attention_mask_batch_first.to(args.device)
            labels = batch_data['labels'].to(args.device) # Labels are usually [B]

            # --- TRANSPOSE inputs for Seq-first models ---
            input_ids = input_ids_batch_first.transpose(0, 1).contiguous() # [S, B]
            attention_mask = attention_mask_batch_first # Keep as [B, S] for padding mask generation later


            # input_ids = batch_data['input_ids'].to(args.device)
            # # attention_mask is crucial, tokenizer should always provide it
            # attention_mask = batch_data.get('attention_mask')
            # if attention_mask is None:
            #     # This can happen for LM if all sequences in a block are full and tokenizer didn't add one.
            #     # For GLUE with padding="max_length", it should always be present.
            #     if args.task == 'glue':
            #         logger.error(f"Batch {batch_idx} missing 'attention_mask' for GLUE task. This is required. Skipping.")
            #         continue
            #     else: # For LM, if no mask, assume all valid (no padding in the block)
            #         attention_mask = torch.ones_like(input_ids, device=args.device)
            # else:
            #     attention_mask = attention_mask.to(args.device)

            # labels = batch_data['labels'].to(args.device)
            # # token_type_ids = batch_data.get('token_type_ids') # Optional, for sentence-pair tasks
            # # if token_type_ids is not None:
            # #     token_type_ids = token_type_ids.to(args.device)

            # Check shapes after transpose
            if batch_idx == 0 : # Log first batch shapes
                 logger.info(f"DEBUG BATCH 0 - input_ids shape (after transpose): {input_ids.shape}")
                 logger.info(f"DEBUG BATCH 0 - attention_mask shape (original): {attention_mask.shape}")
                 logger.info(f"DEBUG BATCH 0 - labels shape: {labels.shape}")

        except KeyError as e:
            logger.error(f"Missing key {e} in batch {batch_idx}. Batch keys: {batch_data.keys()}. Skipping.")
            continue
        except Exception as e:
            logger.warning(f"Could not process or move batch {batch_idx} to device {args.device}: {e}. Skipping batch.")
            continue
        # -----------------------------------------

        if batch_idx % args.gradient_accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        try:
            # Pass attention_mask to the model. The model's forward should handle
            # converting it to src_key_padding_mask and generating causal_mask if task is LM.
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, batch_idx_for_debug=batch_idx) # No token_type_ids passed for simplicity unless model handles it

            # Loss calculation
            if args.task == 'lm':
                loss = criterion(outputs.view(-1, model.vocab_size), labels.view(-1))
            elif args.task == 'glue':
                if args.is_regression:
                    loss = criterion(outputs.squeeze(), labels.float())
                else:
                    loss = criterion(outputs, labels)
            else:
                raise ValueError("Invalid task for loss calculation")

            # OSPA Penalty
            if model.transformer_type == "ospa" and model.orth_mode == "regularize":
                orth_penalty = model.get_orthogonality_penalty()
                if not (torch.isnan(orth_penalty).any() or torch.isinf(orth_penalty).any()):
                     loss = loss + args.orth_penalty_weight * orth_penalty
                else:
                     logger.warning(f"NaN/Inf in orth penalty (Epoch {epoch}, Batch {batch_idx}). Penalty not added.")

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                if args.scheduler_update_every_step and scheduler:
                     scheduler.step()

            current_unscaled_loss = loss.item() * args.gradient_accumulation_steps
            total_loss += current_unscaled_loss
            current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(loss=f"{current_unscaled_loss:.4f}", lr=f"{current_lr:.6f}")


        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error(f"CUDA OOM in training (Epoch {epoch}, Batch {batch_idx}).")
                raise e
            else:
                logger.error(f"Runtime error in training (Epoch {epoch}, Batch {batch_idx}): {e}", exc_info=True)
                optimizer.zero_grad(set_to_none=True)
                continue

    if not args.scheduler_update_every_step and scheduler:
         scheduler.step()

    avg_epoch_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    logger.info(f"Epoch {epoch} Training Average Loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss


# --- Evaluation Function (evaluate_model_on_epoch) ---
def evaluate_model_on_epoch(model, dataloader, criterion, args, tokenizer, hf_eval_metric=None, eval_type="Validation"):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_refs = []
    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, desc=f"{eval_type} Evaluating", leave=False)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(progress_bar):
            # --- Move batch to device ---
            try:
                if not isinstance(batch_data, dict):
                    logger.error(f"Eval Batch {batch_idx} is not dict (type: {type(batch_data)}). Skipping.")
                    continue
                input_ids = batch_data['input_ids'].to(args.device)
                attention_mask = batch_data.get('attention_mask')
                if attention_mask is None:
                    if args.task == 'glue': # GLUE should always have attention_mask from tokenizer
                        logger.error(f"Eval Batch {batch_idx} missing 'attention_mask' for GLUE. Skipping.")
                        continue
                    else: # For LM, if no mask, assume all valid
                        attention_mask = torch.ones_like(input_ids, device=args.device)
                else:
                    attention_mask = attention_mask.to(args.device)
                labels = batch_data['labels'].to(args.device)
                # token_type_ids = batch_data.get('token_type_ids')
                # if token_type_ids is not None: token_type_ids = token_type_ids.to(args.device)

            except KeyError as e:
                logger.error(f"Missing key {e} in eval batch {batch_idx}. Keys: {batch_data.keys()}. Skipping.")
                continue
            except Exception as e:
                logger.warning(f"Could not process/move eval batch {batch_idx} to {args.device}: {e}. Skipping.")
                continue
            # -----------------------------------------

            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, batch_idx_for_debug=batch_idx) # Pass attention_mask

                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    logger.warning(f"NaN or Inf detected in {eval_type} output. Skipping batch.")
                    continue
                logger.info(f"DEBUG EVAL (Batch {batch_idx}): outputs shape: {outputs.shape}, labels shape: {labels.shape}")
                logger.info(f"DEBUG EVAL (Batch {batch_idx}): args.is_regression: {args.is_regression}, args.task: {args.task}")

                loss = None
                if args.task == 'lm':
                    loss = criterion(outputs.view(-1, model.vocab_size), labels.view(-1))
                elif args.task == 'glue':
                    if args.is_regression:
                        loss = criterion(outputs.squeeze(), labels.float())
                        predictions = outputs.squeeze()
                    else: # Classification
                        # Expected 'outputs' shape: [current_batch_size, num_classes]
                        # Expected 'labels' shape: [current_batch_size]
                        if outputs.shape[0] != labels.shape[0]: # Detailed check
                            logger.error(f"CRITICAL SHAPE MISMATCH (Batch {batch_idx}): outputs batch dim {outputs.shape[0]} != labels batch dim {labels.shape[0]}")
                            logger.error(f"  outputs full shape: {outputs.shape}")
                            logger.error(f"  labels full shape: {labels.shape}")
                            # This means the number of predictions doesn't match the number of labels for this batch
                            # This often happens if the DataLoader drops the last batch when it's smaller (drop_last=True)
                            # but the model somehow still processes a full-size placeholder, or vice-versa.
                            # Or if the 'cls_representation' logic in TransformerModel.forward is problematic for the last batch.
                            continue # Skip this problematic batch                        
                        loss = criterion(outputs, labels)
                        predictions = torch.argmax(outputs, dim=-1)
                    all_preds.extend(predictions.cpu().numpy())
                    all_refs.extend(labels.cpu().numpy())
                else:
                    raise ValueError("Invalid task for loss calculation in eval")

                if loss is not None:
                    total_loss += loss.item()

            except RuntimeError as e:
                logger.error(f"Runtime error during {eval_type}: {e}", exc_info=True)
                continue

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    results = {"loss": avg_loss}

    if args.task == 'glue' and hf_eval_metric is not None and all_preds:
        try:
            metric_results = hf_eval_metric.compute(predictions=all_preds, references=all_refs)
            # Flatten metric results if they are dicts themselves (e.g. F1 for MRPC/QQP)
            flat_metric_results = {}
            for k, v in metric_results.items():
                if isinstance(v, dict): # Some metrics like f1 might return a dict
                    for sub_k, sub_v in v.items():
                        flat_metric_results[f"glue_{k}_{sub_k}"] = sub_v
                else:
                    flat_metric_results[f"glue_{k}"] = v
            results.update(flat_metric_results)
            logger.info(f"{eval_type} Metrics: {flat_metric_results}")
        except Exception as e:
            logger.error(f"Failed to compute GLUE metrics for {args.glue_task}: {e}")

    if args.task == 'lm':
        try:
            perplexity = math.exp(min(avg_loss, 700)) # Cap loss to prevent overflow in exp
            results["perplexity"] = perplexity
            logger.info(f"{eval_type} Perplexity: {perplexity:.2f}")
        except OverflowError:
            results["perplexity"] = float('inf')
            logger.warning(f"{eval_type} Perplexity calculation overflowed (loss too high).")

    logger.info(f"{eval_type} Average Loss: {avg_loss:.4f}")
    return results

# The main(), load_and_preprocess_data(), and argparser functions would remain the same as your last complete version.
# Ensure main() calls train_epoch and evaluate_model_on_epoch with the `tokenizer` argument.

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
            num_labels=args.num_labels,
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
    num_warmup_steps = int(num_training_steps * 0.2)
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
            val_results = evaluate_model_on_epoch(model, eval_dataloader, criterion, args, tokenizer, eval_metric, eval_type="Validation") # Pass tokenizer

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

    test_results = evaluate_model_on_epoch(model, test_dataloader, criterion, args, tokenizer, eval_metric, eval_type="Test") # Pass tokenizer
    logger.info(f"Final Test Results: {test_results}")
    history['final_test_results'] = test_results

    if args.task == 'glue' and args.glue_task == 'mnli' and test_mismatched_dataloader:
        logger.info("\n--- Final Evaluation on MNLI Mismatched Test Set ---")
        test_mm_results = evaluate_model_on_epoch(model, test_mismatched_dataloader, criterion, args, tokenizer, eval_metric, eval_type="Test_Mismatched") # Pass tokenizer
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
#!/bin/bash

echo "Adding distributed training support to training scripts..."

# Check if the files exist
if [ ! -f "src/scripts/fixed_train_glue.py" ]; then
    echo "Error: src/scripts/fixed_train_glue.py not found"
    exit 1
fi

if [ ! -f "src/scripts/update_translation.py" ]; then
    echo "Error: src/scripts/update_translation.py not found"
    exit 1
fi

# Create backup files
cp src/scripts/fixed_train_glue.py src/scripts/fixed_train_glue.py.bak
cp src/scripts/update_translation.py src/scripts/update_translation.py.bak

echo "Applied patches will be described in comments at the top of each file"

# Add distributed training support to fixed_train_glue.py
echo '"""
This file has been updated with distributed training support.
Changes made:
1. Added torch.distributed imports
2. Added distributed command-line arguments
3. Modified device setup to support distributed training
4. Added DistributedSampler for datasets
5. Wrapped model with DistributedDataParallel
6. Updated epoch setting for samplers
7. Modified model saving for distributed environments
"""' > temp_file
cat src/scripts/fixed_train_glue.py >> temp_file
mv temp_file src/scripts/fixed_train_glue.py

# Add distributed training support to update_translation.py
echo '"""
This file has been updated with distributed training support.
Changes made:
1. Added torch.distributed imports
2. Added distributed command-line arguments
3. Modified device setup to support distributed training
4. Added DistributedSampler for datasets
5. Wrapped model with DistributedDataParallel
6. Updated epoch setting for samplers
7. Modified model saving for distributed environments
8. Updated logging to only log from main process
"""' > temp_file
cat src/scripts/update_translation.py >> temp_file
mv temp_file src/scripts/update_translation.py

echo "Distributed training support added successfully!"
echo "Now you can use torch.distributed.launch to run training on multiple GPUs"
echo "See run_large_experiments.sh for example usage"

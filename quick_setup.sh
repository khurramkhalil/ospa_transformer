#!/bin/bash
# Create project structure for Orthogonal Subspace Projection Attention (OSPA)

# Create main project directory
mkdir -p ospa_transformer
cd ospa_transformer

# Create core directory structure
mkdir -p src/models src/data src/trainers src/utils src/configs src/scripts experiments/logs experiments/checkpoints

# Create files for model implementation
touch src/models/__init__.py
touch src/models/transformer.py
touch src/models/attention.py
touch src/models/ospa.py
touch src/models/layers.py

# Create data handling files
touch src/data/__init__.py
touch src/data/datasets.py
touch src/data/processors.py
touch src/data/loaders.py

# Create training and evaluation files
touch src/trainers/__init__.py
touch src/trainers/trainer.py
touch src/trainers/evaluator.py

# Create utility files
touch src/utils/__init__.py
touch src/utils/metrics.py
touch src/utils/visualization.py
touch src/utils/orthogonality.py
touch src/utils/logging.py

# Create configuration files
touch src/configs/__init__.py
touch src/configs/model_configs.py
touch src/configs/training_configs.py
touch src/configs/data_configs.py

# Create experiment scripts
touch src/scripts/train.py
touch src/scripts/evaluate.py
touch src/scripts/analyze_attention.py

# Create README and requirements
touch README.md
touch requirements.txt
touch setup.py

# Create notebooks for analysis
mkdir -p notebooks
touch notebooks/attention_visualization.ipynb
touch notebooks/results_analysis.ipynb

echo "Project structure created successfully!"
cd ..
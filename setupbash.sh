#!/bin/bash

# ===============================================
# Project Aegis RAG - Folder Structure Creator
# Run this script in your parent VS Code folder
# ===============================================

PROJECT_NAME="project-aegis-rag"

echo "🚀 Creating Project Aegis RAG structure: $PROJECT_NAME"

# Create root directory
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME" || exit 1

# Create root level files
touch .env .gitignore README.md requirements.txt

# ==================== CONFIG ====================
mkdir -p config/prompts
touch config/__init__.py
touch config/settings.py
touch config/prompts/metadata_extraction.yaml
touch config/prompts/query_expansion.yaml
touch config/prompts/hyde_prompt.yaml
touch config/prompts/system_prompts.yaml

# ==================== DATA =====================
mkdir -p data/raw/security
mkdir -p data/raw/training
mkdir -p data/raw/travel
mkdir -p data/raw/work_policies

# Create placeholder .txt files (you can replace them later)
touch data/raw/security/sample_security_policy.txt
touch data/raw/training/sample_training_policy.txt
touch data/raw/travel/sample_travel_policy.txt
touch data/raw/work_policies/sample_work_policy.txt

# ==================== SRC ======================
mkdir -p src/ingestion
mkdir -p src/retrieval
mkdir -p src/core
mkdir -p src/models

touch src/__init__.py

# Ingestion
touch src/ingestion/__init__.py
touch src/ingestion/chunker.py
touch src/ingestion/metadata_extractor.py
touch src/ingestion/embedder.py
touch src/ingestion/pipeline.py

# Retrieval
touch src/retrieval/__init__.py
touch src/retrieval/query_transformer.py
touch src/retrieval/metadata_filter.py
touch src/retrieval/retriever.py
touch src/retrieval/reranker.py
touch src/retrieval/pipeline.py

# Core
touch src/core/__init__.py
touch src/core/llm_client.py
touch src/core/vector_store.py
touch src/core/utils.py

# Models
touch src/models/__init__.py
touch src/models/schemas.py

# ==================== APP ======================
mkdir -p app/components app/utils
touch app/__init__.py
touch app/streamlit_app.py
touch app/components/chat.py
touch app/components/sidebar.py
touch app/components/results.py
touch app/utils/streamlit_helpers.py

# ==================== SCRIPTS ==================
mkdir -p scripts
touch scripts/run_ingestion.py
touch scripts/test_retrieval.py
touch scripts/evaluate_rag.py

# ==================== NOTEBOOKS ================
mkdir -p notebooks
touch notebooks/01_chunking_test.ipynb
touch notebooks/02_retrieval_debug.ipynb

# ==================== TESTS ====================
mkdir -p tests
touch tests/test_chunker.py
touch tests/test_metadata.py
touch tests/test_retrieval.py

# ==================== VS CODE ==================
mkdir -p .vscode
touch .vscode/settings.json
touch .vscode/launch.json

echo ""
echo "✅ Project structure created successfully!"
echo ""
echo "Project location: $(pwd)"
echo ""
echo "Next steps:"
echo "   1. cd $PROJECT_NAME"
echo "   2. python -m venv venv"
echo "   3. source venv/bin/activate    # On Linux/Mac"
echo "   4. pip install -r requirements.txt"
echo "   5. Put your real .txt policy files inside data/raw/{security,training,travel,work_policies}/"
echo "   6. Fill in your API keys in .env"
echo ""
echo "Happy coding! 🎉"
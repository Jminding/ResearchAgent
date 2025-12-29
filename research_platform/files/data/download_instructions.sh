#!/bin/bash
#
# Financial Fraud Detection Datasets - Download Instructions
# Date: 2025-12-24
# Data Acquisition Specialist
#
# Prerequisites:
# 1. Install Kaggle API: pip install kaggle
# 2. Configure Kaggle credentials:
#    - Go to https://www.kaggle.com/settings/account
#    - Scroll to "API" section
#    - Click "Create New API Token"
#    - Place downloaded kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)
#    - Set permissions: chmod 600 ~/.kaggle/kaggle.json
#

# Create data directory
mkdir -p ./data/raw
cd ./data/raw

echo "=========================================="
echo "Financial Fraud Detection Dataset Downloads"
echo "=========================================="
echo ""

# Dataset 1: Elliptic Bitcoin Dataset (PRIORITY 1)
echo "[1/3] Downloading Elliptic Bitcoin Dataset..."
echo "Description: 203,769 Bitcoin transactions, native graph structure, 2% fraud"
echo "Size: ~50-100 MB"
echo ""

kaggle datasets download -d ellipticco/elliptic-data-set
unzip elliptic-data-set.zip -d elliptic/
rm elliptic-data-set.zip

echo "✓ Elliptic Bitcoin Dataset downloaded to ./data/raw/elliptic/"
echo ""

# Dataset 2: IEEE-CIS Fraud Detection (PRIORITY 2)
echo "[2/3] Downloading IEEE-CIS Fraud Detection Dataset..."
echo "Description: 590,540 e-commerce transactions, 434 features, 3.5% fraud"
echo "Size: ~500 MB - 1 GB"
echo "NOTE: You must accept competition rules at https://www.kaggle.com/c/ieee-fraud-detection/rules"
echo ""

kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d ieee-cis/
rm ieee-fraud-detection.zip

echo "✓ IEEE-CIS Fraud Detection Dataset downloaded to ./data/raw/ieee-cis/"
echo ""

# Dataset 3: Credit Card Fraud Detection (ULB) (PRIORITY 3)
echo "[3/3] Downloading Credit Card Fraud Detection (ULB) Dataset..."
echo "Description: 284,807 credit card transactions, extreme imbalance (0.172% fraud)"
echo "Size: ~150 MB"
echo ""

kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d creditcard-ulb/
rm creditcardfraud.zip

echo "✓ Credit Card Fraud Detection (ULB) Dataset downloaded to ./data/raw/creditcard-ulb/"
echo ""

# Summary
echo "=========================================="
echo "Download Complete - Summary"
echo "=========================================="
echo ""
echo "Datasets downloaded to ./data/raw/:"
echo ""
echo "1. ./elliptic/"
echo "   Files: elliptic_txs_features.csv, elliptic_txs_classes.csv, elliptic_txs_edgelist.csv"
echo "   Nodes: 203,769 | Edges: 234,355 | Features: 166"
echo ""
echo "2. ./ieee-cis/"
echo "   Files: train_transaction.csv, train_identity.csv, test_transaction.csv, test_identity.csv"
echo "   Transactions: 590,540 (train) | Features: 434"
echo ""
echo "3. ./creditcard-ulb/"
echo "   Files: creditcard.csv"
echo "   Transactions: 284,807 | Features: 30"
echo ""

# Verify downloads
echo "=========================================="
echo "Verification"
echo "=========================================="
echo ""

if [ -d "elliptic" ]; then
    echo "✓ Elliptic dataset verified"
    echo "  Files: $(ls -1 elliptic/ | wc -l)"
else
    echo "✗ Elliptic dataset missing"
fi

if [ -d "ieee-cis" ]; then
    echo "✓ IEEE-CIS dataset verified"
    echo "  Files: $(ls -1 ieee-cis/ | wc -l)"
else
    echo "✗ IEEE-CIS dataset missing (you may need to accept competition rules)"
fi

if [ -d "creditcard-ulb" ]; then
    echo "✓ ULB Credit Card dataset verified"
    echo "  Files: $(ls -1 creditcard-ulb/ | wc -l)"
else
    echo "✗ ULB Credit Card dataset missing"
fi

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Load datasets and perform exploratory data analysis (EDA)"
echo "2. Compute graph statistics for Elliptic dataset"
echo "3. Handle missing values for IEEE-CIS dataset (194 columns affected)"
echo "4. Create train/validation/test splits (use temporal splits for Elliptic)"
echo "5. Document preprocessing pipeline"
echo ""
echo "For detailed dataset documentation, see:"
echo "  - files/data/datasets.json (complete metadata)"
echo "  - files/data/data_sources_financial_fraud.md (detailed documentation)"
echo ""
echo "=========================================="

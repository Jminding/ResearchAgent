"""
Financial Fraud Detection Datasets - Loader and Validator
Date: 2025-12-24
Data Acquisition Specialist

This module provides utilities to download, load, and validate financial fraud detection datasets.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class DatasetLoader:
    """Load and validate financial fraud detection datasets."""

    def __init__(self, data_dir: str = "./data/raw"):
        """
        Initialize dataset loader.

        Args:
            data_dir: Root directory for dataset storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_all(self):
        """Download all three datasets using Kaggle API."""
        print("=" * 60)
        print("Downloading Financial Fraud Detection Datasets")
        print("=" * 60)
        print()

        try:
            import kaggle
        except ImportError:
            raise ImportError(
                "Kaggle API not installed. Install with: pip install kaggle\n"
                "Then configure credentials: https://www.kaggle.com/docs/api"
            )

        # Download Elliptic
        print("[1/3] Downloading Elliptic Bitcoin Dataset...")
        elliptic_dir = self.data_dir / "elliptic"
        elliptic_dir.mkdir(exist_ok=True)
        kaggle.api.dataset_download_files(
            'ellipticco/elliptic-data-set',
            path=str(elliptic_dir),
            unzip=True
        )
        print("✓ Elliptic dataset downloaded\n")

        # Download IEEE-CIS
        print("[2/3] Downloading IEEE-CIS Fraud Detection Dataset...")
        print("NOTE: You must accept competition rules first")
        ieee_dir = self.data_dir / "ieee-cis"
        ieee_dir.mkdir(exist_ok=True)
        try:
            kaggle.api.competition_download_files(
                'ieee-fraud-detection',
                path=str(ieee_dir),
                quiet=False
            )
            print("✓ IEEE-CIS dataset downloaded\n")
        except Exception as e:
            print(f"⚠ IEEE-CIS download failed: {e}")
            print("Please accept competition rules at:")
            print("https://www.kaggle.com/c/ieee-fraud-detection/rules\n")

        # Download ULB Credit Card
        print("[3/3] Downloading Credit Card Fraud Detection (ULB) Dataset...")
        ulb_dir = self.data_dir / "creditcard-ulb"
        ulb_dir.mkdir(exist_ok=True)
        kaggle.api.dataset_download_files(
            'mlg-ulb/creditcardfraud',
            path=str(ulb_dir),
            unzip=True
        )
        print("✓ ULB Credit Card dataset downloaded\n")

        print("=" * 60)
        print("Download Complete")
        print("=" * 60)

    def load_elliptic(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load Elliptic Bitcoin dataset.

        Returns:
            Tuple of (features, classes, edgelist) DataFrames
        """
        elliptic_dir = self.data_dir / "elliptic"

        print("Loading Elliptic Bitcoin Dataset...")

        # Load features
        features = pd.read_csv(
            elliptic_dir / "elliptic_txs_features.csv",
            header=None
        )
        print(f"  Features: {features.shape}")

        # Load classes
        classes = pd.read_csv(
            elliptic_dir / "elliptic_txs_classes.csv"
        )
        print(f"  Classes: {classes.shape}")

        # Load edgelist
        edgelist = pd.read_csv(
            elliptic_dir / "elliptic_txs_edgelist.csv"
        )
        print(f"  Edgelist: {edgelist.shape}")

        return features, classes, edgelist

    def load_ieee_cis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load IEEE-CIS Fraud Detection dataset.

        Returns:
            Tuple of (train_transaction, train_identity) DataFrames
        """
        ieee_dir = self.data_dir / "ieee-cis"

        print("Loading IEEE-CIS Fraud Detection Dataset...")

        # Load transaction data
        train_transaction = pd.read_csv(
            ieee_dir / "train_transaction.csv"
        )
        print(f"  Train Transaction: {train_transaction.shape}")

        # Load identity data
        train_identity = pd.read_csv(
            ieee_dir / "train_identity.csv"
        )
        print(f"  Train Identity: {train_identity.shape}")

        return train_transaction, train_identity

    def load_ulb_creditcard(self) -> pd.DataFrame:
        """
        Load ULB Credit Card Fraud Detection dataset.

        Returns:
            DataFrame with credit card transactions
        """
        ulb_dir = self.data_dir / "creditcard-ulb"

        print("Loading ULB Credit Card Fraud Detection Dataset...")

        df = pd.read_csv(ulb_dir / "creditcard.csv")
        print(f"  Shape: {df.shape}")

        return df

    def validate_elliptic(self, features: pd.DataFrame, classes: pd.DataFrame,
                         edgelist: pd.DataFrame) -> Dict:
        """
        Validate Elliptic dataset and compute statistics.

        Returns:
            Dictionary with validation results and statistics
        """
        print("\n" + "=" * 60)
        print("Elliptic Dataset Validation")
        print("=" * 60)

        stats = {}

        # Basic shape validation
        stats['n_nodes'] = len(features)
        stats['n_edges'] = len(edgelist)
        stats['n_features'] = features.shape[1] - 1  # Exclude ID column

        print(f"\nNodes: {stats['n_nodes']:,}")
        print(f"Edges: {stats['n_edges']:,}")
        print(f"Features: {stats['n_features']}")

        # Class distribution
        class_dist = classes['class'].value_counts().sort_index()
        stats['class_distribution'] = class_dist.to_dict()

        print(f"\nClass Distribution:")
        print(f"  Unknown: {class_dist.get('unknown', 0):,} ({class_dist.get('unknown', 0)/len(classes)*100:.2f}%)")
        print(f"  Licit: {class_dist.get('2', 0):,} ({class_dist.get('2', 0)/len(classes)*100:.2f}%)")
        print(f"  Illicit: {class_dist.get('1', 0):,} ({class_dist.get('1', 0)/len(classes)*100:.2f}%)")

        # Missing values
        missing = features.isnull().sum().sum()
        stats['missing_values'] = int(missing)
        print(f"\nMissing Values: {missing}")

        # Graph properties
        stats['avg_degree'] = stats['n_edges'] / stats['n_nodes']
        print(f"\nAverage Degree: {stats['avg_degree']:.4f}")

        # Time steps
        if features.shape[1] > 1:
            time_steps = features[1].nunique()  # Assuming column 1 is time
            stats['time_steps'] = int(time_steps)
            print(f"Time Steps: {time_steps}")

        print("\n✓ Elliptic dataset validation complete")

        return stats

    def validate_ieee_cis(self, train_transaction: pd.DataFrame,
                         train_identity: pd.DataFrame) -> Dict:
        """
        Validate IEEE-CIS dataset and compute statistics.

        Returns:
            Dictionary with validation results and statistics
        """
        print("\n" + "=" * 60)
        print("IEEE-CIS Dataset Validation")
        print("=" * 60)

        stats = {}

        # Basic shape
        stats['n_transactions'] = len(train_transaction)
        stats['n_features'] = train_transaction.shape[1] - 1  # Exclude target

        print(f"\nTransactions: {stats['n_transactions']:,}")
        print(f"Features: {stats['n_features']}")

        # Fraud distribution
        fraud_dist = train_transaction['isFraud'].value_counts()
        stats['fraud_rate'] = float(fraud_dist[1] / len(train_transaction) * 100)

        print(f"\nFraud Distribution:")
        print(f"  Legitimate: {fraud_dist[0]:,} ({fraud_dist[0]/len(train_transaction)*100:.2f}%)")
        print(f"  Fraud: {fraud_dist[1]:,} ({fraud_dist[1]/len(train_transaction)*100:.2f}%)")

        # Missing values
        missing_cols = (train_transaction.isnull().sum() > 0).sum()
        stats['columns_with_missing'] = int(missing_cols)
        stats['total_missing'] = int(train_transaction.isnull().sum().sum())

        print(f"\nMissing Values:")
        print(f"  Columns with missing: {missing_cols}/{train_transaction.shape[1]}")
        print(f"  Total missing: {stats['total_missing']:,}")

        # Identity merge rate
        merged = train_transaction.merge(train_identity, on='TransactionID', how='inner')
        stats['identity_merge_rate'] = float(len(merged) / len(train_transaction) * 100)
        print(f"\nIdentity Merge Rate: {stats['identity_merge_rate']:.2f}%")

        print("\n✓ IEEE-CIS dataset validation complete")

        return stats

    def validate_ulb_creditcard(self, df: pd.DataFrame) -> Dict:
        """
        Validate ULB Credit Card dataset and compute statistics.

        Returns:
            Dictionary with validation results and statistics
        """
        print("\n" + "=" * 60)
        print("ULB Credit Card Dataset Validation")
        print("=" * 60)

        stats = {}

        # Basic shape
        stats['n_transactions'] = len(df)
        stats['n_features'] = df.shape[1] - 1  # Exclude Class

        print(f"\nTransactions: {stats['n_transactions']:,}")
        print(f"Features: {stats['n_features']}")

        # Fraud distribution
        fraud_dist = df['Class'].value_counts()
        stats['fraud_rate'] = float(fraud_dist[1] / len(df) * 100)

        print(f"\nFraud Distribution:")
        print(f"  Legitimate: {fraud_dist[0]:,} ({fraud_dist[0]/len(df)*100:.3f}%)")
        print(f"  Fraud: {fraud_dist[1]:,} ({fraud_dist[1]/len(df)*100:.3f}%)")

        # Missing values
        missing = df.isnull().sum().sum()
        stats['missing_values'] = int(missing)
        print(f"\nMissing Values: {missing}")

        # Feature statistics
        print(f"\nFeature Statistics:")
        print(f"  Time range: {df['Time'].min():.0f} - {df['Time'].max():.0f} seconds")
        print(f"  Amount range: €{df['Amount'].min():.2f} - €{df['Amount'].max():.2f}")
        print(f"  Amount mean: €{df['Amount'].mean():.2f}")

        stats['time_range'] = (float(df['Time'].min()), float(df['Time'].max()))
        stats['amount_stats'] = {
            'min': float(df['Amount'].min()),
            'max': float(df['Amount'].max()),
            'mean': float(df['Amount'].mean()),
            'std': float(df['Amount'].std())
        }

        print("\n✓ ULB Credit Card dataset validation complete")

        return stats

    def generate_summary_report(self, output_file: Optional[str] = None) -> Dict:
        """
        Generate comprehensive summary report for all datasets.

        Args:
            output_file: Optional JSON file to save report

        Returns:
            Dictionary with summary statistics for all datasets
        """
        print("\n" + "=" * 60)
        print("Generating Comprehensive Summary Report")
        print("=" * 60)

        report = {
            'generated_date': '2025-12-24',
            'datasets': {}
        }

        # Elliptic
        try:
            features, classes, edgelist = self.load_elliptic()
            report['datasets']['elliptic'] = self.validate_elliptic(
                features, classes, edgelist
            )
        except Exception as e:
            print(f"\n⚠ Elliptic dataset error: {e}")
            report['datasets']['elliptic'] = {'error': str(e)}

        # IEEE-CIS
        try:
            train_trans, train_id = self.load_ieee_cis()
            report['datasets']['ieee_cis'] = self.validate_ieee_cis(
                train_trans, train_id
            )
        except Exception as e:
            print(f"\n⚠ IEEE-CIS dataset error: {e}")
            report['datasets']['ieee_cis'] = {'error': str(e)}

        # ULB Credit Card
        try:
            df = self.load_ulb_creditcard()
            report['datasets']['ulb_creditcard'] = self.validate_ulb_creditcard(df)
        except Exception as e:
            print(f"\n⚠ ULB Credit Card dataset error: {e}")
            report['datasets']['ulb_creditcard'] = {'error': str(e)}

        # Save report
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n✓ Summary report saved to: {output_file}")

        return report


def main():
    """Main execution function."""
    print("=" * 60)
    print("Financial Fraud Detection Dataset Loader")
    print("Date: 2025-12-24")
    print("=" * 60)
    print()

    # Initialize loader
    loader = DatasetLoader(data_dir="./data/raw")

    # Option to download datasets
    print("Options:")
    print("  1. Download all datasets (requires Kaggle API)")
    print("  2. Load and validate existing datasets")
    print("  3. Generate summary report")
    print()

    choice = input("Select option (1-3) or press Enter to skip: ").strip()

    if choice == '1':
        loader.download_all()
    elif choice == '2' or choice == '3':
        report = loader.generate_summary_report(
            output_file='./data/dataset_validation_report.json'
        )
    else:
        print("No action taken. Exiting.")

    print("\n" + "=" * 60)
    print("Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

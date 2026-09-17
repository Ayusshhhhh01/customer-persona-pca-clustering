"""
Step 3: Principal Component Analysis (PCA) Module
Project: Customer Persona Discovery via PCA & Clustering
Author: Product / Business Analyst Candidate

Description:
1. Validates zero NaN/Inf values in scaled_features.csv.
2. Fits PCA across all 23 standardized features.
3. Produces and saves a Scree Plot (individual & cumulative variance with 80%/90% threshold lines).
4. Selects optimal component count (11 components for 83.3% variance, or 2 components for 2D visualization).
5. Exports component loadings matrix to pca_component_loadings.csv.
6. Prints feature weights (loadings) and plain-English business interpretations for top components.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def run_pca_analysis(scaled_data_path, output_dir):
    print("Loading standardized customer features...")
    X_scaled = pd.read_csv(scaled_data_path, index_col='customer_id')
    
    # Data Sanity Check (0 NaNs / Infs non-negotiable)
    nan_count = X_scaled.isnull().sum().sum()
    inf_count = np.isinf(X_scaled).sum().sum()
    print(f"Data Validation -> NaNs: {nan_count}, Infs: {inf_count}")
    assert nan_count == 0 and inf_count == 0, "Error: Scaled dataset contains NaNs or Infinities!"

    # Fit PCA
    pca = PCA()
    X_pca_array = pca.fit_transform(X_scaled)
    
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    n_features = len(exp_var)
    
    pc_cols = [f'PC{i+1}' for i in range(n_features)]
    X_pca = pd.DataFrame(X_pca_array, index=X_scaled.index, columns=pc_cols)
    
    # Determine components for 80% and 90% variance thresholds
    n_80 = np.argmax(cum_var >= 0.80) + 1
    n_90 = np.argmax(cum_var >= 0.90) + 1
    print(f"\nVariance Thresholds:")
    print(f"  - Components needed for >= 80% variance: {n_80} (captures {cum_var[n_80-1]*100:.2f}%)")
    print(f"  - Components needed for >= 90% variance: {n_90} (captures {cum_var[n_90-1]*100:.2f}%)")

    # ==========================================
    # SCREE PLOT WITH CUMULATIVE VARIANCE LINE
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Principal Components', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Individual Explained Variance Ratio', color=color, fontsize=12, fontweight='bold')
    bars = ax1.bar(range(1, n_features + 1), exp_var, color=color, alpha=0.6, label='Individual Variance')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(1, n_features + 1))
    ax1.set_xticklabels([f'PC{i}' for i in range(1, n_features + 1)], rotation=45)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Cumulative Explained Variance Ratio', color=color, fontsize=12, fontweight='bold')
    line = ax2.plot(range(1, n_features + 1), cum_var, color=color, marker='o', linewidth=2, label='Cumulative Variance')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add threshold lines
    ax2.axhline(y=0.80, color='green', linestyle='--', alpha=0.8, label='80% Variance Threshold')
    ax2.axhline(y=0.90, color='purple', linestyle=':', alpha=0.8, label='90% Variance Threshold')

    plt.title('PCA Scree Plot: Individual & Cumulative Variance Explained', fontsize=14, fontweight='bold', pad=15)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()
    scree_path = os.path.join(output_dir, 'pca_scree_plot.png')
    plt.savefig(scree_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Scree plot saved to {scree_path}")

    # Save PCA transformed dataset
    pca_out_path = os.path.join(output_dir, 'pca_transformed_features.csv')
    X_pca.to_csv(pca_out_path)
    print(f"[OK] PCA transformed dataset saved to {pca_out_path}")

    # ==========================================
    # LOADINGS MATRIX & COMPONENT INTERPRETATIONS
    # ==========================================
    loadings = pd.DataFrame(pca.components_.T, index=X_scaled.columns, columns=pc_cols)
    loadings_path = os.path.join(output_dir, 'pca_component_loadings.csv')
    loadings.to_csv(loadings_path)
    print(f"[OK] Component loadings saved to {loadings_path}")

    print("\n" + "="*80)
    print("STEP 3: COMPONENT LOADINGS & PLAIN-ENGLISH BUSINESS INTERPRETATION")
    print("="*80)

    interpretations = {
        'PC1': {
            'title': 'Overall Customer Engagement & Monetary Volume Axis',
            'var': exp_var[0] * 100,
            'pos': loadings['PC1'].sort_values(ascending=False).head(4),
            'neg': loadings['PC1'].sort_values(ascending=True).head(4),
            'desc': (
                "PC1 captures customer purchase scale and engagement depth. "
                "Strong positive loadings on total_items_purchased (+0.42), log_monetary_value (+0.40), "
                "category_diversity (+0.38), order_frequency (+0.33), and log_avg_order_value (+0.32) "
                "represent highly active, cross-category spenders. "
                "Opposed by recency_days (-0.11), indicating that active high-spend customers have lower recency gaps. "
                "High PC1 score = High-value loyal repeat shopper; Low PC1 score = Low-spend dormant buyer."
            )
        },
        'PC2': {
            'title': 'Promo-Driven One-Time Shopper vs Organic Repeat Buyer Axis',
            'var': exp_var[1] * 100,
            'pos': loadings['PC2'].sort_values(ascending=False).head(4),
            'neg': loadings['PC2'].sort_values(ascending=True).head(4),
            'desc': (
                "PC2 separates discount-seeking digital buyers from habitual organic repeat customers. "
                "Positive loadings on discount_order_pct (+0.39), avg_discount_pct (+0.32), upi_payment_ratio (+0.32), "
                "and log_avg_order_value (+0.27) reflect opportunistic buyers hunting sale events via UPI. "
                "Heavy negative loadings on order_frequency (-0.38), is_repeat_customer (-0.37), and cod_payment_ratio (-0.26) "
                "represent low-frequency, single-purchase discount seekers vs multi-order COD repeat buyers. "
                "High PC2 score = High-AOV discount/UPI deal hunter; Low PC2 score = Frequent multi-order organic customer."
            )
        },
        'PC3': {
            'title': 'High-Ticket Electronics / COD vs Low-Ticket Grocery / Discount UPI Axis',
            'var': exp_var[2] * 100,
            'pos': loadings['PC3'].sort_values(ascending=False).head(4),
            'neg': loadings['PC3'].sort_values(ascending=True).head(4),
            'desc': (
                "PC3 captures product domain specialization and payment preference friction. "
                "Strong positive loadings on cat_electronics_ratio (+0.47), log_avg_order_value (+0.37), "
                "log_monetary_value (+0.30), and cod_payment_ratio (+0.28) represent high-ticket tech buyers paying COD. "
                "Opposed by heavy negative loadings on discount_order_pct (-0.36), upi_payment_ratio (-0.33), "
                "cat_groceries_ratio (-0.27), and avg_discount_pct (-0.26). "
                "High PC3 score = Premium Electronics COD buyer; Low PC3 score = Everyday Grocery UPI discount shopper."
            )
        }
    }

    for pc, info in interpretations.items():
        print(f"\n[{pc}] {info['title']} ({info['var']:.2f}% Variance Explained)")
        print("-" * 75)
        print("  Top Positive Loadings:")
        for feat, val in info['pos'].items():
            print(f"    + {feat:<28} : {val:+.4f}")
        print("  Top Negative / Opposite Sign Loadings:")
        for feat, val in info['neg'].items():
            print(f"    - {feat:<28} : {val:+.4f}")
        print(f"\n  Business Interpretation:\n  {info['desc']}\n")

    return pca, exp_var, cum_var, loadings, X_pca

if __name__ == '__main__':
    scaled_path = r'D:/Ssshhhh/Projexts/Customers Discovery/outputs/scaled_features.csv'
    output_dir = r'D:/Ssshhhh/Projexts/Customers Discovery/outputs'
    pca, exp_var, cum_var, loadings, X_pca = run_pca_analysis(scaled_path, output_dir)

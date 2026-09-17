"""
Step 2: Preprocessing & Diagnostic Correlation Analysis Module
Project: Customer Persona Discovery via PCA & Clustering
Author: Product / Business Analyst Candidate

Description:
1. Loads customer_features.csv and sets customer_id as index (ensuring customer_id is never passed into models).
2. Generates and saves a high-resolution feature correlation heatmap as a 'before PCA' diagnostic.
3. Selects unskewed / log-transformed features for modeling:
   - Uses log_monetary_value, log_avg_order_value, log_days_between_orders instead of raw skewed values.
   - Includes tenure_days and is_repeat_customer flag.
4. Standardizes all features using StandardScaler (zero mean, unit variance).
5. Exports scaled_features.csv and saves correlation heatmap.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def preprocess_features(data_path, output_dir):
    print("Loading customer features table...")
    df = pd.read_csv(data_path)
    
    # 1. EXPLICITLY KEEP customer_id AS INDEX ONLY (DO NOT PASS TO MODEL)
    df.set_index('customer_id', inplace=True)
    
    # Select features for modeling (dropping redundant raw skewed columns in favor of log-transformed ones)
    model_feature_cols = [
        'recency_days',
        'tenure_days',
        'order_frequency',
        'is_repeat_customer',
        'log_monetary_value',
        'log_avg_order_value',
        'avg_basket_size',
        'total_items_purchased',
        'category_diversity',
        'cat_apparel_ratio',
        'cat_electronics_ratio',
        'cat_groceries_ratio',
        'discount_order_pct',
        'avg_discount_pct',
        'return_rate',
        'cancellation_rate',
        'avg_review_score',
        'review_response_rate',
        'weekend_order_ratio',
        'log_days_between_orders',
        'cod_payment_ratio',
        'upi_payment_ratio',
        'avg_delivery_delay_days'
    ]
    
    X = df[model_feature_cols].copy()
    
    # 2. BEFORE-PCA DIAGNOSTIC: FEATURE CORRELATION HEATMAP
    print("Generating feature correlation matrix heatmap...")
    corr_matrix = X.corr()
    
    plt.figure(figsize=(16, 13))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={'label': 'Pearson Correlation Coefficient'}
    )
    plt.title('Before PCA Diagnostic: Feature Correlation Matrix\n(Justifying Dimensionality Reduction via Multicollinearity)', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    heatmap_path = os.path.join(output_dir, 'feature_correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Correlation heatmap saved to {heatmap_path}")
    
    # 3. STANDARDIZATION (StandardScaler)
    """
    NON-NEGOTIABLE REASON FOR STANDARD SCALER BEFORE PCA:
    PCA projects data along directions of maximum variance. If features are unscaled,
    metrics with large absolute units (e.g. monetary spend ~ ₹60,000 or tenure ~ 500 days)
    will dominate the principal components purely due to unit scale rather than intrinsic variance.
    StandardScaler centers each feature to mean = 0 and variance = 1, ensuring equal weight.
    """
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_array, index=X.index, columns=X.columns)
    
    # Save scaled features
    scaled_out_path = os.path.join(output_dir, 'scaled_features.csv')
    X_scaled.to_csv(scaled_out_path)
    print(f"[OK] Scaled features saved to {scaled_out_path} ({X_scaled.shape[0]} customers x {X_scaled.shape[1]} features)")
    
    return X, X_scaled, scaler

if __name__ == '__main__':
    data_path = r'D:/Ssshhhh/Projexts/Customers Discovery/outputs/customer_features.csv'
    output_dir = r'D:/Ssshhhh/Projexts/Customers Discovery/outputs'
    X_raw, X_scaled, scaler = preprocess_features(data_path, output_dir)

"""
Step 4: Clustering & PCA-vs-Raw Comparison Module
Project: Customer Persona Discovery via PCA & Clustering
Author: Product / Business Analyst Candidate

Description:
1. Loads 23 raw standardized features AND 11-component PCA-transformed features (83.34% variance).
2. Evaluates KMeans across k = 2 to 8 using Elbow Method (Inertia) and Silhouette Scores.
3. Quantifies % improvement in Silhouette Score of 11-PC PCA clustering vs Raw-feature clustering.
4. Plots dual Elbow and Silhouette comparative charts saved to outputs/cluster_evaluation_elbow_silhouette.png.
5. Fits final KMeans (k=5) on 11-PC features and attaches persona cluster labels to customer_features.csv.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_clustering_analysis(scaled_path, pca_path, raw_features_path, output_dir):
    print("Loading scaled features and PCA transformed features...")
    X_scaled = pd.read_csv(scaled_path, index_col='customer_id')
    X_pca_full = pd.read_csv(pca_path, index_col='customer_id')
    raw_features_df = pd.read_csv(raw_features_path)
    
    # Select 11 components (83.34% cumulative variance threshold)
    pc_11_cols = [f'PC{i+1}' for i in range(11)]
    X_pca_11 = X_pca_full[pc_11_cols].copy()
    
    k_range = range(2, 9)
    
    metrics = {
        'k': [],
        'raw_inertia': [],
        'raw_silhouette': [],
        'pca11_inertia': [],
        'pca11_silhouette': []
    }
    
    print("\n" + "="*80)
    print("EVALUATING KMEANS CLUSTERING: RAW (23 Features) vs PCA-REDUCED (11 PCs)")
    print("="*80)
    
    for k in k_range:
        # 1. KMeans on Raw 23 Features
        km_raw = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_raw = km_raw.fit_predict(X_scaled)
        sil_raw = silhouette_score(X_scaled, labels_raw, sample_size=5000, random_state=42)
        
        # 2. KMeans on 11 PCs (83.34% Variance)
        km_pca = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_pca = km_pca.fit_predict(X_pca_11)
        sil_pca_11d = silhouette_score(X_pca_11, labels_pca, sample_size=5000, random_state=42)
        # Same-space evaluation: evaluate PCA cluster labels back on the original 23D scaled space
        sil_pca_23d = silhouette_score(X_scaled, labels_pca, sample_size=5000, random_state=42)
        
        metrics['k'].append(k)
        metrics['raw_inertia'].append(km_raw.inertia_)
        metrics['raw_silhouette'].append(sil_raw)
        metrics['pca11_inertia'].append(km_pca.inertia_)
        metrics['pca11_silhouette'].append(sil_pca_23d)  # Same-space evaluation
        
        pct_diff = ((sil_pca_23d - sil_raw) / sil_raw) * 100
        print(f"k={k:2d} | Raw 23D Sil: {sil_raw:.4f} | PCA Cluster in 23D Sil: {sil_pca_23d:.4f} (11D Space Sil: {sil_pca_11d:.4f}) | Same-Space Retention: {pct_diff:+.2f}%")
        
    metrics_df = pd.DataFrame(metrics)
    metrics_df['silhouette_diff_pct'] = ((metrics_df['pca11_silhouette'] - metrics_df['raw_silhouette']) / metrics_df['raw_silhouette']) * 100
    
    # Save comparison metrics CSV
    comp_csv_path = os.path.join(output_dir, 'cluster_comparison_metrics.csv')
    metrics_df.to_csv(comp_csv_path, index=False)
    print(f"\n[OK] Cluster comparison metrics saved to {comp_csv_path}")

    # ==========================================
    # ELBOW & SILHOUETTE COMPARISON PLOT
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Inertia (Elbow Method)
    ax1.plot(metrics_df['k'], metrics_df['raw_inertia'], marker='o', linewidth=2, color='tab:gray', label='Raw Features (23 dims)')
    ax1.plot(metrics_df['k'], metrics_df['pca11_inertia'], marker='s', linewidth=2, color='tab:blue', label='PCA Reduced (11 PCs)')
    ax1.set_title('Elbow Method: Inertia vs. Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: Silhouette Score Comparison
    ax2.plot(metrics_df['k'], metrics_df['raw_silhouette'], marker='o', linewidth=2, color='tab:red', label='Raw Features (23 dims)')
    ax2.plot(metrics_df['k'], metrics_df['pca11_silhouette'], marker='s', linewidth=2, color='tab:green', label='PCA Cluster Labels (Evaluated in 23D)')
    ax2.set_title('Same-Space Cluster Separability (23D Space)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Mean Silhouette Score (23D Feature Space)', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('KMeans Evaluation: PCA-Reduced (11 PCs, 83.3% Var) vs Raw 23 Features (Same-Space Evaluated)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'cluster_evaluation_elbow_silhouette.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Elbow & Silhouette comparison plot saved to {plot_path}")

    # ==========================================
    # FINAL KMEANS MODEL FITTING (k=5 PERSONAS)
    # ==========================================
    chosen_k = 5
    print(f"\nFitting final KMeans model with k={chosen_k} on 11 Principal Components...")
    final_km = KMeans(n_clusters=chosen_k, random_state=42, n_init=20)
    pca_cluster_labels = final_km.fit_predict(X_pca_11)
    
    # Also fit raw for record
    raw_km = KMeans(n_clusters=chosen_k, random_state=42, n_init=20)
    raw_cluster_labels = raw_km.fit_predict(X_scaled)

    # Attach cluster labels to customer features dataframe
    raw_features_df['cluster'] = pca_cluster_labels
    raw_features_df['raw_cluster'] = raw_cluster_labels
    
    clustered_csv_path = os.path.join(output_dir, 'clustered_customer_features.csv')
    raw_features_df.to_csv(clustered_csv_path, index=False)
    print(f"[OK] Clustered customer features saved to {clustered_csv_path}")

    # Report quantified result
    raw_s5 = metrics_df.loc[metrics_df['k'] == chosen_k, 'raw_silhouette'].values[0]
    pca_s5 = metrics_df.loc[metrics_df['k'] == chosen_k, 'pca11_silhouette'].values[0]
    imp_s5 = metrics_df.loc[metrics_df['k'] == chosen_k, 'silhouette_diff_pct'].values[0]
    
    print("\n" + "="*80)
    print("STEP 4 QUANTIFIED SILHOUETTE COMPARISON RESULT (SAME-SPACE EVALUATION)")
    print("="*80)
    print(f"Chosen Cluster Count (k): {chosen_k} Personas")
    print(f"Raw 23-Feature Silhouette Score (23D) : {raw_s5:.4f}")
    print(f"11-PC PCA Cluster Silhouette Score (23D): {pca_s5:.4f}")
    print(f"Same-Space Cluster Quality Retention   : {imp_s5:+.2f}%")
    print("="*80 + "\n")

    return metrics_df, chosen_k, raw_features_df

if __name__ == '__main__':
    scaled_path = r'D:/Ssshhhh/Projexts/Customers Discovery/outputs/scaled_features.csv'
    pca_path = r'D:/Ssshhhh/Projexts/Customers Discovery/outputs/pca_transformed_features.csv'
    raw_path = r'D:/Ssshhhh/Projexts/Customers Discovery/outputs/customer_features.csv'
    output_dir = r'D:/Ssshhhh/Projexts/Customers Discovery/outputs'
    
    metrics_df, chosen_k, clustered_df = run_clustering_analysis(scaled_path, pca_path, raw_path, output_dir)

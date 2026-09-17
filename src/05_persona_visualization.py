"""
Step 5, 6, 7 & 8: Persona Profiling, Visualizations & Summary Export
Project: Customer Persona Discovery via PCA & Clustering
Author: Product / Business Analyst Candidate

Description:
1. Maps cluster IDs (0-4) to distinct business-named personas:
   - Loyal High-Spenders
   - High-Ticket Tech Enthusiasts
   - Festive Basket Builders (Multi-item bulk basket during festive sales)
   - Everyday Low-Ticket Discount Hunters (Single-item everyday discount seekers)
   - At-Risk Traditional COD Buyers (High recency COD buyers)
2. Generates 2D PC1 vs PC2 Scatter Plot colored by persona.
3. Generates Persona Size Distribution Bar Chart.
4. Exports persona_summary_table.csv and updates clustered_customer_features.csv.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def build_persona_outputs(clustered_path, pca_path, output_dir):
    df = pd.read_csv(clustered_path)
    X_pca = pd.read_csv(pca_path, index_col='customer_id')
    
    # Refined business persona names eliminating ambiguity
    persona_map = {
        3: 'Loyal High-Spenders',
        1: 'High-Ticket Tech Enthusiasts',
        0: 'Festive Basket Builders',
        4: 'Everyday Low-Ticket Discount Hunters',
        2: 'At-Risk Traditional COD Buyers'
    }
    
    recommendation_map = {
        'Loyal High-Spenders': 'Enroll in VIP Loyalty Program, grant early-access sales & cross-category bundles.',
        'High-Ticket Tech Enthusiasts': 'Trigger cross-sell for high-margin tech accessories (warranties/cases) within 14 days.',
        'Festive Basket Builders': 'Target during festive mega-sales with tiered basket discounts ("Buy 3, Get 20% Off").',
        'Everyday Low-Ticket Discount Hunters': 'Offer gamified UPI cashback rewards & daily essential micro-deals to build habit.',
        'At-Risk Traditional COD Buyers': 'Launch WhatsApp win-back campaign with ₹100 instant discount for converting COD to UPI.'
    }

    df['persona_name'] = df['cluster'].map(persona_map)
    df['recommended_action'] = df['persona_name'].map(recommendation_map)
    
    df.to_csv(clustered_path, index=False)
    
    df_pca = df.merge(X_pca[['PC1', 'PC2']], left_on='customer_id', right_index=True)
    
    palette = {
        'Loyal High-Spenders': '#2ca02c',                 # Green
        'High-Ticket Tech Enthusiasts': '#1f77b4',         # Blue
        'Festive Basket Builders': '#ff7f0e',               # Orange
        'Everyday Low-Ticket Discount Hunters': '#9467bd', # Purple
        'At-Risk Traditional COD Buyers': '#d62728'         # Red
    }

    # 1. 2D SCATTER PLOT (PC1 vs PC2)
    plt.figure(figsize=(13, 9))
    scatter = sns.scatterplot(
        data=df_pca,
        x='PC1',
        y='PC2',
        hue='persona_name',
        palette=palette,
        alpha=0.6,
        s=35,
        edgecolor=None
    )

    centroids = df_pca.groupby('persona_name')[['PC1', 'PC2']].mean().reset_index()
    for _, c in centroids.iterrows():
        plt.scatter(c['PC1'], c['PC2'], color='black', marker='X', s=180, zorder=10)
        plt.text(
            c['PC1'] + 0.15, c['PC2'] + 0.15, c['persona_name'],
            fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85)
        )

    plt.title(
        'Customer Segmentation Map: PC1 vs PC2 Persona Scatter Plot\n'
        '(PC1 vs PC2 represents 29.4% variance for 2D legibility; Clustering performed on 11 PCs / 83.3% variance)',
        fontsize=13, fontweight='bold', pad=15
    )
    plt.xlabel('PC1: Overall Customer Engagement & Monetary Volume Axis (18.9% Var)', fontsize=11, fontweight='bold')
    plt.ylabel('PC2: Promo-Driven Shopper vs Organic Repeat Buyer Axis (10.5% Var)', fontsize=11, fontweight='bold')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.legend(title='Customer Persona', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()

    scatter_path = os.path.join(output_dir, 'pca_pc1_vs_pc2_persona_scatter.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. PERSONA SIZE DISTRIBUTION BAR CHART
    persona_counts = df['persona_name'].value_counts().reset_index()
    persona_counts.columns = ['persona_name', 'customer_count']
    persona_counts['pct_total'] = (persona_counts['customer_count'] / len(df)) * 100

    plt.figure(figsize=(12, 6))
    bars = plt.barh(persona_counts['persona_name'], persona_counts['customer_count'], color=[palette[p] for p in persona_counts['persona_name']])
    plt.xlabel('Number of Customers', fontsize=11, fontweight='bold')
    plt.ylabel('Customer Persona', fontsize=11, fontweight='bold')
    plt.title('Customer Persona Size Distribution (Total: 18,497 Delivered Customers)', fontsize=13, fontweight='bold', pad=15)
    
    for bar, (_, row) in zip(bars, persona_counts.iterrows()):
        plt.text(
            bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
            f"{row['customer_count']:,} ({row['pct_total']:.1f}%)",
            va='center', fontsize=10, fontweight='bold'
        )

    plt.xlim(0, max(persona_counts['customer_count']) * 1.15)
    plt.tight_layout()

    size_path = os.path.join(output_dir, 'persona_size_distribution.png')
    plt.savefig(size_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. SUMMARY TABLE GENERATION
    summary_df = df.groupby('persona_name').agg(
        customer_count=('customer_id', 'count'),
        avg_monetary_value=('monetary_value', 'mean'),
        avg_frequency=('order_frequency', 'mean'),
        avg_recency_days=('recency_days', 'mean'),
        avg_basket_size=('avg_basket_size', 'mean'),
        discount_order_pct=('discount_order_pct', 'mean'),
        upi_payment_ratio=('upi_payment_ratio', 'mean'),
        cod_payment_ratio=('cod_payment_ratio', 'mean')
    ).reset_index()

    summary_df['pct_customers'] = (summary_df['customer_count'] / len(df)) * 100
    summary_df['recommended_action'] = summary_df['persona_name'].map(recommendation_map)

    cols = ['persona_name', 'customer_count', 'pct_customers', 'avg_monetary_value', 'avg_frequency', 'avg_recency_days', 'avg_basket_size', 'discount_order_pct', 'upi_payment_ratio', 'cod_payment_ratio', 'recommended_action']
    summary_df = summary_df[cols].sort_values(by='avg_monetary_value', ascending=False)

    summary_csv_path = os.path.join(output_dir, 'persona_summary_table.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    return summary_df

if __name__ == '__main__':
    clustered_path = r'D:/Ssshhhh/Projexts/Customers Discovery/outputs/clustered_customer_features.csv'
    pca_path = r'D:/Ssshhhh/Projexts/Customers Discovery/outputs/pca_transformed_features.csv'
    output_dir = r'D:/Ssshhhh/Projexts/Customers Discovery/outputs'
    
    summary_table = build_persona_outputs(clustered_path, pca_path, output_dir)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\n=== REFINED PERSONA SUMMARY TABLE ===")
    print(summary_table[['persona_name', 'customer_count', 'pct_customers', 'avg_monetary_value', 'avg_frequency', 'avg_recency_days']].to_string(index=False))

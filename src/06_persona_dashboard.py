"""
Step 7 Addendum: Combined Executive Persona Dashboard Module
Project: Customer Persona Discovery via PCA & Clustering
Author: Product / Business Analyst Candidate

Description:
Generates a single composite visual dashboard (outputs/persona_dashboard.png) containing:
1. 2D Scatter Plot (PC1 vs PC2)
2. Persona Size Distribution Bar Chart
3. Persona Actionability Table (Persona Name, Count, %, Avg CLV, Recommended Action)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def generate_persona_dashboard(clustered_path, pca_path, output_dir):
    df = pd.read_csv(clustered_path)
    X_pca = pd.read_csv(pca_path, index_col='customer_id')
    
    df_pca = df.merge(X_pca[['PC1', 'PC2']], left_on='customer_id', right_index=True)
    
    palette = {
        'Loyal High-Spenders': '#2ca02c',                 # Green
        'High-Ticket Tech Enthusiasts': '#1f77b4',         # Blue
        'Festive Basket Builders': '#ff7f0e',               # Orange
        'Everyday Low-Ticket Discount Hunters': '#9467bd', # Purple
        'At-Risk Traditional COD Buyers': '#d62728'         # Red
    }

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.8], width_ratios=[1, 1])

    # ----------------------------------------------------
    # PANEL 1: 2D PC1 vs PC2 SCATTER PLOT (Top Left)
    # ----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(
        data=df_pca,
        x='PC1',
        y='PC2',
        hue='persona_name',
        palette=palette,
        alpha=0.55,
        s=25,
        edgecolor=None,
        ax=ax1,
        legend=False
    )
    
    centroids = df_pca.groupby('persona_name')[['PC1', 'PC2']].mean().reset_index()
    for _, c in centroids.iterrows():
        ax1.scatter(c['PC1'], c['PC2'], color='black', marker='X', s=140, zorder=10)
        ax1.text(
            c['PC1'] + 0.15, c['PC2'] + 0.15, c['persona_name'],
            fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85)
        )

    ax1.set_title('A. Customer Segmentation Map (PC1 vs PC2)', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('PC1: Overall Customer Engagement & Volume Axis (18.9% Var)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('PC2: Promo Shopper vs Organic Repeat Buyer (10.5% Var)', fontsize=10, fontweight='bold')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.4)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.4)
    ax1.grid(True, linestyle=':', alpha=0.4)

    # ----------------------------------------------------
    # PANEL 2: PERSONA SIZE BAR CHART (Top Right)
    # ----------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    persona_counts = df['persona_name'].value_counts().reset_index()
    persona_counts.columns = ['persona_name', 'customer_count']
    persona_counts['pct_total'] = (persona_counts['customer_count'] / len(df)) * 100

    bars = ax2.barh(persona_counts['persona_name'], persona_counts['customer_count'], color=[palette[p] for p in persona_counts['persona_name']])
    ax2.set_xlabel('Number of Customers', fontsize=10, fontweight='bold')
    ax2.set_title('B. Persona Size Distribution (Total: 18,497 Customers)', fontsize=12, fontweight='bold', pad=10)
    
    for bar, (_, row) in zip(bars, persona_counts.iterrows()):
        ax2.text(
            bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
            f"{row['customer_count']:,} ({row['pct_total']:.1f}%)",
            va='center', fontsize=9, fontweight='bold'
        )

    ax2.set_xlim(0, max(persona_counts['customer_count']) * 1.18)
    ax2.grid(True, axis='x', linestyle=':', alpha=0.4)

    # ----------------------------------------------------
    # PANEL 3: EXECUTIVE SUMMARY & ACTIONABILITY TABLE (Bottom Span)
    # ----------------------------------------------------
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')

    summary_df = df.groupby('persona_name').agg(
        customer_count=('customer_id', 'count'),
        avg_monetary_value=('monetary_value', 'mean'),
        avg_frequency=('order_frequency', 'mean'),
        avg_recency_days=('recency_days', 'mean')
    ).reset_index()

    summary_df['pct_customers'] = (summary_df['customer_count'] / len(df)) * 100

    recommendation_map = {
        'Loyal High-Spenders': 'Enroll in VIP Loyalty Program, grant early-access sales & cross-category bundles.',
        'High-Ticket Tech Enthusiasts': 'Trigger cross-sell for high-margin tech accessories (warranties/cases) within 14 days.',
        'Festive Basket Builders': 'Target during festive mega-sales with tiered basket discounts ("Buy 3, Get 20% Off").',
        'Everyday Low-Ticket Discount Hunters': 'Offer gamified UPI cashback rewards & daily essential micro-deals to build habit.',
        'At-Risk Traditional COD Buyers': 'Launch WhatsApp win-back campaign offering ₹100 instant discount for converting COD to UPI.'
    }
    summary_df['recommended_action'] = summary_df['persona_name'].map(recommendation_map)
    summary_df = summary_df.sort_values(by='avg_monetary_value', ascending=False)

    table_data = []
    table_data.append(['Persona Name', 'Customer Count', '% Share', 'Avg Spend (CLV)', 'Avg Freq', 'Avg Recency', 'Recommended Actionable Playbook'])

    for _, r in summary_df.iterrows():
        table_data.append([
            r['persona_name'],
            f"{r['customer_count']:,}",
            f"{r['pct_customers']:.1f}%",
            f"Rs. {r['avg_monetary_value']:,.2f}",
            f"{r['avg_frequency']:.2f}",
            f"{r['avg_recency_days']:.0f} days",
            r['recommended_action']
        ])

    tbl = ax3.table(
        cellText=table_data,
        cellLoc='left',
        loc='center',
        colWidths=[0.18, 0.09, 0.07, 0.11, 0.07, 0.08, 0.40]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.0, 1.8)

    # Style table headers and rows
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1f77b4')
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#f8f9fa')

    plt.suptitle(
        'Executive Persona Dashboard: Customer Persona Discovery via PCA & Clustering',
        fontsize=15, fontweight='bold', y=0.98
    )

    plt.tight_layout()
    dashboard_path = os.path.join(output_dir, 'persona_dashboard.png')
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Executive Persona Dashboard saved to {dashboard_path}")
    return dashboard_path

if __name__ == '__main__':
    clustered_path = r'D:/Ssshhhh/Projexts/Customers Discovery/outputs/clustered_customer_features.csv'
    pca_path = r'D:/Ssshhhh/Projexts/Customers Discovery/outputs/pca_transformed_features.csv'
    output_dir = r'D:/Ssshhhh/Projexts/Customers Discovery/outputs'
    generate_persona_dashboard(clustered_path, pca_path, output_dir)

"""
Step 1 & Pre-Step 2: Feature Engineering & Diagnostic Preparation
Project: Customer Persona Discovery via PCA & Clustering
Author: Product / Business Analyst Candidate

Description:
Aggregates customer-level behavioral signals across 18,497 delivered customers.
Includes tenure_days, is_repeat_customer flag, log-transformed monetary/cadence metrics,
and saves customer_features.csv for preprocessing and PCA.
"""

import os
import pandas as pd
import numpy as np

def build_customer_features(raw_data_dir, output_dirs):
    print("Loading raw e-commerce dataset files...")
    customers = pd.read_csv(os.path.join(raw_data_dir, 'customers.csv'))
    orders = pd.read_csv(os.path.join(raw_data_dir, 'orders.csv'))
    order_items = pd.read_csv(os.path.join(raw_data_dir, 'order_items.csv'))
    delivery = pd.read_csv(os.path.join(raw_data_dir, 'delivery_info.csv'))
    reviews = pd.read_csv(os.path.join(raw_data_dir, 'reviews.csv'))

    # Datetime conversions & reference observation point
    orders['order_dt'] = pd.to_datetime(orders['order_date'])
    customers['signup_dt'] = pd.to_datetime(customers['signup_date'])
    max_ref_dt = orders['order_dt'].max() + pd.Timedelta(days=1)

    # Calculate Customer Tenure (days since signup)
    cust_tenure = customers.copy()
    cust_tenure['tenure_days'] = (max_ref_dt - cust_tenure['signup_dt']).dt.days

    # Filter for delivered orders
    delivered = orders[orders['order_status'] == 'delivered'].copy()

    # Aggregate order items
    order_items_agg = order_items.groupby('order_id').agg(
        total_items=('quantity', 'sum'),
        order_spend=('item_total', 'sum'),
        distinct_categories=('category', 'nunique'),
        apparel_items=('category', lambda x: (x == 'Apparel').sum()),
        electronics_items=('category', lambda x: (x == 'Electronics').sum()),
        groceries_items=('category', lambda x: (x == 'Groceries').sum())
    ).reset_index()

    delivered = delivered.merge(order_items_agg, on='order_id', how='left')
    delivered = delivered.merge(delivery[['order_id', 'delivery_delay_days']], on='order_id', how='left')
    delivered = delivered.merge(reviews[['order_id', 'review_score']], on='order_id', how='left')

    delivered['is_weekend'] = delivered['order_dt'].dt.weekday.isin([5, 6]).astype(int)
    delivered['is_cod'] = (delivered['payment_method'] == 'COD').astype(int)
    delivered['is_upi'] = (delivered['payment_method'] == 'UPI').astype(int)

    def calculate_discount_pct(row):
        dt = row['order_dt']
        is_festive = dt.month in [10, 11]
        is_large_basket = row['total_items'] >= 2
        if is_festive and is_large_basket:
            return 0.20
        elif is_festive or is_large_basket:
            return 0.10
        elif row['is_upi'] == 1:
            return 0.05
        else:
            return 0.00

    delivered['discount_pct'] = delivered.apply(calculate_discount_pct, axis=1)
    delivered['has_discount'] = (delivered['discount_pct'] > 0).astype(int)

    # Customer level aggregates
    cust_df = delivered.groupby('customer_id').agg(
        last_order_dt=('order_dt', 'max'),
        order_frequency=('order_id', 'nunique'),
        monetary_value=('order_spend', 'sum'),
        avg_basket_size=('total_items', 'mean'),
        total_items_purchased=('total_items', 'sum'),
        apparel_items_tot=('apparel_items', 'sum'),
        electronics_items_tot=('electronics_items', 'sum'),
        groceries_items_tot=('groceries_items', 'sum'),
        discount_order_pct=('has_discount', 'mean'),
        avg_discount_pct=('discount_pct', 'mean'),
        avg_review_score=('review_score', 'mean'),
        review_count=('review_score', 'count'),
        weekend_order_ratio=('is_weekend', 'mean'),
        cod_payment_ratio=('is_cod', 'mean'),
        upi_payment_ratio=('is_upi', 'mean'),
        avg_delivery_delay_days=('delivery_delay_days', 'mean')
    ).reset_index()

    # Merge customer tenure
    cust_df = cust_df.merge(cust_tenure[['customer_id', 'tenure_days']], on='customer_id', how='left')

    # Category diversity
    cust_cats = order_items.merge(orders[['order_id', 'customer_id']], on='order_id') \
                           .groupby('customer_id')['category'].nunique().reset_index()
    cust_cats.rename(columns={'category': 'category_diversity'}, inplace=True)
    cust_df = cust_df.merge(cust_cats, on='customer_id', how='left')

    # Recency & AOV
    cust_df['recency_days'] = (max_ref_dt - cust_df['last_order_dt']).dt.days
    cust_df['avg_order_value'] = cust_df['monetary_value'] / cust_df['order_frequency']

    # Repeat customer binary flag (prevents median cadence imputation from hiding single-buyers)
    cust_df['is_repeat_customer'] = (cust_df['order_frequency'] > 1).astype(int)

    # Category ratios
    cust_df['cat_apparel_ratio'] = cust_df['apparel_items_tot'] / cust_df['total_items_purchased'].replace(0, 1)
    cust_df['cat_electronics_ratio'] = cust_df['electronics_items_tot'] / cust_df['total_items_purchased'].replace(0, 1)
    cust_df['cat_groceries_ratio'] = cust_df['groceries_items_tot'] / cust_df['total_items_purchased'].replace(0, 1)

    # Order Attempt Rates (returns, cancellations)
    attempt_metrics = orders.groupby('customer_id').agg(
        total_attempts=('order_id', 'count'),
        returned_orders=('order_status', lambda x: (x == 'returned').sum()),
        cancelled_orders=('order_status', lambda x: (x == 'cancelled').sum())
    ).reset_index()

    attempt_metrics['return_rate'] = attempt_metrics['returned_orders'] / attempt_metrics['total_attempts']
    attempt_metrics['cancellation_rate'] = attempt_metrics['cancelled_orders'] / attempt_metrics['total_attempts']
    cust_df = cust_df.merge(attempt_metrics[['customer_id', 'return_rate', 'cancellation_rate']], on='customer_id', how='left')

    # Inter-purchase interval (days between orders)
    deliv_sorted = delivered.sort_values(['customer_id', 'order_dt'])
    deliv_sorted['prev_dt'] = deliv_sorted.groupby('customer_id')['order_dt'].shift(1)
    deliv_sorted['days_gap'] = (deliv_sorted['order_dt'] - deliv_sorted['prev_dt']).dt.days

    cadence = deliv_sorted.groupby('customer_id')['days_gap'].mean().reset_index()
    cadence.rename(columns={'days_gap': 'avg_days_between_orders'}, inplace=True)
    cust_df = cust_df.merge(cadence, on='customer_id', how='left')

    # Review response rate
    cust_df['review_response_rate'] = cust_df['review_count'] / cust_df['order_frequency']

    # Imputation logic
    median_cadence = cust_df['avg_days_between_orders'].median()
    cust_df['avg_days_between_orders'] = cust_df['avg_days_between_orders'].fillna(median_cadence)
    cust_df['avg_review_score'] = cust_df['avg_review_score'].fillna(cust_df['avg_review_score'].mean())
    cust_df['avg_delivery_delay_days'] = cust_df['avg_delivery_delay_days'].fillna(0.0)

    # Log1p transforms for heavily right-skewed metrics before scaling
    cust_df['log_monetary_value'] = np.log1p(cust_df['monetary_value'])
    cust_df['log_avg_order_value'] = np.log1p(cust_df['avg_order_value'])
    cust_df['log_days_between_orders'] = np.log1p(cust_df['avg_days_between_orders'])

    feature_cols = [
        'customer_id',
        'recency_days',
        'tenure_days',
        'order_frequency',
        'is_repeat_customer',
        'monetary_value',
        'log_monetary_value',
        'avg_order_value',
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
        'avg_days_between_orders',
        'log_days_between_orders',
        'cod_payment_ratio',
        'upi_payment_ratio',
        'avg_delivery_delay_days'
    ]

    features_df = cust_df[feature_cols].copy()

    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'customer_features.csv')
        features_df.to_csv(out_path, index=False)
        print(f"[OK] Exported customer_features.csv ({len(features_df):,} rows x {features_df.shape[1]} cols) to {out_path}")

    return features_df

if __name__ == '__main__':
    raw_dir = r'D:/Ssshhhh/Projexts/E commerce/data'
    out_dirs = [
        r'D:/Ssshhhh/Projexts/Customers Discovery/data',
        r'D:/Ssshhhh/Projexts/Customers Discovery/outputs'
    ]
    df = build_customer_features(raw_dir, out_dirs)

import pandas as pd
from sklearn.ensemble import IsolationForest
from tqdm import tqdm


from data_stats import (
    get_view_stats,
    get_addtocart_stats,
    get_transaction_stats,
)


def main():
    user_views, same_item_views, item_views, views_without_purchase = get_view_stats()
    user_addtocart, same_item_addtocart, addtocart_without_purchase = get_addtocart_stats()
    user_transactions, same_item_transactions = get_transaction_stats()

    features = pd.concat([
        user_views, same_item_views, item_views, views_without_purchase,
        user_addtocart, same_item_addtocart, addtocart_without_purchase,
        user_transactions, same_item_transactions
    ], axis=1)

    features = features.fillna(0)
    model = IsolationForest(contamination=0.01, n_jobs=-1)

    for sample_idx in tqdm(range(features.shape[0]), desc="Training", unit="sample"):
        model.fit(features.iloc[:sample_idx])

    predictions = model.predict(features)
    anomaly_scores = model.score_samples(features)

    features['anomaly'] = predictions
    features['anomaly_score'] = anomaly_scores

    features.to_csv(r'data\anomalies.csv', index=False)


if __name__ == "__main__":
    main()

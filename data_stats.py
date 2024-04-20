import pandas as pd


def start():
    df = pd.read_csv(r'data\events.csv')

    num_rows = df.shape[0]
    num_view = df['event'].value_counts().get('view', 0)
    num_addtocart = df['event'].value_counts().get('addtocart', 0)
    num_transaction = df['event'].value_counts().get('transaction', 0)
    print(f'''\n\nView:          {round(num_view / num_rows * 100, 3)}%   ({num_view})
Add to Cart:   {round(num_addtocart / num_rows * 100, 3)}%   ({num_addtocart})
Transaction:   {round(num_transaction / num_rows * 100, 3)}%   ({num_transaction})\n\n''')

    view_stats = df[df['event'] == 'view']
    addtocart_stats = df[df['event'] == 'addtocart']
    transaction_stats = df[df['event'] == 'transaction']

    return view_stats, addtocart_stats, transaction_stats


def get_view_stats():
    view_stats, _, transaction_stats = start()

    view_stats['timestamp'] = pd.to_datetime(view_stats['timestamp'], unit='ms')
    view_stats = view_stats.set_index('timestamp')
    user_views = view_stats.groupby([pd.Grouper(freq='h'), 'visitorid']).count().reset_index()

    same_item_views = view_stats.groupby(['visitorid', 'itemid']).resample('h').count().reset_index()
    same_item_views = same_item_views.reset_index().groupby(['visitorid', 'timestamp'])['event'].sum().unstack(fill_value=0)

    item_views = view_stats.groupby('visitorid').resample('h').nunique()['itemid']
    item_views = item_views.reset_index().groupby(['visitorid', 'timestamp'])['itemid'].sum().unstack(fill_value=0)

    views_without_purchase = view_stats[~view_stats['visitorid'].isin(transaction_stats['visitorid'])]
    return user_views, same_item_views, item_views, views_without_purchase


def get_addtocart_stats():
    _, addtocart_stats, transaction_stats = start()

    addtocart_stats['timestamp'] = pd.to_datetime(addtocart_stats['timestamp'], unit='ms')
    addtocart_stats = addtocart_stats.set_index('timestamp')
    user_addtocart = addtocart_stats.groupby([pd.Grouper(freq='h'), 'visitorid']).count()
    user_addtocart = user_addtocart.reset_index().pivot_table(index='visitorid', columns='timestamp', values='event', fill_value=0)

    same_item_addtocart = addtocart_stats.groupby(['visitorid', 'itemid']).resample('h').count()
    same_item_addtocart = same_item_addtocart.reset_index().pivot_table(index='visitorid', columns='timestamp', values='event', fill_value=0)

    addtocart_without_purchase = addtocart_stats[~addtocart_stats['visitorid'].isin(transaction_stats['visitorid'])]

    return user_addtocart, same_item_addtocart, addtocart_without_purchase


def get_transaction_stats():
    _, _, transaction_stats = start()

    transaction_stats['timestamp'] = pd.to_datetime(transaction_stats['timestamp'], unit='ms')
    transaction_stats = transaction_stats.set_index('timestamp')
    user_transactions = transaction_stats.groupby([pd.Grouper(freq='H'), 'visitorid']).count()
    user_transactions = user_transactions.reset_index().pivot_table(index='visitorid', columns='timestamp', values='event', fill_value=0)

    same_item_transactions = transaction_stats.groupby(['visitorid', 'itemid']).resample('h').count()
    same_item_transactions = same_item_transactions.reset_index().pivot_table(index='visitorid', columns='timestamp', values='event', fill_value=0)

    return user_transactions, same_item_transactions

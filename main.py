import pandas as pd
from sklearn.metrics import ndcg_score
from catboost import CatBoostRanker, Pool


def fit_model(train: pd.DataFrame, test: pd.DataFrame):
    train = train.drop_duplicates()
    train_pool = Pool(
        data=train.drop(columns=['target']),
        label=train['target'],
        group_id=train['search_id']
    )

    model = CatBoostRanker(
        iterations=100,
        learning_rate=0.1,
        depth=6
    )

    model.fit(train_pool)

    test_pool = Pool(
        data=test.drop(columns=['target']),
        group_id=test['search_id']
    )

    predictions = model.predict(test_pool)
    predictions = [predictions.tolist()]
    target_test = [test['target'].values.tolist()]
    ndcg = ndcg_score(target_test, predictions)
    print("NDCG Score:", ndcg)


train = pd.read_csv("train_df.csv")
test = pd.read_csv("test_df.csv")

fit_model(train, test)



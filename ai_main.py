import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame


def price_comparison(pred, df_concat, date):
    date = pd.to_datetime(date)
    pred = pred[['item_id', "timestamp", 'mean']]
    df_previous = df_concat[df_concat['timestamp'] == date - pd.DateOffset(years=1)].copy()
    df_previous['item_id'] = df_previous.ID.str[0:6]

    comparison_df = pd.merge(
        df_previous[['item_id', 'price(원/kg)']].rename(columns={'price(원/kg)': 'price_previous'}),
        pred[['item_id', 'mean']].rename(columns={'mean': 'mean_current'}),
        on='item_id',
        how='inner'
    )
    comparison_df['price_increase_percent'] = ((comparison_df['mean_current'] - comparison_df['price_previous']) /
                                               comparison_df['price_previous']) * 100
    comparison_df[['item_id', 'price_previous', 'mean_current', 'price_increase_percent']]
    comparison_df = comparison_df[np.isfinite(comparison_df['price_increase_percent'])]  # 값 폭주하는 거 제거
    comparison_df = comparison_df[comparison_df['price_increase_percent'] > 0]  # 양수만 남김
    item_mapping = {
        'TG': '감귤',
        'BC': '브로콜리',
        'RD': '무',
        'CR': '당근',
        'CB': '양배추'
    }

    # item_id의 첫 두 글자를 통해 품목 이름을 매핑하여 새로운 열 추가
    comparison_df['item_name'] = comparison_df['item_id'].str[:2].map(item_mapping)

    # 열 이름 변경
    comparison_df.rename(columns={'mean': 'mean_current'}, inplace=True)

    # 결과 반환
    return comparison_df[['item_id', 'item_name', 'price_previous', 'mean_current', 'price_increase_percent']]


def main_ai(target_date='2021-01-30'):
    # 데이터 불러오는 걸 너무 대충하긴 했는데...
    train_df = pd.read_csv("/home/yoon/Desktop/Others/open/train.csv")
    train_df = train_df[['ID', "timestamp", "supply(kg)", "price(원/kg)"]]

    weather_2019 = pd.read_csv("weather_average2019.csv")
    weather_2019.rename(columns={'TM': 'timestamp'}, inplace=True)
    weather_2020 = pd.read_csv("weather_average2020.csv")
    weather_2020.rename(columns={'TM': 'timestamp'}, inplace=True)

    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    weather_2019['timestamp'] = pd.to_datetime(weather_2019['timestamp'])
    weather_2020['timestamp'] = pd.to_datetime(weather_2020['timestamp'])
    weather = pd.concat([weather_2019, weather_2020])

    df_concat = pd.merge(train_df, weather, on='timestamp', how='left')
    df_concat.dropna(inplace=True)
    df_concat['item_id'] = df_concat['ID'].str[:6]

    data = TimeSeriesDataFrame(df_concat.drop(columns=['ID']))
    predictor = TimeSeriesPredictor(
        prediction_length=30,
        target="price(원/kg)",
        eval_metric="RMSE",
    )
    # seed 고정
    predictor.fit(data, random_seed=42, )
    predictor.refit_full()
    pred = predictor.predict(data, random_seed=42, )
    pred = pred.reset_index()
    final_pred = pred.sort_values('timestamp').groupby('item_id').tail(1).reset_index(drop=True)
    compare = price_comparison(final_pred, df_concat, date=target_date)
    compare["item_id"] = compare["item_id"].str[:2]
    numeric_columns = compare.select_dtypes(include=[float, int]).columns
    compare_mean = compare.groupby("item_id")[numeric_columns].mean().reset_index()

    item_mapping = {
        'TG': '감귤',
        'BC': '브로콜리',
        'RD': '무',
        'CR': '당근',
        'CB': '양배추'
    }

    # item_id의 첫 두 글자를 통해 품목 이름을 매핑하여 새로운 열 추가
    compare_mean['item_name'] = compare_mean['item_id'].str[:2].map(item_mapping)

    # 열 이름 변경
    compare_mean.rename(columns={'mean': 'mean_current'}, inplace=True)

    return compare_mean



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ta
import mplfinance as mpf
import gradio as gr
from tqdm import tqdm

def load_data(company_name: str):
    ticker = company_name.lower()
    path = f'https://stooq.com/q/d/l/?s={ticker}&i=d'
    df = pd.read_csv(path)
    
    if df.shape[0] > 10:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        if any(df.loc[df.isnull().any(axis=1)]) == True:
            df.Volume.interpolate(method='linear', inplace=True)
            df = df.dropna(axis=0)
    else:
        for ext in ['.us', '.uk', '.de', '.hu']:
            ticker_ext = company_name.lower() + ext
            path_ext = f'https://stooq.com/q/d/l/?s={ticker_ext}&i=d'
            df_ext = pd.read_csv(path_ext)
            if df_ext.shape[0] > 10:
                df = df_ext
                df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
                if any(df.loc[df.isnull().any(axis=1)]) == True:
                    df.Volume.interpolate(method='linear', inplace=True)
                    df = df.dropna(axis=0)
                break
    
    return df

def plot_candle(df, company_name: str, interval: int, figsize=(10,5)):
    df_plot = df.copy()
    df_plot = df_plot[df_plot['Company'] == company_name.upper()]
    df_plot.set_index('Date', inplace=True)
    mpf.plot(
        df_plot[-interval:],
        type='candle',
        style='charles',
        title=f'Wykres {company_name.upper()}',
        ylabel='Price ($)',
        volume=True,
        figsize=figsize
            )


def generate_target(df, days_to_forecast=1, first_test_day='2023-01-01', last_test_day='2023-07-31'):
    y_df = pd.DataFrame(columns=['Date', 'Company'] + [f'y_{i+1}' for i in range(days_to_forecast)])
    
    for name in tqdm(df['Company'].unique(), desc="Generating Targets"):
        company = df[df['Company'] == name].copy()

        y_columns = []
        for i in range(days_to_forecast):
            company[f'y_{i+1}'] = company['Close'].shift(-i - days_to_forecast)
            y_columns.append(f'y_{i+1}')

        y_df = pd.concat([y_df, company[['Date', 'Company'] + y_columns]], ignore_index=True)

    y_df.sort_values(by=['Date', 'Company'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    y_df.reset_index(drop=True, inplace=True)
    X = df
    
    date_features = pd.DataFrame()

    for name in tqdm(X['Company'].unique(), desc="Generating Date Features"):
        company = X[X['Company'] == name]
        df = generate_features(company)
        date_features = date_features.append(df, ignore_index=True)

    date_features.sort_values(by=['Date', 'Company'], inplace=True)
    date_features.reset_index(drop=True, inplace=True)
    nan_indices = date_features[date_features.isna().any(axis=1)].index
    X_cleaned = date_features.drop(nan_indices)
    y_cleaned = y_df.drop(nan_indices)
    X_cleaned.reset_index(drop=True, inplace=True)
    y_cleaned.reset_index(drop=True, inplace=True)
    
    train_condition = X_cleaned['Date'] < first_test_day
    X_train, y_train = X_cleaned[train_condition], y_cleaned[train_condition]
    
    test_condition = (X_cleaned['Date'] >= first_test_day) & (X_cleaned['Date'] <= last_test_day)
    X_test, y_test = X_cleaned[test_condition], y_cleaned[test_condition]
    
    y_train, y_test = y_train.iloc[:,2:].values, y_test.iloc[:,2:].values
    return X_train, y_train, X_test, y_test

def calculate_mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def generate_features(date):
    df = pd.DataFrame()
    # pobrane cechy
    df['Date'] = date['Date']
    df['Company'] = date['Company']
    df['Open'] = date['Open']
    df['Close'] = date['Close']
    df['High'] = date['High']
    df['Low'] = date['Low']
    df['Volume'] = date['Volume']

    df['Close_diff'] = date['Close'] - date['Close'].shift(1)
    df['Close_diff_perc'] = (date['Close'] - date['Close'].shift(1)) / date['Close'].shift(1) * 100
    df['Open_diff'] = date['Open'] - date['Open'].shift(1)
    df['Open_diff_perc'] = (date['Open'] - date['Open'].shift(1)) / date['Open'].shift(1) * 100
    # Średnie wolumeny
    df['Volume_1w'] = df['Volume'].rolling(5).mean()
    df['Volume_2w'] = df['Volume'].rolling(10).mean()
    df['Volume_1m'] = df['Volume'].rolling(21).mean()
#     df['Ratio_volume_1w_2w'] = df['Volume_1w'] / df['Volume_2w']
    df['Ratio_volume_1w_1m'] = df['Volume_1w'] / df['Volume_1m']
    # Odchylenia standardowe
    df['Std_price_1w'] = df['Close'].rolling(5).std()
    df['Std_price_1m'] = df['Close'].rolling(21).std()
    df['Ratio_std_price_1w_1m'] = df['Std_price_1w'] / df['Std_price_1m']
#     df['Std_volume_1w'] = df['Volume'].rolling(5).std()
#     df['Std_volume_1m'] = df['Volume'].rolling(21).std()
#     df['Ratio_std_volume_1w_1m'] = df['Std_volume_1w'] / df['Std_volume_1m']
    # Zwroty
    df['Return_1d'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
    df['Return_1w'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)).shift(1)
    df['Return_1m'] = ((df['Close'] - df['Close'].shift(21)) / df['Close'].shift(21)).shift(1)
#     df['Moving_avg_1w'] = df['Return_1d'].rolling(5).mean().shift(1)
    df['Moving_avg_1m'] = df['Return_1d'].rolling(21).mean().shift(1)
    # Wskaźniki
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_30'] = df['Close'].rolling(21).mean()
    df['EMA_10'] = ta.trend.EMAIndicator(df['Close'], 10).ema_indicator()
    df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], 20).ema_indicator()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['MACD_diff'] = ta.trend.MACD(df['Close']).macd_diff()
    df['MACD_signal'] = ta.trend.MACD(df['Close']).macd_signal()
    df['BOLL_high']= ta.volatility.BollingerBands(df['Close']).bollinger_hband()
    df['BOLL_low']= ta.volatility.BollingerBands(df['Close']).bollinger_lband()
#     df['BOLL_perc']= ta.volatility.BollingerBands(df['Close']).bollinger_pband()
    df['ROC_10'] = ta.momentum.ROCIndicator(df['Close'], 10).roc()
    df['ROC_20'] = ta.momentum.ROCIndicator(df['Close'], 20).roc()
#     df['MI'] = ta.trend.MassIndex(df['High'], df['Low']).mass_index()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    df['NVI'] = ta.volume.NegativeVolumeIndexIndicator(df['Close'], df['Volume']).negative_volume_index()
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
#     df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume']).chaikin_money_flow()
#     df['PVO'] = ta.momentum.pvo(df['Volume'])
#     df['PVO_hist'] = ta.momentum.pvo_hist(df['Volume'])
#     df['PVO_signal'] = ta.momentum.pvo_signal(df['Volume'])
#     df['UO'] = ta.momentum.UltimateOscillator(df['High'], df['Low'], df['Close']).ultimate_oscillator()
#     df['DPO'] = ta.trend.DPOIndicator(df['Close']).dpo() 
    return df

def predictions(company_name: str, model):
    company_name = company_name
    df = load_data(company_name)
    df['Company'] = company_name.upper()
    df = generate_features(df)
    df = df.iloc[-30:]
    y_pred = model.predict(df.iloc[:,2:])
    
    y_true = df['Close']
    y_pred = y_pred[-1]
    
    start_date = pd.to_datetime(df['Date'].iloc[0])
    dates_pred = pd.date_range(start=start_date, periods=(len(y_true) + len(y_pred)), freq='B')

    y_true = np.concatenate([y_true.values, [np.nan] * len(y_pred)])

    plt.figure(figsize=(10, 6))
    plt.plot(dates_pred, y_true, label='Rzeczywiste wartości', marker='o')
    plt.plot(dates_pred[-len(y_pred):], y_pred, label='Przewidywane wartości', marker='o', linestyle='--')
    plt.xlabel('Data')
    plt.ylabel('Wartości')
    plt.title('Porównanie rzeczywistych i przewidywanych wartości')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

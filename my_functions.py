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
            company[f'y_{i+1}'] = company['Close_diff'].shift(-i - days_to_forecast)
            y_columns.append(f'y_{i+1}')

        y_df = pd.concat([y_df, company[['Date', 'Company'] + y_columns]], ignore_index=True)

    y_df.sort_values(by=['Date', 'Company'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    y_df.reset_index(drop=True, inplace=True)
    
    price_df = pd.DataFrame(columns=['Date', 'Company'] + [f'y_{i+1}' for i in range(days_to_forecast)])
    
    for name in tqdm(df['Company'].unique(), desc="Generating Targets"):
        company = df[df['Company'] == name].copy()

        price_columns = []
        for i in range(days_to_forecast):
            company[f'y_{i+1}'] = company['Close'].shift(-i - days_to_forecast)
            price_columns.append(f'y_{i+1}')

        price_df = pd.concat([price_df, company[['Date', 'Company'] + price_columns]], ignore_index=True)

    price_df.sort_values(by=['Date', 'Company'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    price_df.reset_index(drop=True, inplace=True)
    
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
    y_price_cleaned = price_df.drop(nan_indices)
    X_cleaned.reset_index(drop=True, inplace=True)
    y_cleaned.reset_index(drop=True, inplace=True)
    y_price_cleaned.reset_index(drop=True, inplace=True)
    
    train_condition = X_cleaned['Date'] < first_test_day
    X_train, y_train, y_train_price = X_cleaned[train_condition], y_cleaned[train_condition], y_price_cleaned[train_condition]
    
    test_condition = (X_cleaned['Date'] >= first_test_day) & (X_cleaned['Date'] <= last_test_day)
    X_test, y_test, y_test_price = X_cleaned[test_condition], y_cleaned[test_condition], y_price_cleaned[test_condition]
    
    y_train, y_test = y_train.iloc[:,2:].values, y_test.iloc[:,2:].values
    y_train_price, y_test_price = y_train_price.iloc[:,2:].values, y_test_price.iloc[:,2:].values
    return X_train, y_train, X_test, y_test, y_train_price, y_test_price

def calculate_mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def generate_features(date):
    df = pd.DataFrame()
    # pobrane cechy
    df['Date'] = date['Date']
    df['Company'] = date['Company']
    df['Close_pred'] = date['Close_pred']
    df['Open'] = date['Open']
    df['Close'] = date['Close']
    df['High'] = date['High']
    df['Low'] = date['Low']
    df['Volume'] = date['Volume']
    df['Close_diff'] = date['Close_diff']
    #Utworzone cechy
    df['Close_diff_perc'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
    df['Open_diff'] = df['Open'] - df['Open'].shift(1)
    df['Close_Open'] = df['Close'] - df['Open']
    df['Ratio_volume_1d_yd'] = df['Volume'] / df['Volume'].shift(1)
    df['Ratio_volume_1d_1w'] = df['Volume'] / df['Volume'].rolling(5).mean()
    df['Ratio_close_diff_1d_1w'] = df['Close_diff'] / df['Close_diff'].rolling(5).std()
    df['Ratio_close_diff_1d_2w'] = df['Close_diff'] / df['Close_diff'].rolling(10).std()
    df['Ratio_close_diff_1w_2w'] = df['Close_diff'].rolling(5).std() / df['Close_diff'].rolling(10).std()
    # Zwroty
    df['Return_1d'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
    df['Return_1w'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)).shift(1)
    # Wskaźniki
    df['Ratio_close_SMA_5'] = df['Close'] / df['Close'].rolling(5).mean()
    df['Ratio_close_SMA_10'] = df['Close'] / df['Close'].rolling(10).mean()
    df['Close_SMA_5_diff'] = df['Close'] - df['Close'].rolling(5).mean()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['MACD_diff'] = ta.trend.MACD(df['Close']).macd_diff()
    df['MACD_signal'] = ta.trend.MACD(df['Close']).macd_signal()
    df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
    df['TSI'] = ta.momentum.TSIIndicator(df['Close']).tsi()
    df['ROC_10'] = ta.momentum.ROCIndicator(df['Close'], 10).roc()
    df['MI'] = ta.trend.MassIndex(df['High'], df['Low']).mass_index()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
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

def edit_predictions(company_name: str, model, X_test):
    """
    Funkcja służy do wygenerowania wykresu prognoz ceny zamknięcia na najbliższe 5 dni
    na podstawie prognoz zmiany ceny.
    
    company_name - ticker spółki w formacie string
    model - nazwa modelu, który chcemy użyć
    """
    company_name = company_name
    pred_df = load_data(company_name)
    pred_df['Company'] = company_name.upper()
    pred_df['Close_pred'] = pred_df['Close'].shift(-1)
    pred_df['Close_diff'] = pred_df['Close'] - pred_df['Close'].shift(1)
    pred_df = generate_features(pred_df)
    pred_df = pred_df.iloc[-30:]
    y_pred = model.predict(pred_df.iloc[:,8:])
    
    y_true = pred_df['Close']
    X_test.reset_index(drop=True, inplace=True)
    initial_prices = pred_df['Close'].values[-30:]
    new_prices = np.zeros((len(initial_prices), 5))
    
    for i in range(5):
        new_prices[:, i] = np.cumsum(y_pred[:, i]) + initial_prices
    
    y_pred = new_prices[-1]
    
    start_date = pd.to_datetime(pred_df['Date'].iloc[0])
    dates_pred = pd.date_range(start=start_date, periods=(len(y_true) + len(y_pred)), freq='B')

    y_true = np.concatenate([y_true.values, [np.nan] * len(y_pred)])

    plt.figure(figsize=(10, 6))
    plt.plot(dates_pred, y_true, label='Rzeczywiste wartości', marker='o')
    plt.plot(dates_pred[-len(y_pred):], y_pred, label='Przewidywane wartości', marker='o', linestyle='--')
    plt.xlabel('Data')
    plt.ylabel('Wartości')
    plt.title(f'Prognoza na najbliższe 5 dni dla {company_name.upper()}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()    

def calculate_MAPE_diff_price(X_test, y_test_price, model):
    y_pred = model.predict(X_test.iloc[:,8:])
    X_test.reset_index(drop=True, inplace=True)
    initial_prices = X_test['Close'].values
    n_days = 5
    new_prices = np.zeros((len(initial_prices), n_days))
    for i in range(n_days):
        new_prices[:, i] = np.cumsum(y_pred[:, i]) + initial_prices
    mape_1 = calculate_mape(new_prices[:,0], X_test['Close_pred'])
    mape_5 = calculate_mape(new_prices, y_test_price)
    print(f'MAPE 1 dzień: {mape_1:.2f}%')
    print(f'MAPE 5 dni: {mape_5:.2f}%')

# Importing necessary libraries
from imports import *
import candlesticks as cs
from time import sleep

w10 = 10
w20 = 20
w30 = 30
w40 = 40
w50 = 50
w100 = 100
w200 = 200

# Fetching historical data
def get_stock_data(ticker, start_date, end_date, TF = '1d'):
    df = yf.download(ticker, start=start_date, end=end_date, interval = TF, auto_adjust=False)
    # yfinance started giving multi-index df. Get rid of ticker col.
    df = df.reset_index() 
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

def get_fed_rates(start_date, end_date):
    rates = web.DataReader('FEDFUNDS', 'fred', start_date, end_date)
    return rates

# Adding technical indicators
def add_technical_indicators (df):
    # Optimization: Compute rolling statistics in one go where possible to avoid repeated calls
    close_prices = df.Close

    df['SMA1'], df['SMA2'], df['SMA3'] = calSMAs(close_prices)
    df['EMA1'], df['EMA2'], df['EMA3'] = calEMAs(close_prices)
    df['RSI']= calculate_rsi(df)
    df['RSI2']  = df['RSI'].rolling(window=20).mean()
    df['OBV'] = calculate_obv(df)
    df['PVT'] = calculate_pvt(df)
    df['MFI'] = calculate_mfi(df)
    df['CCI'] = calculate_cci(df)
    df[['+DI', '-DI', 'ADX']] = calculate_dmi(df, n=14)
    
    df = calculate_stochrsi(df)
    df = calcBollingerBands(df)
    df['ATR'] = calculate_atr(df.High, df.Low, df.Close)
    
    df['Mom1'] = close_prices - close_prices.shift(9)
    df['Mom2'] = close_prices - close_prices.shift(20)
    
    df['ROC1'] = close_prices.pct_change(periods=9) * 100
    df['ROC2'] = close_prices.pct_change(periods=20) * 100
    
    df['buy_volume'] = (df.Close > df.Close.shift(1)) * df['Volume']
    df['sell_volume'] = (df.Close < df.Close.shift(1)) * df['Volume']

    df['sumBuyVol'] = df['buy_volume'].rolling(window=20).sum()
    df['sumSellVol'] = df['sell_volume'].rolling(window=20).sum()

    # Drop rows with NaN values resulting from rolling calculations
    df.dropna(inplace=True)

    return df

def calSMAs (close):
    sma1 = close.rolling(window=20).mean()
    sma2 = close.rolling(window=50).mean()
    sma3 = close.rolling(window=100).mean()
    return sma1, sma2, sma3

def calEMAs (close):
    ema1 = close.ewm(span=20, adjust=False).mean()
    ema2 = close.ewm(span=50, adjust=False).mean()
    ema3 = close.ewm(span=100, adjust=False).mean()
    return ema1, ema2, ema3

def calculate_vwma(df, window=20):
    vwma = (df['Close'] * df['Volume']).rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()
    return vwma.set_axis(df.index)  # Preserve original index

def calculate_keltner(df, ema_window=20, atr_window=10, multiplier=2):
    middle = df['Close'].ewm(span=ema_window).mean()
    atr = calculate_atr(df.High, df.Low, df.Close)
    upper = middle + multiplier * atr
    lower = middle - multiplier * atr
    return pd.DataFrame({
        'KCm': middle,
        'KCu': upper,
        'KCl': lower
    }, index=df.index)  # Explicit index

def calculate_vortex(df, window=14):
    vm_plus = abs(df['High'] - df['Low'].shift(1))
    vm_minus = abs(df['Low'] - df['High'].shift(1))
    atr = calculate_atr(df.High, df.Low, df.Close)
    vi_plus = vm_plus.rolling(window).sum() / atr.rolling(window).sum()
    vi_minus = vm_minus.rolling(window).sum() / atr.rolling(window).sum()
    return pd.DataFrame({
        'VI+': vi_plus,
        'VI-': vi_minus
    }, index=df.index)  # Explicit index

def calculate_ichimoku(df):
    high, low, close = df['High'], df['Low'], df['Close']
    df['Tenkan'] = (high.rolling(9).max() + low.rolling(9).min()) / 2
    df['Kijun'] = (high.rolling(26).max() + low.rolling(26).min()) / 2
    df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['Senkou_B'] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return df[['Tenkan', 'Kijun', 'Senkou_A', 'Senkou_B']]

def calculate_supertrend(df, multiplier=3, window=10):
    atr = calculate_atr(df.High, df.Low, df.Close)
    df['Upper'] = (df['High'] + df['Low']) / 2 + multiplier * atr
    df['Lower'] = (df['High'] + df['Low']) / 2 - multiplier * atr
    return df[['Upper', 'Lower']]

def calcBollingerBands (df):
    # Bollinger Bands
    close = df.Close
    rolling_20 = close.rolling(window=w20)
    df['BBm'] = rolling_20.mean()
    rolling_std = rolling_20.std()
    df['BBu'] = df['BBm'] + 2 * rolling_std
    df['BBl'] = df['BBm'] - 2 * rolling_std
    return df
    
def calculate_rsi(df, w=14):
    close_prices = df['Close']
    # Calculate price changes (delta)
    delta = close_prices.diff()

    # Separate positive gains (where the price went up) and negative losses (where the price went down)
    gain = delta.clip(lower=0)  # gains (positive deltas)
    loss = -delta.clip(upper=0) # losses (negative deltas)

    # Calculate the rolling mean of gains and losses
    avg_gain = gain.rolling(window=w, min_periods=1).mean()
    avg_loss = loss.rolling(window=w, min_periods=1).mean()

    # Calculate the relative strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calculate_stochrsi(df, rsi_period=14, stoch_period=20, d_period=9):
    # Calculate lowest low and highest high for RSI over the stoch_period
    lowest_low = df['RSI'].rolling(window=stoch_period).min()
    highest_high = df['RSI'].rolling(window=stoch_period).max()
    
    # Calculate Stochastic RSI (%K)
    df['StochRSI'] = (df['RSI'] - lowest_low) / (highest_high - lowest_low) * 100
    
    # Calculate %D as the SMA of %K
    df['StochRSI_D'] = df['StochRSI'].rolling(window=d_period).mean()
    
    return df

def calculate_atr(high, low, close):
    # ATR (Optimized true range calculation)
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()

    # Concatenate the Series into a DataFrame
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    
    # Calculate the ATR with a rolling mean
    atr = tr.rolling(window=14).mean()
    return atr

def calculate_obv(data):
    obv = [0]  # Initialize OBV with 0
    for i in range(1, len(data)):
        # Check for NaN explicitly for each value
        if pd.isna(data['Close'].iloc[i]) or pd.isna(data['Close'].iloc[i-1]) or pd.isna(data['Volume'].iloc[i]):
            obv.append(obv[-1])  # Append the previous OBV value if there is a NaN
        elif data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return obv

def calculate_pvt(df):
    tmp = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)) * df['Volume']
    return tmp.cumsum()

def calculate_mfi(data, period=20):
    required_columns = ['High', 'Low', 'Close', 'Volume']
    if not all(column in data.columns for column in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    data['TP'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['RMF'] = data['TP'] * data['Volume']
    data['TP_diff'] = data['TP'].diff()

    data['Positive_MF'] = np.where(data['TP_diff'] > 0, data['RMF'], 0)
    data['Negative_MF'] = np.where(data['TP_diff'] < 0, data['RMF'], 0)

    # Step 4: Calculate the rolling sums of Positive and Negative Money Flow
    data['Positive_MF_sum'] = data['Positive_MF'].rolling(window=period).sum()
    data['Negative_MF_sum'] = data['Negative_MF'].rolling(window=period).sum()

    data['MFR'] = data['Positive_MF_sum'] / data['Negative_MF_sum']
    data['MFI'] = 100 - (100 / (1 + data['MFR']))

    data['MFI'] = np.where(data['Negative_MF_sum'] == 0, 100, data['MFI'])

    # Drop unnecessary columns before returning MFI
    data.drop(columns=['TP', 'RMF', 'TP_diff', 'Positive_MF', 
                       'Negative_MF', 'Positive_MF_sum', 'Negative_MF_sum', 
                       'MFR'], inplace=True)

    return data['MFI']

def calculate_cci(data, period=20):
    if not all(col in data.columns for col in ['High', 'Low', 'Close']):
        raise ValueError("DataFrame must contain 'High', 'Low', and 'Close' columns.")

    data['Typical Price'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['SMA'] = data['Typical Price'].rolling(window=period).mean()
    data['Mean Deviation'] = data['Typical Price'].rolling(window=period).apply(lambda x: (abs(x - x.mean())).mean(), raw=True)
    data['CCI'] = (data['Typical Price'] - data['SMA']) / (0.015 * data['Mean Deviation'])
    
    # Drop unnecessary columns before returning 
    data.drop(columns=['Typical Price', 'SMA', 'Mean Deviation'], inplace=True)
    
    return data['CCI']

def calculate_dmi(df, n=14):
    # Calculate True Range (TR)
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = np.abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)

    # Calculate Directional Movements
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > 
                               (df['Low'].shift(1) - df['Low']), 
                               df['High'] - df['High'].shift(1), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > 
                               (df['High'] - df['High'].shift(1)), 
                               df['Low'].shift(1) - df['Low'], 0)

    # Smooth the True Range, +DM, and -DM with an exponential moving average (EMA)
    df['TR_smooth'] = df['TR'].rolling(window=n).mean()
    df['+DM_smooth'] = df['+DM'].rolling(window=n).mean()
    df['-DM_smooth'] = df['-DM'].rolling(window=n).mean()

    # Calculate +DI and -DI
    df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])

    # Calculate DX (Directional Index)
    df['DX'] = 100 * (np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))

    # Calculate ADX (Average Directional Index)
    df['ADX'] = df['DX'].rolling(window=n).mean()

    # Return relevant columns
    return df[['+DI', '-DI', 'ADX']]
    
# Preparing the data for Machine Learning
def prepare_ml_data(df):
    # Include candlestick pattern features and new indicators
    features = ['SMA1', 'SMA2', 'SMA3', 'EMA1', 'EMA2', 'EMA3', 'RSI', 'RSI2', 
                'BBm', 'BBu', 'BBl', 'Mom1', 'Mom2', 'ROC1', 'ROC2', 
                'Candlesticks', 'Volume', 'ATR', 'MFI', 'CCI', '+DI', '-DI', 'ADX',
                'StochRSI', 'StochRSI_D', 'sumBuyVol', 'sumSellVol']
    
    df = df.dropna()
    X = df[features]
    y = df['Close']  # Target variable
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

# Building the Random Forest Model
def train_model(X_train, y_train, nest=1000, md=6):
    model = RandomForestRegressor(n_estimators=nest, random_state=42,
                                  max_depth=md)
    model.fit(X_train, y_train)
    return model

def train_booster(X_train, y_train, nest=1000, lr=0.001, md=6, ss=0.8,
                             ra=0.1, rl=1):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=nest, 
                             learning_rate=lr, max_depth=md, subsample=ss,
                             reg_alpha=ra, reg_lambda=rl)
    model.fit(X_train, y_train)
    return model

# Evaluate the model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
    r2 = round(r2_score(y_test, y_pred), 3)
    
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R^2 Score: {r2}")

def generate_signal(predicted_prices, current_price, df):
    # Get the last available values of the technical indicators from the dataframe
    last_row = df.iloc[-1]
    
    # Technical indicator thresholds
    SMA2_threshold = 0.02 
    SMA3_threshold = 0.02
    EMA1_threshold = 0.02
    EMA2_threshold = 0.02
    rsi_threshold_buy = 35
    rsi_threshold_sell = 65
    bb_threshold = 0.02

    # Extract the latest values of the technical indicators
    SMA2 = last_row['SMA2']
    SMA3 = last_row['SMA3']
    EMA1 = last_row['EMA1']
    EMA2 = last_row['EMA2']
    rsi = last_row['RSI']
    BBl = last_row['BBl']
    BBu = last_row['BBu']

    # Initialize score variables
    buy_score = 0
    sell_score = 0

    # Combine conditions by adding scores for buy and sell signals
    # Check if current price is above SMA and EMA thresholds
    if current_price > (1 + SMA2_threshold) * SMA2:
        buy_score += 1
    if current_price < (1 - SMA2_threshold) * SMA2:
        sell_score += 1

    if current_price > (1 + SMA3_threshold) * SMA3:
        buy_score += 1
    if current_price < (1 - SMA3_threshold) * SMA3:
        sell_score += 1

    if current_price > (1 + EMA1_threshold) * EMA1:
        buy_score += 1
    if current_price < (1 - EMA1_threshold) * EMA1:
        sell_score += 1

    if current_price > (1 + EMA2_threshold) * EMA2:
        buy_score += 1
    if current_price < (1 - EMA2_threshold) * EMA2:
        sell_score += 1

    # Check RSI for buy/sell signals
    if rsi < rsi_threshold_buy:
        buy_score += 1
    if rsi > rsi_threshold_sell:
        sell_score += 1

    # Check if current price is near the Bollinger Bands
    if current_price < (1 - bb_threshold) * BBl:
        buy_score += 1
    if current_price > (1 + bb_threshold) * BBu:
        sell_score += 1

    # Generate final signal based on scores
    if buy_score > sell_score:
        return "BUY"
    elif sell_score > buy_score:
        return "SELL"
    else:
        return "HODL / SIDELINES"
    
##### PREDICT PRICES #####
def predict_prices(model, data, scaler, num_days=5, window_size=300):
    # Use the same features during prediction
    features = ['SMA1', 'SMA2', 'SMA3', 'EMA1', 'EMA2', 'EMA3', 'RSI', 'RSI2',
                'BBm', 'BBu', 'BBl', 'Mom1', 'Mom2', 'ROC1', 'ROC2', 
                'Candlesticks', 'Volume', 'ATR', 'MFI', 'CCI', '+DI', '-DI', 'ADX',
                'StochRSI', 'StochRSI_D', 'sumBuyVol', 'sumSellVol']
    
    last_data = data.copy()  # Copy the whole dataframe to modify
    
    predicted_prices = []  # List to store predicted values
    wt = 0.25

    for i in range(num_days):
        # Use a rolling window of size 'window_size' from actual data (historical data)
        xdf = last_data.iloc[-window_size:].copy()
        
        # Ensure all technical indicators are calculated before prediction
        xdf['SMA1'], xdf['SMA2'], xdf['SMA3'] = calSMAs(xdf['Close'])
        xdf['EMA1'], xdf['EMA2'], xdf['EMA3'] = calEMAs(xdf['Close'])
        xdf['RSI'] = calculate_rsi(xdf)
        xdf['RSI2']  = xdf['RSI'].rolling(window=14).mean()
        xdf['MFI'] = calculate_mfi(xdf)
        xdf['CCI'] = calculate_cci(xdf)
        xdf[['+DI', '-DI', 'ADX']] = calculate_dmi(xdf, n=14)

        xdf = calcBollingerBands(xdf)
        xdf = calculate_stochrsi(xdf)
        
        # Momentum and ROC
        xdf['Mom1'] = xdf['Close'] - xdf['Close'].shift(9)
        xdf['Mom2'] = xdf['Close'] - xdf['Close'].shift(20)
        
        xdf['ROC1'] = xdf['Close'].pct_change(periods=9) * 100
        xdf['ROC2'] = xdf['Close'].pct_change(periods=20) * 100  # Change to 30
        
        # Extract features for the current prediction
        inData = xdf[features].iloc[-1:]
        
        # Scale features
        inData_scaled = scaler.transform(inData)
        
        # Predict the price for the next day using the model
        predicted_price = model.predict(inData_scaled)
        rounded_price = round(predicted_price[0], 4)
        
        # Blend predicted price with the last actual price
        last_actual = last_data['Close'].iloc[-1]
        wtPrice = wt * rounded_price + (1 - wt) * last_actual
        predicted_prices.append(wtPrice)  # Store the scalar value
        
        # Update the 'Close' price with the predicted value for the next business day
        next_index = pd.bdate_range(last_data.index[-1], periods=2)[-1]  # Next business day
        last_data.loc[next_index] = np.nan  # Add the new row
        last_data.at[next_index, 'Close'] = wtPrice  # Only update 'Close' with predicted value
        
        # Append a new row with the predicted 'Close' value only
        new_row = pd.DataFrame({
            'Close': [wtPrice],
            'Date': [next_index]
        }).set_index('Date')
        
        # Append the new row to the dataframe
        last_data = pd.concat([last_data, new_row])
    
    return predicted_prices
    
def add_candlestickpatterns(df):
    # Ensure df is a copy, not a view, to avoid the SettingWithCopyWarning
    df = df.copy()
    # Detect candlestick patterns and add to dataframe
    df['Bullish_Engulfing'] = cs.detect_bullish_engulfing(df)
    df['Doji'] = cs.detect_doji(df)
    df['Hammer'] = cs.detect_hammer(df)
    df['Hanging_Man'] = cs.detect_hanging_man(df)
    df['Morning_Star'] = cs.detect_morning_star(df)
    df['Evening_Star'] = cs.detect_evening_star(df)
    df['Shooting_Star'] = cs.detect_shooting_star(df)
    df['Three_White_Soldiers'] = cs.detect_three_white_soldiers(df)
    df['Three_Black_Crows'] = cs.detect_three_black_crows(df)

    # Combine all patterns into one column
    df['Candlesticks'] = (df['Doji'] +
                          df['Hammer'] * 2 + 
                          df['Hanging_Man'] * 3 + 
                          df['Morning_Star'] * 4 + 
                          df['Evening_Star'] * 5 +
                          df['Shooting_Star'] * 6 +
                          df['Three_White_Soldiers'] * 7 + 
                          df['Three_Black_Crows'] * 8 + 
                          df['Bullish_Engulfing'] * 9)

    return df
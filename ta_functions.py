# Importing necessary libraries
from imports import *
import candlesticks as cs

w10 = 10
w20 = 20
w30 = 30
w40 = 40
w50 = 50
w100 = 100
w200 = 200

# Fetching historical data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def get_fed_rates(start_date, end_date):
    rates = web.DataReader('FEDFUNDS', 'fred', start_date, end_date)
    return rates

# Adding technical indicators
def add_technical_indicators (df):
    # Optimization: Compute rolling statistics in one go where possible to avoid repeated calls
    close_prices = df.Close

    df['SMA1'], df['SMA2'], df['SMA3'] = calSMAs(close_prices)
    df['EMA1'], df['EMA2'], df['EMA3'] = calEMAs(close_prices)
    df['RSI'] = calculate_rsi(df)
    df['OBV'] = calculate_obv(df)
    df['MFI'] = calculate_mfi(df)
    df['CCI'] = calculate_cci(df)
    
    df = calculate_stochrsi(df)
    df = calcBollingerBands(df)
    
    df['ATR'] = calculate_atr(df.High, df.Low, df.Close)
    
    df['Mom1'] = close_prices - close_prices.shift(20)
    df['Mom2'] = close_prices - close_prices.shift(50)
    
    df['ROC1'] = close_prices.pct_change(periods=20) * 100
    df['ROC2'] = close_prices.pct_change(periods=50) * 100

    # Drop rows with NaN values resulting from rolling calculations
    df.dropna(inplace=True)

    return df

def calSMAs (close):
    sma1 = close.rolling(window=10).mean()
    sma2 = close.rolling(window=50).mean()
    sma3 = close.rolling(window=200).mean()
    return sma1, sma2, sma3

def calEMAs (close):
    ema1 = close.ewm(span=10, adjust=False).mean()
    ema2 = close.ewm(span=50, adjust=False).mean()
    ema3 = close.ewm(span=200, adjust=False).mean()
    return ema1, ema2, ema3

def calcBollingerBands (df):
    # Bollinger Bands
    close = df.Close
    rolling_20 = close.rolling(window=w20)
    df['BBm'] = rolling_20.mean()
    rolling_std = rolling_20.std()
    df['BBu'] = df['BBm'] + 2 * rolling_std
    df['BBl'] = df['BBm'] - 2 * rolling_std
    return df
    
def calculate_rsi(df, window=14):
    close_prices = df['Close']
    # Calculate price changes (delta)
    delta = close_prices.diff()

    # Separate positive gains (where the price went up) and negative losses (where the price went down)
    gain = delta.clip(lower=0)  # gains (positive deltas)
    loss = -delta.clip(upper=0) # losses (negative deltas)

    # Calculate the rolling mean of gains and losses
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

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
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])  # No change if close prices are equal

    return obv

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

# Preparing the data for Machine Learning
def prepare_ml_data(df):
    # Include candlestick pattern features and new indicators
    features = ['SMA1', 'SMA2', 'SMA3', 'EMA1', 'EMA2', 'EMA3', 'RSI', 
                'BBm', 'BBu', 'BBl', 'Mom1', 'Mom2', 'ROC1', 'ROC2', 
                'Candlesticks', 'Volume', 'ATR', 'MFI', 'CCI', 
                'StochRSI', 'StochRSI_D']
    
    df = df.dropna()
    X = df[features]
    y = df['Close']  # Target variable
    
    scaler = MinMaxScaler() #StandardScaler()
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

    # Initialize signals
    buy_signal = False
    sell_signal = False
    
    # Check if current price is above the SMA and EMA thresholds
    if current_price > (1 + SMA2_threshold) * SMA2:
        buy_signal = True
    if current_price > (1 + SMA3_threshold) * SMA3:
        buy_signal = True
    if current_price > (1 + EMA1_threshold) * EMA1:
        buy_signal = True
    if current_price > (1 + EMA2_threshold) * EMA2:
        buy_signal = True

    # Check RSI for buy/sell signals
    if rsi < rsi_threshold_buy:
        buy_signal = True
    if rsi > rsi_threshold_sell:
        sell_signal = True

    # Check if current price is below the Bollinger Bands Lower Band
    if current_price < (1 - bb_threshold) * BBl:
        buy_signal = True
    if current_price > (1 + bb_threshold) * BBu:
        sell_signal = True

    # Generate final signal
    if buy_signal and not sell_signal:
        return "BUY"
    elif sell_signal and not buy_signal:
        return "SELL"
    else:
        return "HODL / SIDELINES"
    
##### PREDICT PRICES #####
def predict_prices(model, recent_data, scaler, num_days=5, window_size=300):
    # Use the same features during prediction
    features = ['SMA1', 'SMA2', 'SMA3', 'EMA1', 'EMA2', 'EMA3', 'RSI', 
                'BBm', 'BBu', 'BBl', 'Mom1', 'Mom2', 'ROC1', 'ROC2', 
                'Candlesticks', 'Volume', 'ATR', 'MFI', 'CCI', 
                'StochRSI', 'StochRSI_D']
    
    last_data = recent_data.copy()  # Copy the whole dataframe to modify
    
    predicted_prices = []  # List to store predicted values

    for i in range(num_days):
        # Use a rolling window of size 'window_size' from actual data (historical data)
        sliced = last_data.iloc[-window_size:].copy()
        
        # Ensure all technical indicators are calculated before prediction
        sliced['SMA1'], sliced['SMA2'], sliced['SMA3'] = calSMAs(sliced['Close'])
        sliced['EMA1'], sliced['EMA2'], sliced['EMA3'] = calEMAs(sliced['Close'])
        sliced['RSI'] = calculate_rsi(sliced)
        sliced['MFI'] = calculate_mfi(sliced)
        sliced['CCI'] = calculate_cci(sliced)

        sliced = calcBollingerBands(sliced)
        sliced = calculate_stochrsi(sliced)
        
        # Momentum and ROC
        sliced['Mom1'] = sliced['Close'] - sliced['Close'].shift(20)
        sliced['Mom2'] = sliced['Close'] - sliced['Close'].shift(50)
        
        sliced['ROC1'] = sliced['Close'].pct_change(periods=20) * 100
        sliced['ROC2'] = sliced['Close'].pct_change(periods=50) * 100  # Change to 30

        # Extract features for the current prediction
        inData = sliced[features].iloc[-1:]
        
        # Scale features
        inData_scaled = scaler.transform(inData)
        
        # Predict the price for the next day using the model
        predicted_price = model.predict(inData_scaled)
        rounded_price = round(predicted_price[0], 4)
        predicted_prices.append(rounded_price)  # Store the scalar value
        
        # Update the 'Close' price with the predicted value for the next business day
        next_index = pd.bdate_range(last_data.index[-1], periods=2)[-1]  # Next business day
        last_data.loc[next_index] = np.nan  # Add the new row
        last_data.at[next_index, 'Close'] = rounded_price  # Only update 'Close' with predicted value
        
        # Append a new row with the predicted 'Close' value only
        new_row = pd.DataFrame({
            'Close': [rounded_price],
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
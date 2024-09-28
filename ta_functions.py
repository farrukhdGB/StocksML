# Importing necessary libraries
from imports import *
import candlesticks as cs

# Fetching historical data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Adding technical indicators
def add_technical_indicators (df):
    # Optimization: Compute rolling statistics in one go where possible to avoid repeated calls
    close_prices = df['Close']
    
    # SMA
    df['SMA_14'] = close_prices.rolling(window=14).mean()
    df['SMA_50'] = close_prices.rolling(window=50).mean()
    df['SMA_200'] = close_prices.rolling(window=200).mean()

    # EMA
    df['EMA_50'] = close_prices.ewm(span=50, adjust=False).mean()
    df['EMA_200'] = close_prices.ewm(span=200, adjust=False).mean()

    # RSI (Combined gain and loss into a single calculation to reduce repeated operations)
    delta = close_prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # On-balance Volume
    df['OBV'] = calculate_obv(df)

    # Money Flow Index
    df['MFI'] = calculate_mfi(df, period=10)

    # Calculate CCI
    df['CCI'] = calculate_cci(df, period=10)

    # Bollinger Bands
    rolling_20 = close_prices.rolling(window=20)
    df['BB_Middle'] = rolling_20.mean()
    rolling_std = rolling_20.std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * rolling_std
    df['BB_Lower'] = df['BB_Middle'] - 2 * rolling_std

    # Momentum and ROC
    df['Momentum_10'] = close_prices - close_prices.shift(10)
    df['Momentum_30'] = close_prices - close_prices.shift(30)
    df['ROC_10'] = close_prices.pct_change(periods=10) * 100
    df['ROC_30'] = close_prices.pct_change(periods=14) * 100

    # ATR (Optimized true range calculation)
    df['High_Low'] = df['High'] - df['Low']
    df['High_Close'] = (df['High'] - df['Close'].shift(1)).abs()
    df['Low_Close'] = (df['Low'] - df['Close'].shift(1)).abs()

    df['True_Range'] = pd.concat([df['High_Low'], df['High_Close'], df['Low_Close']], axis=1).max(axis=1)
    df['ATR'] = df['True_Range'].rolling(window=14).mean()

    # Drop rows with NaN values resulting from rolling calculations
    df.dropna(inplace=True)

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

def calculate_mfi(data, period=9):
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
    data.drop(columns=['Positive_MF', 'Negative_MF', 'Positive_MF_sum', 'Negative_MF_sum', 'MFR'], inplace=True)

    return data['MFI']

def calculate_cci(data, period=9):
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
    features = ['SMA_14', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower',
                'Momentum_10', 'Momentum_30', 'ROC_10', 'ROC_30', 'Bullish_Engulfing', 'Doji', 'Hammer', 
                'Hanging_Man', 'Morning_Star', 'Evening_Star', 'Shooting_Star', 'Three_White_Soldiers', 
                'Three_Black_Crows', 'Volume', 'ATR', 'MFI', 'CCI']
    
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
    sma_50_threshold = 0.02 
    sma_200_threshold = 0.02
    ema_50_threshold = 0.02
    ema_200_threshold = 0.02
    rsi_threshold_buy = 35
    rsi_threshold_sell = 65
    bb_threshold = 0.02

    # Extract the latest values of the technical indicators
    sma_50 = last_row['SMA_50']
    sma_200 = last_row['SMA_200']
    ema_50 = last_row['EMA_50']
    ema_200 = last_row['EMA_200']
    rsi = last_row['RSI']
    bb_lower = last_row['BB_Lower']
    bb_upper = last_row['BB_Upper']

    # Initialize signals
    buy_signal = False
    sell_signal = False
    
    # Check if current price is above the SMA and EMA thresholds
    if current_price > (1 + sma_50_threshold) * sma_50:
        buy_signal = True
    if current_price > (1 + sma_200_threshold) * sma_200:
        buy_signal = True
    if current_price > (1 + ema_50_threshold) * ema_50:
        buy_signal = True
    if current_price > (1 + ema_200_threshold) * ema_200:
        buy_signal = True

    # Check RSI for buy/sell signals
    if rsi < rsi_threshold_buy:
        buy_signal = True
    if rsi > rsi_threshold_sell:
        sell_signal = True

    # Check if current price is below the Bollinger Bands Lower Band
    if current_price < (1 - bb_threshold) * bb_lower:
        buy_signal = True
    if current_price > (1 + bb_threshold) * bb_upper:
        sell_signal = True

    # Generate final signal
    if buy_signal and not sell_signal:
        return "BUY"
    elif sell_signal and not buy_signal:
        return "SELL"
    else:
        return "HODL / SIDELINES"
    
#### PREDICT PRICES #####
def predict_prices(model, recent_data, scaler, num_days=5, window_size=30):
    # Use the same features during prediction
    features = ['SMA_14', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower',
                'Momentum_10', 'Momentum_30', 'ROC_10', 'ROC_30', 'Bullish_Engulfing', 'Doji', 'Hammer', 
                'Hanging_Man', 'Morning_Star', 'Evening_Star', 'Shooting_Star', 'Three_White_Soldiers', 
                'Three_Black_Crows', 'Volume', 'ATR', 'MFI', 'CCI']
    
    last_data = recent_data.copy()  # Copy the whole dataframe to modify
    
    predicted_prices = []  # List to store predicted values

    for i in range(num_days):
        # Use a rolling window of size 'window_size' from actual data (historical data)
        sliced = last_data.iloc[-window_size:].copy()
        
        # Ensure all technical indicators are calculated before prediction
        sliced['SMA_14'] = sliced['Close'].rolling(window=14, min_periods=1).mean()
        sliced['SMA_50'] = sliced['Close'].rolling(window=50, min_periods=1).mean()
        sliced['SMA_200'] = sliced['Close'].rolling(window=200, min_periods=1).mean()
        sliced['EMA_50'] = sliced['Close'].ewm(span=50, adjust=False).mean()
        sliced['EMA_200'] = sliced['Close'].ewm(span=200, adjust=False).mean()

        # Momentum and ROC
        sliced['Momentum_10'] = sliced['Close'] - sliced['Close'].shift(10)
        sliced['Momentum_30'] = sliced['Close'] - sliced['Close'].shift(30)
        sliced['ROC_10'] = sliced['Close'].pct_change(periods=10) * 100
        sliced['ROC_30'] = sliced['Close'].pct_change(periods=30) * 100  # Change to 30

        sliced['RSI'] = calculate_rsi(sliced)
        sliced['MFI'] = calculate_mfi(sliced)
        sliced['CCI'] = calculate_cci(sliced)
        
        sliced['BB_Middle'] = sliced['Close'].rolling(window=20, min_periods=1).mean()
        sliced['BB_Upper'] = sliced['BB_Middle'] + 2 * sliced['Close'].rolling(window=20, min_periods=1).std()
        sliced['BB_Lower'] = sliced['BB_Middle'] - 2 * sliced['Close'].rolling(window=20, min_periods=1).std()
        
        # Candlestick patterns (use actual data for pattern detection)
        sliced = add_candlestickpatterns(sliced)

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

    return df
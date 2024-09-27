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
    df['MFI'] = calculate_mfi(df)

    # Bollinger Bands
    rolling_20 = close_prices.rolling(window=20)
    df['BB_Middle'] = rolling_20.mean()
    rolling_std = rolling_20.std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * rolling_std
    df['BB_Lower'] = df['BB_Middle'] - 2 * rolling_std

    # Momentum and ROC
    df['Momentum_10'] = close_prices - close_prices.shift(10)
    df['Momentum_30'] = close_prices - close_prices.shift(14)
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

def calculate_mfi(data, period=14):
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

    return data['MFI']

# Preparing the data for Machine Learning
def prepare_ml_data(df):
    # Include candlestick pattern features and new indicators
    features = ['SMA_14', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower',
                'Momentum_10', 'Momentum_30', 'ROC_10', 'ROC_30', 'Bullish_Engulfing', 'Doji', 'Hammer', 
                'Hanging_Man', 'Morning_Star', 'Evening_Star', 'Shooting_Star', 'Three_White_Soldiers', 
                'Three_Black_Crows', 'Volume', 'ATR', 'MFI']
    
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


# Plotting the stock price and technical indicators
def plot_technical_indicators(df, ticker = '   ' ):
    plt.figure(figsize=(14, 10))
    
    # Close Price and Moving Averages
    plt.subplot(3, 1, 1)
    plt.plot(df['Close'], label='Close Price', alpha=0.6)

    plt.plot(df['EMA_50'], color = 'red', label='50-day EMA', alpha=0.6)
    plt.plot(df['EMA_200'], color = 'magenta', label='200-day EMA', alpha=0.6)
    
    plt.title(f'{ticker} Price and Moving Averages')
    
    plt.legend()
    #plt.yscale('log')
    plt.minorticks_on()
    plt.tick_params(which='both', axis='y', direction='in', length=6)
    plt.tick_params(which='minor', axis='y', direction='in', length=4)
    plt.grid(alpha=0.5)

    # OBV
    plt.subplot(3, 1, 2)
    plt.plot(df['OBV'], label='OBV', color='gray', alpha=0.5)
    plt.title('On Balance Volume')
    plt.legend()
    plt.grid(alpha=0.5)
    
    # RSI
    plt.subplot(3, 1, 3)
    plt.plot(df['RSI'], label='RSI', color='gray', alpha=0.5)
    plt.title('Relative Strength Index (RSI)')
    plt.axhline(70, color='red', linestyle='--', alpha=0.5)
    plt.axhline(30, color='green', linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.grid(alpha=0.5)
    plt.show()
    
#### PREDICT PRICES #####
def predict_prices(model, recent_data, scaler, num_days=5, window_size=30):
    # Use the same features during prediction
    features = ['SMA_14', 'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower',
                'Momentum_10', 'Momentum_30', 'ROC_10', 'ROC_30', 'Bullish_Engulfing', 'Doji', 'Hammer', 
                'Hanging_Man', 'Morning_Star', 'Evening_Star', 'Shooting_Star', 'Three_White_Soldiers', 
                'Three_Black_Crows', 'Volume', 'ATR', 'MFI']
    
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
        
        sliced['RSI'] = calculate_rsi(sliced)
        sliced['MFI'] = calculate_mfi(sliced)
        
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

def plot_with_predictions(stock_df, predicted_prices, ticker='NONE', num_days=5):
    # Get the last month of historical data
    
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    end_date = stock_df.index[-1]
    start_date = end_date - pd.DateOffset(months=1)
    one_month_data = stock_df.loc[start_date:end_date]
    
    # Get the last known closing price
    last_close = one_month_data['Close'].iloc[-1]
    
    # Generate dates for the predicted prices
    prediction_dates = pd.date_range(start=end_date + pd.DateOffset(days=1), periods=num_days)
    
    # Create a DataFrame for predicted prices with the last known close included
    predictions_df = pd.DataFrame({
        'Date': [end_date] + list(prediction_dates),
        'Predicted_Price': [last_close] + predicted_prices
    }).set_index('Date')
    
    # Combine historical data with predicted prices
    combined_df = pd.concat([one_month_data[['Close']], predictions_df])
    
    # Plot historical closing prices and predicted prices
    plt.figure(figsize=(14, 7))
    
    # Plot historical closing prices
    plt.plot(one_month_data.index, one_month_data['Close'], label='Historical Close Prices', 
             color='blue', alpha=0.7)
    
    # Plot predicted prices
    plt.plot(predictions_df.index, predictions_df['Predicted_Price'], label='Predicted Prices', 
             color='blue', linestyle='--', marker='o', alpha=0.4)
    
    # Formatting the plot
    plt.title(f'{ticker} {current_date} - Closing Prices and Next {num_days} Days Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Calculate statistics
    predicted_max = np.max(predicted_prices)
    predicted_min = np.min(predicted_prices)
    predicted_change = ((predicted_prices[-1] - last_close) / last_close) * 100

    # Add text annotation
    textstr = (f'Predicted % Change: {predicted_change:.2f}%\n'
               f'Min Price: ${predicted_min:.2f}\n'
               f'Max Price: ${predicted_max:.2f}')

    plt.text(0.5, 0.5, ticker, transform=plt.gca().transAxes, 
             fontsize=100, color='grey', alpha=0.1,  # Adjust transparency here
             horizontalalignment='center', verticalalignment='center',
             rotation=45, weight='bold', style='italic')
    
    plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', alpha=0.2, facecolor='white'))

    path = r'C:\Users\Farrukh\jupyter-Notebooks\STOCKS\predictions'
    fname = f'{current_date}_{ticker}.png'
    fpath = os.path.join(path, fname)
    plt.savefig(fpath, bbox_inches='tight')
    plt.show()
    plt.close()
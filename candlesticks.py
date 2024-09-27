# Importing necessary libraries
import pandas as pd
import numpy as np

### CANDLESTICK PATTERNS

# Function to detect Doji pattern
def detect_doji(df, tolerance=0.001):
    return np.where(abs(df['Open'] - df['Close']) / df['Open'] < tolerance, 1, 0)

# Function to detect Hammer pattern
def detect_hammer(df):
    body = abs(df['Close'] - df['Open'])
    lower_shadow = np.where(df['Open'] > df['Close'], df['Open'] - df['Low'], df['Close'] - df['Low'])
    upper_shadow = np.where(df['Close'] > df['Open'], df['High'] - df['Close'], df['High'] - df['Open'])
    
    return np.where((lower_shadow > 2 * body) & (upper_shadow < body), 1, 0)

# Function to detect Hanging Man pattern
def detect_hanging_man(df):
    body = abs(df['Close'] - df['Open'])
    lower_shadow = np.where(df['Open'] > df['Close'], df['Open'] - df['Low'], df['Close'] - df['Low'])
    upper_shadow = np.where(df['Close'] > df['Open'], df['High'] - df['Close'], df['High'] - df['Open'])
    
    return np.where((lower_shadow > 2 * body) & (upper_shadow < body) & (df['Open'] > df['Close']), 1, 0)

# Function to detect Morning Star pattern
def detect_morning_star(df):
    pattern = np.zeros(len(df))
    
    for i in range(2, len(df)):
        first_candle = df.iloc[i-2]
        second_candle = df.iloc[i-1]
        third_candle = df.iloc[i]
        
        if (first_candle['Close'] < first_candle['Open'] and 
            abs(second_candle['Close'] - second_candle['Open']) < 0.002 * second_candle['Open'] and 
            third_candle['Close'] > third_candle['Open'] and 
            third_candle['Close'] > first_candle['Open']):
            pattern[i] = 1
            
    return pattern

# Function to detect Evening Star pattern
def detect_evening_star(df):
    pattern = np.zeros(len(df))
    
    for i in range(2, len(df)):
        first_candle = df.iloc[i-2]
        second_candle = df.iloc[i-1]
        third_candle = df.iloc[i]
        
        if (first_candle['Close'] > first_candle['Open'] and 
            abs(second_candle['Close'] - second_candle['Open']) < 0.002 * second_candle['Open'] and 
            third_candle['Close'] < third_candle['Open'] and 
            third_candle['Close'] < first_candle['Open']):
            pattern[i] = 1
    
    return pattern

# Function to detect Shooting Star pattern
def detect_shooting_star(df):
    body = abs(df['Close'] - df['Open'])
    upper_shadow = np.where(df['Close'] > df['Open'], df['High'] - df['Close'], df['High'] - df['Open'])
    lower_shadow = np.where(df['Open'] > df['Close'], df['Open'] - df['Low'], df['Close'] - df['Low'])
    
    return np.where((upper_shadow > 2 * body) & (lower_shadow < body), 1, 0)

# Function to detect Three White Soldiers pattern
def detect_three_white_soldiers(df):
    pattern = np.zeros(len(df))
    
    for i in range(2, len(df)):
        first_candle = df.iloc[i-2]
        second_candle = df.iloc[i-1]
        third_candle = df.iloc[i]
        
        if (first_candle['Close'] > first_candle['Open'] and 
            second_candle['Close'] > second_candle['Open'] and 
            third_candle['Close'] > third_candle['Open'] and 
            third_candle['Close'] > second_candle['Close'] and 
            second_candle['Close'] > first_candle['Close']):
            pattern[i] = 1
    
    return pattern

# Function to detect Three Black Crows pattern
def detect_three_black_crows(df):
    pattern = np.zeros(len(df))
    
    for i in range(2, len(df)):
        first_candle = df.iloc[i-2]
        second_candle = df.iloc[i-1]
        third_candle = df.iloc[i]
        
        if (first_candle['Close'] < first_candle['Open'] and 
            second_candle['Close'] < second_candle['Open'] and 
            third_candle['Close'] < third_candle['Open'] and 
            third_candle['Close'] < second_candle['Close'] and 
            second_candle['Close'] < first_candle['Close']):
            pattern[i] = 1
    
    return pattern

# Function to detect bullish engulfing pattern
def detect_bullish_engulfing(df):
    pattern = np.zeros(len(df))
    
    for i in range(1, len(df)):
        prev_close = df['Close'].iloc[i-1]
        prev_open = df['Open'].iloc[i-1]
        curr_close = df['Close'].iloc[i]
        curr_open = df['Open'].iloc[i]
        
        if (prev_open > prev_close and 
            curr_close > curr_open and 
            curr_close > prev_open and 
            curr_open < prev_close):
            pattern[i] = 1
    
    return pattern

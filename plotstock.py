from imports import *
from ipywidgets import interact

def plot_stock_by_year(stock_df, year, ticker='NONE'):
    # Filter the stock data by the selected year
    year_data = stock_df[stock_df.index.year == year]
    
    # Plot the data for the selected year
    plt.figure(figsize=(14, 7))
    plt.plot(year_data.index, year_data['Close'], label=f'{ticker} {year} Close Prices', color='blue')
    
    plt.title(f'{ticker} - Closing Prices for Year {year}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_with_slider(stock_df, ticker='NONE'):
    min_year = stock_df.index.year.min()  # Get the earliest year in the dataset
    max_year = stock_df.index.year.max()  # Get the latest year in the dataset
    
    # Use `interact` to create a slider for year selection
    interact(lambda year: plot_stock_by_year(stock_df, year, ticker),
             year=(min_year, max_year))
    
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

    # RSI and CCI with Secondary Axis
    plt.subplot(3, 1, 3)
    #plt.plot(df['RSI'], label='RSI', color='gray', alpha=0.5)
    plt.plot(df['RSI2'], label='RSI-long', color='red', alpha=0.5)
    plt.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
    
    ax2 = plt.gca().twinx()  # Create a secondary y-axis
    ax2.plot(df['CCI'], label='CCI', color='orange', alpha=0.5)  # Plot CCI on the secondary axis
    ax2.axhline(100, color='purple', linestyle='--', alpha=0.5, label='CCI Overbought (100)')
    ax2.axhline(-100, color='brown', linestyle='--', alpha=0.5, label='CCI Oversold (-100)')
    
    plt.title('RSI and CCI')
    plt.legend(loc='upper left')
    ax2.legend(loc='upper right')  # Legend for CCI on the right

    plt.tight_layout()
    plt.grid(alpha=0.5)
    plt.show()

def plot_with_predictions(stock_df, predicted_prices, ticker='NONE', num_days=5):
    # Get the current date
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    end_date = stock_df.index[-1]
    start_date = end_date - pd.DateOffset(months=12)
    one_month_data = stock_df.loc[start_date:end_date].copy()

    last_close = one_month_data['Close'].iloc[-1]
    
    # Calculate statistics for predictions
    predicted_max = round(np.max(predicted_prices), 3)
    predicted_min = round(np.min(predicted_prices), 3)
    predicted_change = ((predicted_prices[-1] - last_close) / last_close) * 100
    print("Predicted (min, max): " f'{predicted_min}, {predicted_max}')

    prediction_dates = pd.date_range(start=end_date + pd.DateOffset(days=1), periods=num_days)
    
    predictions_df = pd.DataFrame({
        'Date': [end_date] + list(prediction_dates),
        'Predicted_Price': [last_close] + predicted_prices
    }).set_index('Date')
    
    combined_df = pd.concat([one_month_data[['Close']], predictions_df])
    
    # Create subplots
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12, 12), dpi=300, sharex=True)
    
    # Plot historical closing prices and indicators (EMA50, EMA200)
    axs[0].plot(one_month_data.index, one_month_data['Close'], label='Historical Close Prices', 
                color='gray', alpha=0.7)
    axs[0].plot(one_month_data.index, one_month_data['EMA1'], label='EMA20', 
                color='orange', alpha=0.7)
    axs[0].plot(one_month_data.index, one_month_data['EMA2'], label='EMA50', 
                color='red', alpha=0.7)
    axs[0].plot(one_month_data.index, one_month_data['EMA3'], label='EMA100', 
                color='magenta', alpha=0.7)
    
    # Plot predicted prices
    axs[0].plot(predictions_df.index, predictions_df['Predicted_Price'], label='Predicted Prices', 
                color='blue', marker='o', markersize=2, alpha=0.4)

    # Set labels and grid for the first subplot
    axs[0].set_title(f'{ticker} {current_date} - Closing Prices and ML Predictions (TIs w/o News/Media)')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Price')
    axs[0].legend(loc = 'upper left')
    axs[0].grid(True)
    axs[0].tick_params(axis='x', rotation=0)
    

    # Add annotation with prediction statistics
    textstr = (f'Predicted % Change: {predicted_change:.2f}%\n'
               f'Min Price: ${predicted_min:.2f}\n'
               f'Max Price: ${predicted_max:.2f}')

    axs[0].text(0.5, 0.5, ticker, transform=axs[0].transAxes, 
                fontsize=50, color='grey', alpha=0.2,  # Adjust transparency here
                horizontalalignment='center', verticalalignment='center',
                rotation=0, weight='bold', style='italic')
    
    axs[0].text(0.95, 0.05, textstr, transform=axs[0].transAxes, fontsize=12,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', alpha=0.2, facecolor='white'))

    # Plot On Balance Volume (OBV) on the second subplot
    axs[1].plot(one_month_data.index, one_month_data['OBV'], label='On Balance Volume', 
                color='gray', alpha=0.7)
    axs[1].set_ylabel('OBV')
    axs[1].legend(loc = 'upper left')
    axs[1].grid(alpha=0.5)

    # Plot Relative Strength Index (RSI) on the third subplot
    axs[2].plot(one_month_data.index, one_month_data['RSI'], label='Relative Strength Index (RSI)', 
                color='gray', alpha=0.7)
    axs[2].plot(one_month_data.index, one_month_data['RSI2'], label='Smoothed RSI', 
                color='red', alpha=0.7)
    axs[2].axhline(70, color='red', linestyle='--', alpha=0.5)
    axs[2].axhline(30, color='green', linestyle='--', alpha=0.5)
 
    # Fill area above 70 (overbought)
    axs[2].fill_between(one_month_data.index, one_month_data['RSI'], 70, 
                    where=(one_month_data['RSI'] > 70), 
                    color='green', alpha=0.3, interpolate=True)
    
    # Fill area below 30 (oversold)
    axs[2].fill_between(one_month_data.index, one_month_data['RSI'], 30, 
                    where=(one_month_data['RSI'] < 30), 
                    color='red', alpha=0.3, interpolate=True)
    
    axs[2].set_ylabel('RSI')
    axs[2].legend(loc = 'upper left')
    axs[2].grid(alpha=0.5)
    
    buyVol = one_month_data['sumBuyVol']
    sellVol = one_month_data['sumSellVol']

    axs[3].plot(one_month_data.index, buyVol, label='Cumltv. Buys', 
                color='green', alpha=0.7)
    axs[3].plot(one_month_data.index, sellVol, label='Cumltv. Sells', 
                color='red', alpha=0.7)

    # Fill areas of buy/sell pressure
    axs[3].fill_between(one_month_data.index, buyVol, sellVol, where=(buyVol > sellVol), 
                    color='green', alpha=0.3, interpolate=True)
    axs[3].fill_between(one_month_data.index, sellVol, buyVol, where=(buyVol < sellVol), 
                    color='red', alpha=0.3, interpolate=True)
    
    axs[3].set_ylabel('20-period Cumulative Buy-Sell')
    axs[3].legend(loc = 'upper left')
    axs[3].grid(alpha=0.5)

    
    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Decrease vertical spacing between subplots
    plt.subplots_adjust(hspace=0.1)

    # Save the figure to disk
    path = r'C:\Users\Farrukh\jupyter-Notebooks\STOCKS\predictions'
    fname = f'{current_date}_{ticker}.png'
    fpath = os.path.join(path, fname)
    plt.savefig(fpath, bbox_inches='tight')
    
    # Display the plot
    plt.show()
    plt.close()


def plot_obv_pvt(df, pvt=True, obv=True, ticker='NONE', nrMonths = 12):
    end_date = df.index[-1]
    start_date = end_date - pd.DateOffset(months=nrMonths)
    
    df2 = df.loc[start_date:end_date].copy()

    # Scale OBV to the range of Close prices
    if obv:
        obv_min, obv_max = df2['OBV'].min(), df2['OBV'].max()
        close_min, close_max = df2['Close'].min(), df2['Close'].max()
        df2['OBV_scaled'] = ((df2['OBV'] - obv_min) / (obv_max - obv_min)) * (close_max - close_min) + close_min

    # Plotting Close Price, PVT, and OBV
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot Close Price
    ax1.plot(df2.index, df2['Close'], color='black', label='Close Price')
    ax1.set_ylabel('Close Price', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Initialize empty variables for lines and labels to manage legend later
    lines1, labels1, lines2, labels2, lines3, labels3 = [], [], [], [], [], []
    
    if pvt:
        # Plot PVT on the secondary axis
        ax2 = ax1.twinx()
        sma1 = df2['PVT'].rolling(window=21).mean()
        ax2.plot(df2.index, df2['PVT'], color='blue', label='Price Volume Trend (PVT)')
        ax2.plot(df2.index, sma1, color='blue', label='SMA PVT')
        ax2.set_ylabel('PVT', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        lines2, labels2 = ax2.get_legend_handles_labels()
    
    if obv:
        # Plot scaled OBV on a third axis
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # Move third axis outward
        ax3.plot(df2.index, df2['OBV_scaled'], color='green', label='On-Balance Volume (OBV)', linestyle='--')
        ax3.set_ylabel('Scaled OBV', color='green')
        ax3.tick_params(axis='y', labelcolor='green')
        lines3, labels3 = ax3.get_legend_handles_labels()
    
    # Collect all lines and labels from each axis for the legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    
    # Add legend within the figure (adjust position)
    fig.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)

    fig.text(0.5, 0.5, ticker, transform=ax1.transAxes, 
            fontsize=50, color='grey', alpha=0.2,  # Adjust transparency here
            horizontalalignment='center', verticalalignment='center',
            rotation=0, weight='bold', style='italic')

    plt.title(f'{ticker} {end_date} - Close Price, PVT, and Scaled OBV Combined')
    plt.grid(True)
    plt.show()


def plot_pvt(df, pvt=True, ticker='NONE', nrMonths=12):
    # Ensure the DataFrame index is a datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be a datetime index.")
    
    # Define the date range for plotting
    end_date = df.index[-1]
    start_date = end_date - pd.DateOffset(months=nrMonths)
    
    # Slice the DataFrame for the given date range
    df2 = df.loc[start_date:end_date].copy()

    # Create a figure and axis for plotting
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot Close Price on the primary y-axis
    ax1.plot(df2.index, df2['Close'], color='black', label='Close Price')
    ax1.set_ylabel('Close Price', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create a second y-axis for PVT
    ax2 = ax1.twinx()

    # Plot PVT and its SMA on the secondary y-axis
    ax2.plot(df2.index, df2['PVT'], color='blue', label='PVT')
    sma1 = df2['PVT'].rolling(window=21).mean()
    ax2.plot(df2.index, sma1, color='red', label='Long PVT (21-Day SMA)')
    
    ax2.set_ylabel('PVT', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Adding a title and grid
    plt.title(f'{ticker} {end_date.strftime("%Y-%m-%d")} - Close Price and PVT')
    ax1.grid(True)
    
    # Combine legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()
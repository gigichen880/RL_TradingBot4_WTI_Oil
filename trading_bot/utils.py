import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Formats Position
format_position = lambda price: ('-$' if price < 0 else '+$') + '{0:.5f}'.format(abs(price))

# Formats Currency
format_currency = lambda price: '${0:.5f}'.format(abs(price))

def show_train_result(result, val_position, initial_offset, train_mlpd, train_md, train_std, val_mlpd, val_md, val_std):
    """ Displays training results
    """
    if val_position == initial_offset or val_position == 0.0:
        print('Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}'
                     .format(result[0]+1, result[1]+1, format_position(result[2]), result[3]))
    else:
        print('Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f})'
                     .format(result[0]+1, result[1]+1, format_position(result[2]), format_position(val_position), result[3],))

    print("Train - Max Loss Per Day: {}; Max Drawdown: {}; Std: {}".format(train_mlpd, train_md, train_std))
    print("Val - Max Loss Per Day: {}; Max Drawdown: {}; Std: {}".format(val_mlpd, val_md, val_std))


def show_eval_result(model_name, profit, initial_offset, test_mlpd, test_md, test_std):
    """ Displays eval results
    """
    if profit == initial_offset or profit == 0.0:
        print('{}: USELESS\n'.format(model_name))
    else:
        print('{}: {}\n'.format(model_name, format_position(profit)))
    print("Test - Max Loss Per Day: {}; Max Drawdown: {}; Std: {}".format(test_mlpd, test_md, test_std))


def get_stock_data(stock_file):
    """Reads stock data from csv file
    """
    df = pd.read_csv(stock_file)
    WTI = df.where(df['Symbol'] == 'Crude Oil WTI').dropna()
    brent = df.where(df['Symbol'] == 'Brent Oil').dropna()
    natgas = df.where(df['Symbol'] == 'Natural Gas').dropna()
    heating = df.where(df['Symbol'] == 'Heating Oil').dropna()
    return list(WTI['Close']), list(brent['Close']), list(natgas['Close']), list(heating['Close'])


def max_drawdown(series):
    series = np.array(series)
    peak = np.maximum.accumulate(series)
    drawdown = (series - peak) / peak
    drawdown[np.isnan(drawdown)] = 0
    max_dd = np.min(drawdown)

    return max_dd

def interpret_results(pos_list):
    arr = pd.Series(pos_list[50:])
    mlpd = min(arr.diff()[1:])
    std = np.std(arr.diff()[1:])
    mdd = max_drawdown(arr)
    profit = pos_list[len(pos_list)-1] - pos_list[0]
    return mlpd, mdd, std, profit

def get_benchmarks(df):

    result_list = []
    for i in range(4):
        arr = pd.Series(df[i][:1000])
        tup = dual_moving_average_trading(arr)
        result_list.append(tup)
    return result_list

def dual_moving_average_trading(orig_dat, short_window=50, long_window=200, initial_balance=10000, trade_size=1):
    # Convert prices to a pandas DataFrame
    df = pd.DataFrame(orig_dat, columns=['Price'])
    last = df['Price'].iloc[len(df)-1]
    # Calculate the short-term and long-term moving averages
    df['SMA_short'] = df['Price'].rolling(window=short_window, min_periods=1).mean()
    df['SMA_long'] = df['Price'].rolling(window=long_window, min_periods=1).mean()

    # Signals: Buy when short-term crosses above long-term, sell when short-term crosses below long-term
    df['Signal'] = 0  # 0 = hold, 1 = buy, -1 = sell
    df['Signal'] = np.where(df['SMA_short'] > df['SMA_long'], 1, -1)

    # Track money and stock holding
    money = initial_balance
    stock = 0
    money_series = []
    ii = 0
    # Loop over each day to simulate trading
    for i in range(1, len(df)):
        if df['Signal'].iloc[i] == 1 and df['Signal'].iloc[i - 1] == -1:
            # Buy signal (Golden Cross)
            money -= df['Price'].iloc[i] * trade_size  # Deduct from the money in the account
            # print(f"Buying at {df['Price'].iloc[i]:.2f}, money: {money:.2f}")
            ii = -1

        elif df['Signal'].iloc[i] == -1 and df['Signal'].iloc[i - 1] == 1:
            # Sell signal (Death Cross)
            money += trade_size * df['Price'].iloc[i]  # Sell all stock
            # print(f"Selling at {df['Price'].iloc[i]:.2f}, money: {money:.2f}")
            ii = 1

        # Track money in the account each day
        money_series.append(money)
    return interpret_results(money_series)

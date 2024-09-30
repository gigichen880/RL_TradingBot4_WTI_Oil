import os
import math
import logging

import pandas as pd
import numpy as np

# Formats Position
format_position = lambda price: ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))

# Formats Currency
format_currency = lambda price: '${0:.2f}'.format(abs(price))

def show_train_result(result, val_position, initial_offset, train_mlpd, train_md, train_std, val_mlpd, val_md, val_std):
    """ Displays training results
    """
    if val_position == initial_offset or val_position == 0.0:
        print('Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}'
                     .format(result[0], result[1], format_position(result[2]), result[3]))
    else:
        print('Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f})'
                     .format(result[0], result[1], format_position(result[2]), format_position(val_position), result[3],))

    print("Train - Max Loss Per Day: {}; Max Drawdown: {}%; Std: {}".format(train_mlpd, train_md, train_std))
    print("Val - Max Loss Per Day: {}; Max Drawdown: {}; Std: {}".format(val_mlpd, val_md, val_std))


def show_eval_result(model_name, profit, initial_offset, test_mlpd, test_md, test_std):
    """ Displays eval results
    """
    if profit == initial_offset or profit == 0.0:
        print('{}: USELESS\n'.format(model_name))
    else:
        print('{}: {}\n'.format(model_name, format_position(profit)))
    print("Test - Max Loss Per Day: {}; Max Drawdown: {}%; Std: {}".format(test_mlpd, test_md, test_std))


def get_stock_data(stock_file):
    """Reads stock data from csv file
    """
    df = pd.read_csv(stock_file)
    WTI = df.where(df['Symbol'] == 'Crude Oil WTI').dropna()
    brent = df.where(df['Symbol'] == 'Brent Oil').dropna()
    return list(WTI['Close']), list(brent['Close'])

def interpret_results(profit_list):
    # print(profit_list)
    arr = pd.Series(profit_list)
    mlpd = min(arr.diff()[1:])
    md = (max(arr) - min(arr)) / max(arr) * 100
    std = np.std(arr)
    return mlpd, md, std



# def switch_k_backend_device():
#     """ Switches `keras` backend from GPU to CPU if required.
#
#     Faster computation on CPU (if using tensorflow-gpu).
#     """
#     if K.backend() == "tensorflow":
#         logging.debug("switching to TensorFlow for CPU")
#         os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Benchmark: Buy-and-Hold Strategy for Brent Oil:

# file = "../../../pricedata.csv"
# arr = pd.Series(get_stock_data(file)[0])

 # - Profit: 88.73 (Brent)
# - Profit: 84.01 (WTI)
# print(arr[len(arr)-1] - arr[0])
#
# #  - Maximum Loss Per Day: -16.84 (Brent)
# #  - Maximum Loss Per Day: -18.02 (WTI)
# print(min(arr.diff()[1:]))
#
# #  - Max Drawdown: 87.897% (Brent)
# #  - Max Drawdown: 87.989% (WTI)
# print((max(arr) - min(arr)) / max(arr) * 100, "%")
#
# #  - Volatility: 29.3807 (Brent)
# #  - Volatility: 25.5687 (WTI)
# print(np.std(arr))
# Test - Max Loss Per Day: -14.310000000000002; Std: 19.083059014320696
# 20.64% decrease in max daily loss, 25.37% decrease in std, 15.66%


# Test - Max Loss Per Day: -6.990000000000013; Max Drawdown: 4865.346534653422%; Std: 12.56465977587323
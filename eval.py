"""
Script for evaluating Stock Trading Bot.

Usage:
  eval.py <eval-stock> [--window-size=<window-size>] [--model-name=<model-name>] [--debug]

Options:
  --window-size=<window-size>   Size of the n-day window stock data representation used as the feature vector. [default: 10]
  --model-name=<model-name>     Name of the pretrained model to use (will eval all models in `models/` if unspecified).
  --debug                       Specifies whether to use verbose logs during eval operation.
"""

import os
import coloredlogs

from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import evaluate_model
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_eval_result,
    interpret_results
    # switch_k_backend_device
)


def main(eval_stock, window_size, model_name, debug, noep=5):
    """ Evaluates the stock trading bot.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python eval.py --help]
    """    
    data = get_stock_data(eval_stock)[1]
    initial_offset = data[1] - data[0]

    # Single Model Evaluation
    if model_name is not None:
        agent = Agent(window_size, pretrained = True, model_name=model_name, noep=noep)
        profit, _, test_profit_li = evaluate_model(agent, data, window_size, debug)
        test_mlpd, test_md, test_std = interpret_results(test_profit_li)
        show_eval_result(model_name, profit, initial_offset, test_mlpd, test_md, test_std)
        
    # Multiple Model Evaluation
    else:
        for model in os.listdir("models"):
            if os.path.isfile(os.path.join("models", model)):
                agent = Agent(window_size, pretrained = True, model_name=model, noep=noep)
                profit, _, test_profit_li = evaluate_model(agent, data, window_size, debug)
                test_mlpd, test_md, test_std = interpret_results(test_profit_li)
                show_eval_result(model, profit, initial_offset, test_mlpd, test_md, test_std)
                del agent


if __name__ == "__main__":

    eval_stock = "../../pricedata.csv"
    window_size = 6
    batch_size = 20
    ep_count = 3
    strategy = "t-dqn"
    model_name = "tryout"
    coloredlogs.install(level="DEBUG")
    # switch_k_backend_device()

    noeps = [0, 1, 2, 3, 4, 5]
    for noep in noeps:
        try:
            main(eval_stock, window_size, model_name, debug=False, noep=noep)
        except KeyboardInterrupt:
            print("Aborted")


import logging
import coloredlogs
import torch
torch.autograd.set_detect_anomaly(True)
from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import train_model, evaluate_model
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_train_result,
    interpret_results
)


def main(train_stock, val_stock, window_size, batch_size, ep_count,
         strategy="t-dqn", model_name="model_debug", pretrained = False,
         debug=True, noep=5):
    """ Trains the stock trading bot using Deep Q-Learning.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python train.py --help]
    """
    agent = Agent(window_size, strategy=strategy, model_name=model_name, noep=noep)
    
    train_data = get_stock_data(train_stock)[0]
    val_data = get_stock_data(val_stock)[1]

    initial_offset = val_data[1] - val_data[0]

    for episode in range(0, ep_count + 1):
        train_result = train_model(agent, episode, train_data, ep_count=ep_count,
                                   batch_size=batch_size, window_size=window_size)
        val_result, _ = evaluate_model(agent, val_data, window_size, debug)
        train_mlpd, train_md, train_std = interpret_results(train_result[4])
        val_mlpd, val_md, val_std = interpret_results(val_result[2])
        show_train_result(train_result, val_result, initial_offset, train_mlpd, train_md, train_std, val_mlpd, val_md, val_std)


if __name__ == "__main__":
    train_stock = "../../pricedata.csv"
    val_stock = "../../pricedata.csv"
    window_size = 5
    batch_size = 50
    ep_count = 3
    strategy = "t-dqn"
    model_name = "rl1"

    try:
        main(train_stock, val_stock, window_size, batch_size,
             ep_count, strategy=strategy, model_name=model_name, pretrained = False,
             debug=False, noep=5)
    except KeyboardInterrupt:
        print("Aborted!")

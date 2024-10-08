import torch
import pandas as pd
torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt
from trading_bot.agent import Agent
from trading_bot.methods import train_model, evaluate_model
from trading_bot.utils import (
    get_stock_data,
    show_train_result,
    interpret_results,
    get_benchmarks
)


def main(train_stock, val_stock, window_size, batch_size, ep_count,
         strategy="t-dqn", model_name="model_debug", pretrained = False,
         debug=True, noep=5):
    agent = Agent(window_size, strategy=strategy, model_name=model_name, noep=noep)
    
    train_data = get_stock_data(train_stock)[0][:1000]
    val_data = get_stock_data(val_stock)[1][:1000]
    initial_offset = val_data[1] - val_data[0]

    for episode in range(0, ep_count + 1):
        train_result = train_model(agent, episode, train_data, ep_count=ep_count,
                                   batch_size=batch_size, window_size=window_size)
        val_result = evaluate_model(agent, val_data, window_size, debug)

        plt.figure(figsize=(12, 8))
        plt.plot(train_result[4], label="WTI")
        plt.title(f"Episode {episode+1}: WTI Training")
        plt.savefig(f'visuals/WTI/episode{episode+1}_WTI.png')
        # plt.show()

        train_mlpd, train_md, train_std, train_profit = interpret_results(train_result[4])

        plt.plot(val_result[2], label="Brent")
        plt.title(f"Episode {episode+1}: Brent Validation")
        plt.legend()
        plt.savefig(f'visuals/Brent/episode{episode + 1}_Brent.png')
        # plt.show()

        val_mlpd, val_md, val_std, val_profit = interpret_results(val_result[2])
        show_train_result(train_result, val_result[0], initial_offset, train_mlpd, train_md, train_std, val_mlpd, val_md, val_std)


if __name__ == "__main__":
    train_stock = "../../pricedata.csv"
    val_stock = "../../pricedata.csv"
    window_size = 5
    batch_size = 10
    ep_count = 9
    strategy = "t-dqn"
    model_name = "rl1"

    try:
        main(train_stock, val_stock, window_size, batch_size,
             ep_count, strategy=strategy, model_name=model_name, pretrained = False,
             debug=False, noep=5)
        b = get_benchmarks(get_stock_data(train_stock))
        print("Benchmark (Buy-Hold) Results:")
        print(f"Training set (WTI Oil): Profit: {b[0][3]}; Max Loss Per Day: {b[0][0]}; Max Drawdown: {b[0][1]}; Std (Volatility): {b[0][2]}")
        print(f"Validation set (Brent Oil): Profit: {b[1][3]}; Max Loss Per Day: {b[1][0]}; Max Drawdown: {b[1][1]}; Std (Volatility): {b[1][2]}")
    except KeyboardInterrupt:
        print("Aborted!")

    id2comm = {0: "WTI", 1: "Brent", 2: "NatGas", 3: "Heating"}
    for i in range(4):
        arr = pd.Series(get_stock_data(train_stock)[i][:1000])
        plt.figure(figsize=(12, 8))
        plt.plot(arr)
        plt.title(f"{id2comm[i]} Price")
        plt.savefig(f'visuals/{id2comm[i]}/{id2comm[i]}_price_fig.png')
        # plt.show()

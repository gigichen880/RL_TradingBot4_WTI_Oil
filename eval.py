import os
import coloredlogs
from trading_bot.agent import Agent
from trading_bot.methods import evaluate_model
from trading_bot.utils import (
    get_stock_data,
    show_eval_result,
    interpret_results,
    get_benchmarks
)


def main(eval_stock, window_size, model_name, debug, noep=5, id = 1):
    data = get_stock_data(eval_stock)[id]
    initial_offset = data[1] - data[0]
    id2comm = {0: "WTI", 1: "Brent Oil", 2: "Natural Gas", 3: "Heating Oil"}

    # Single Model Evaluation
    if model_name is not None:
        agent = Agent(window_size, pretrained = True, model_name=model_name, noep=noep)
        profit, _, test_profit_li = evaluate_model(agent, data, window_size, debug)
        test_mlpd, test_md, test_std= interpret_results(test_profit_li)
        print(f"Target Commodity is {id2comm[id]}")
        show_eval_result(model_name, profit, initial_offset, test_mlpd, test_md, test_std)
        
    # Multiple Model Evaluation
    else:
        for model in os.listdir("models"):
            if os.path.isfile(os.path.join("models", model)):
                agent = Agent(window_size, pretrained = True, model_name=model, noep=noep)
                profit, _, test_profit_li = evaluate_model(agent, data, window_size, debug)
                test_mlpd, test_md, test_std, test_profit = interpret_results(test_profit_li)
                show_eval_result(model, profit, initial_offset, test_mlpd, test_md, test_std)
                del agent


if __name__ == "__main__":

    eval_stock = "../../pricedata.csv"
    window_size = 5
    batch_size = 10
    ep_count = 2
    strategy = "t-dqn"
    model_name = "rl1"
    coloredlogs.install(level="DEBUG")
    ids = [2, 3]
    noeps = [0, 1, 2]
    for noep in noeps:
        for id in ids:
            try:
                main(eval_stock, window_size, model_name, debug=False, noep=noep, id=id)
            except KeyboardInterrupt:
                print("Aborted")
    b = get_benchmarks()
    print("Benchmark (Buy-Hold) Results:")
    print(f"Test set1 (Natural Gas): Profit: {b[2][3]}; Max Loss Per Day: {b[2][0]}; Max Drawdown: {b[2][1]}; Std (Volatility): {b[2][2]}")
    print(f"Test set2 (Heating Oil): Profit: {b[3][3]}; Max Loss Per Day: {b[3][0]}; Max Drawdown: {b[3][1]}; Std (Volatility): {b[3][2]}")
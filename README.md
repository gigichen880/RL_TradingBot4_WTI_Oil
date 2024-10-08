# Reinforcement Learning (DDQN)-Based Trading Bot

Before Running, please make sure the relative path to pricedata.csv in either train.py or eval.py is corrected to fit your environment path

Here a trading bot of reinforcement learning trained by Double Deep Q Learning is assembled on WTI Oil Future Dataset, with a 73.22% decrease in total loss on average. Applying the same agent to the Brent Crude Oil Future results in a negative-to-positive return (-48.019 to +1.045), compared to the Double Moving Average (DMA) strategy as the benchmark. Check visuals2 for visualized results and log.txt for text result loggings on both training and validation sets; notice the improvement of RL agent in the trading task. 

## Important Elements
1. State: array of daily returns (after a sigmoid mapping) of previous n days
2. Action: Buy, Sell, Hold
3. Agent: trading bot taking actions and receiving feedbacks on a daily basis;
4. Environment: market of natural resource commodities, with fluctuated pricing
5. Reward: profit made by selling 1 unit compared to the lastest 1-unit purchase
6. Loss Function: Huber Loss (Custom for Q-Learning)


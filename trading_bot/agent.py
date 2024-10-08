import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def huber_loss(y_true, y_pred, clip_delta=torch.tensor(1.0)):
    # Huber loss - Custom Loss Function for Q Learning
    error = y_true - y_pred
    cond = torch.abs(error) <= clip_delta
    squared_loss = 0.5 * torch.square(error)
    quadratic_loss = 0.5 * torch.square(clip_delta) + clip_delta * (torch.abs(error) - clip_delta)
    return torch.mean(torch.where(cond, squared_loss, quadratic_loss))

class NN(nn.Module):
    def __init__(self, state_size, action_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(in_features=state_size, out_features=128)
        self.relu =nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=128)
        self.fc5 = nn.Linear(in_features=128, out_features=action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class Agent:
    def __init__(self, state_size, strategy="t-dqn", reset_every=1000, pretrained = False, model_name=None, noep = 5):
        self.strategy = strategy

        # agent config
        self.state_size = state_size    	# normalized previous days
        self.action_size = 3           		# [sit, buy, sell]
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=10000)
        self.first_iter = True

        # model config
        self.model_name = model_name
        self.gamma = 0.95 # affinity for long term reward
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.noep = noep
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory

        if pretrained and self.model_name is not None:
            self.model = self.load(self.state_size, self.action_size, self.noep)
        else:
            self.model = NN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # strategy config
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter = 1
            self.reset_every = reset_every

            # target network
            self.target_model = self.model
            self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Adds relevant data to memory
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_eval=False):
        """Take action from given possible set of actions
        """
        state = state.astype(np.float32)
        state = torch.from_numpy(state)

        # take random action in order to diversify experience at the beginning
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            return 1 # make a definite buy on the first iter

        action_probs = self.model(state)
        return torch.argmax(action_probs[0])

    def train_experience_replay(self, batch_size):
        """Train on previous experiences in memory
        """
        mini_batch = random.sample(self.memory, batch_size)
        y_pred, y_true = [], []
        
        # DQN
        # if self.strategy == "dqn":
        #     for state, action, reward, next_state, done in mini_batch:
        #         state = torch.from_numpy(state)
        #         next_state = torch.from_numpy(next_state)
        #         if done:
        #             target = reward
        #         else:
        #             # approximate deep q-learning equation
        #             target = reward + self.gamma * np.amax(self.model(next_state)[0])
        #
        #         # estimate q-values based on current state
        #         q_values = self.model(state)
        #         y_pred.append(q_values[0])
        #         # update the target for current action based on discounted reward
        #         q_values[0][action] = target
        #         y_true.append(q_values[0])

        # DQN with fixed targets
        if self.strategy == "t-dqn":

            if self.n_iter % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())

            for state, action, reward, next_state, done in mini_batch:

                state = state.astype(np.float32)
                state = torch.from_numpy(state)
                next_state = next_state.astype(np.float32)
                next_state = torch.from_numpy(next_state)
                if done:
                    target = reward
                else:
                    # approximate deep q-learning equation with fixed targets
                    target = reward + self.gamma * torch.amax(self.target_model(next_state)[0][0])

                # estimate q-values based on current state
                q_values = self.model(state)
                y_pred.append(q_values[0])
                # update the target for current action based on discounted reward
                q_copy = q_values.detach().clone()
                q_copy[0][action] = target
                y_true.append(q_copy[0])

        # Double DQN
        # elif self.strategy == "double-dqn":
        #     if self.n_iter % self.reset_every == 0:
        #         # reset target model weights
        #         self.target_model.set_weights(self.model.get_weights())
        #
        #     for state, action, reward, next_state, done in mini_batch:
        #         state = torch.from_numpy(state)
        #         next_state = torch.from_numpy(next_state)
        #         if done:
        #             target = reward
        #         else:
        #             # approximate double deep q-learning equation
        #             target = reward + self.gamma * self.target_model(next_state)[0][np.argmax(self.model(next_state)[0])]
        #
        #         # estimate q-values based on current state
        #         q_values = self.model(state)
        #
        #         y_pred.append(q_values[0])
        #
        #         # update the target for current action based on discounted reward
        #         q_values[0][action] = target
        #         y_true.append(q_values[0])
                
        else:
            raise NotImplementedError()

        # update q-function parameters based on huber loss gradient
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        loss = huber_loss(y_pred, y_true)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # as the training goes on we want the agent to
        # make less random and more optimal decisions
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def save(self, episode):
        torch.save(self.model.state_dict(), "models/{}_{}".format(self.model_name, episode+1))

    def load(self, state_size, action_size, noep):
        model = NN(state_size, action_size)
        model.load_state_dict(torch.load("models/" + self.model_name + "_" + str(noep), weights_only=True))
        return model
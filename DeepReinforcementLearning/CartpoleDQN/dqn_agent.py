import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(
            maxlen=2000
        )  # double-ended queue; acts like list, but elements can be added/removed from either end
        self.gamma = 0.95  # decay or discount rate: enables agent to take into account future actions in addition to
        # the immediate ones, but discounted at this rate
        self.epsilon = 1.0  # exploration rate: how much to act randomly; more initially than later due to epsilon decay
        self.epsilon_decay = 0.995  # decrease number of random explorations as the agent's performance (hopefully)
        # improves over time
        self.epsilon_min = 0.01  # minimum amount of random exploration permitted
        self.learning_rate = (
            0.001  # rate at which NN adjusts models parameters via SGD to reduce cost
        )
        self.model = self._build_model()  # private method

    def _build_model(self):
        # neural net to approximate Q-value function:
        model = Sequential()
        model.add(
            Dense(32, activation="relu", input_dim=self.state_size)
        )  # 1st hidden layer; states as input
        model.add(Dense(32, activation="relu"))  # 2nd hidden layer
        model.add(
            Dense(self.action_size, activation="linear")
        )  # 2 actions, so 2 output neurons: 0 and 1 (L/R)
        # model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )  # list of previous experiences, enabling re-training later

    def train(
        self, batch_size
    ):  # method that trains NN with experiences sampled from memory
        minibatch = random.sample(
            self.memory, batch_size
        )  # sample a minibatch from memory
        for (
            state,
            action,
            reward,
            next_state,
            done,
        ) in minibatch:  # extract data for each minibatch sample
            target = reward  # if done (boolean whether game ended or not, i.e., whether final state or not),
            # then target = reward
            if not done:  # if not done, then predict future discounted reward
                target = (
                    reward
                    + self.gamma
                    * np.amax(  # (target) = reward + (discount rate gamma) *
                        self.model.predict(next_state)[0]
                    )
                )  # (maximum target Q based on future action a')
            target_f = self.model.predict(
                state
            )  # approximately map current state to future discounted reward
            target_f[0][action] = target
            self.model.fit(
                state, target_f, epochs=1, verbose=0
            )  # single epoch of training with x=state, y=target_f; fit decreases loss btwn target_f and y_hat
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:  # if acting randomly, take random action
            return random.randrange(self.action_size)
        act_values = self.model.predict(
            state
        )  # if not acting randomly, predict reward value based on current state
        return np.argmax(
            act_values[0]
        )  # pick the action that will give the highest reward (i.e., go left or right?)

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)


def main():
    print("Я очень люблю тебя, Наталия")
    state_size = 4
    action_size = 2
    agent = DQNAgent(state_size, action_size)  # initialise agent


if __name__ == "__main__":
    main()

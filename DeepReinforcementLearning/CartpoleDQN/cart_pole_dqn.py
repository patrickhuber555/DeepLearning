import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os  # for creating directories
from settings_cart_pole_dqn import OUTPUT_DIR, N_EPISODES, BATCH_SIZE
import dqn_agent


def get_parameters():
    # env = gym.make("CartPole-v0")  # initialise environment
    env = gym.make("CartPole-v1")  # initialise environment
    state_size = env.observation_space.shape[0]
    print(f"State size: {state_size}")
    action_size = env.action_space.n
    print(f"Action size: {action_size}")
    return (env, state_size, action_size)


def create_directory():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def interact_with_environement(agent, env, state_size):
    for e in range(N_EPISODES):  # iterate over episodes of gameplay

        state = env.reset()  # reset state at start of each new episode of the game
        state = np.reshape(state, [1, state_size])

        done = False
        time = 0  # time represents a frame of the episode; goal is to keep pole upright as long as possible
        while not done:
            #         env.render()
            action = agent.act(
                state
            )  # action is either 0 or 1 (move cart left or right); decide on one or other here
            next_state, reward, done, _ = env.step(
                action
            )  # agent interacts with env, gets feedback; 4 state data points, e.g., pole angle, cart position
            reward = (
                reward if not done else -10
            )  # reward +1 for each additional frame with pole upright
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(
                state, action, reward, next_state, done
            )  # remember the previous timestep's state, actions, reward, etc.
            state = next_state  # set "current state" for upcoming iteration to the current next state
            if done:  # if episode ends:
                print(
                    "episode: {}/{}, score: {}, e: {:.2}".format(  # print the episode's score and agent's epsilon
                        e, N_EPISODES - 1, time, agent.epsilon
                    )
                )
            time += 1
        if len(agent.memory) > BATCH_SIZE:
            agent.train(
                BATCH_SIZE
            )  # train the agent by replaying the experiences of the episode
        if e % 50 == 0:
            agent.save(OUTPUT_DIR + "weights_" + "{:04d}".format(e) + ".hdf5")


def main():
    print("Я очень люблю тебя, Наталия")
    env, state_size, action_size = get_parameters()
    create_directory()
    agent = dqn_agent.DQNAgent(state_size, action_size)  # initialise agent
    interact_with_environement(agent, env, state_size)


if __name__ == "__main__":
    main()

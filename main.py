'''
TODO:
    1. Generate multiple epoisdes
    2. At each timestep in a episode, store the sample <s, a, r, s',done> and save it in memory. Then, take a random batch
    of batch size from the memory and train the network.
    3. Take action according to the ε-greedy policy and go the next state.
    4. Update the current Q-value network after every timestep in a episode.
    5. Update the target Q-value network to current Q-value network after training for a episode. This means that weights an
    biases of target Q-value network will become same as current Q-value network.

Note: Penalty of -100 is added if an action make the episode end.

'''
import pylab
import numpy as np
import pandas as pd
from agent import DQNAgent
from env import POEnv
import os

def load_data():
    df = pd.read_csv("data.csv")
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.value_counts())
    return df

def transform_data(data):
    df_lb = data.loc[data['Name'] == 'CLX']
    df_hb = data.loc[data['Name'] == 'TSLA']

    df = df_lb.set_index('Date').join(df_hb.set_index('Date'), lsuffix='_lb_stock', rsuffix='_hb_stock')
    return df


"""
WMA = ( P1 * W1 ) + (P2 *W2) + (P3 *W3) + (Pn * Wn) / (Wn + Wn-1 +…)
Where:
P1 = current price
P2 = Price one period ago
Wn = The period 2days of period stock history for both the HB and LB stock

# take 1 stock portfolio LB : CLX, HB: TSLA
"""

def make_dir():

    directory_model = 'save_model'
    directory_graph = 'save_graph'

    # directory = 'save_graph'
    if not os.path.exists(directory_model):
        os.makedirs(directory_model)

    if not os.path.exists(directory_graph):
        os.makedirs(directory_graph)

def save_model_graph(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("./save_model/PO_dqn_model.json", "w") as json_file:
            json_file.write(model_json)


def main():
    if __name__ == "__main__":
        EPISODES = 2
#get the data
        data = load_data()
        data = transform_data(data)
#get the env

        env = POEnv()
        # get size of state and action from environment
        #state_size = env.state_space.shape[0]
        state_size = len(env.state_space)
        #action_size = env.action_space.n
        action_size = len(env.action_space)
        print("State Size : " , state_size)
        print("Action Size : ", action_size)
        agent = DQNAgent(state_size, action_size)

        rewards, episodes = [], []
        max_len = len(data)
        for e in range(EPISODES):
            done = False
            reward = 0

            state = env.set_init_state(data = data.head(1))

            state = np.reshape(state, [1, state_size])
            print("start episode -------", e)
            print("current_state :", state)
            #use 7 days window for histrical prices
            window = 7
            i = 0
            while not done:
            # get action for the current state and go one step in environment
                action = agent.get_action(state)
                print("action : ", action)
                print("start step -------")
                print("current index", i, "window", window, "max len ", max_len)
                next_state , reward = env.get_next_state(state, action, data[i:window])
                print("end step -------")
                i = window
                window = window + 7
                next_state = np.reshape(next_state, [1, state_size])

                  # save the sample <s, a, r, s'> to the replay memory
                agent.append_sample(state, action, reward, next_state, done)

            # every time step do the training
                agent.train_model()
                state = next_state
            # stop training
                if window > max_len:
                    done = True
                    print("end episode @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", e)
                if done:
                # every episode update the target model to be same with model
                    print("update target model")
                    agent.update_target_model()

                # adding +100K to reward because initially we subtracted 100K as a penalty

                    rewards.append(reward + env.penalty)
                    episodes.append(e)
                    print("plot graph")
                    pylab.plot(episodes, rewards, 'b')
                    print("save graph")
                    pylab.savefig("./save_graph/PO_dqn.png")
                    print("episode:", e, "  profit:", reward, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)
                    print("wrapup episode")



        # save the model
        if e % 50 == 0:
            print("save model")
            agent.model.save_weights("./save_model/PO_dqn.h5")
#""
make_dir()
main()

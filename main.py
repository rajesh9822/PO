
import numpy as np
from agent import DQNAgent
from env import POEnv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from data import load_data, transform_data, do_eda
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

def main(data):
    if __name__ == "__main__":
        EPISODES = 100

# get the env
        env = POEnv()
    # get size of state and action from environment
        # state_size = env.state_space.shape[0]
        state_size = len(env.state_space)
        # action_size = env.action_space.n
        action_size = len(env.action_space)
        print("State Size : " , state_size)
        print("Action Size : ", action_size)
        agent = DQNAgent(state_size, action_size)
        rewards, episodes, do_nothing_rewards = [], [], []
        max_len = len(data)

        for e in range(EPISODES):
            done = False
            reward = 0
            state = env.set_init_state(data)
            state = np.reshape(state, [1, state_size])

            print("start episode -------", e)
            print("current_state :", state)
            # use 7 days window for historical prices
            i = 0
            window = env.period
            while not done:
            # get action for the current state and go one step in environment
                action = agent.get_action(state)
                print("action : ", action)
            #               print("start step -------")
                print("current index", i, "window", window, "max len ", max_len)
                next_state, reward, done = env.get_next_state(state, action, data[i:window])
            #               print("end step -------")
                i = window
                print("period:", env.period)
                window = window + env.period
                next_state = np.reshape(next_state, [1, state_size])

                # save the sample <s, a, r, s'> to the replay memory
                agent.append_sample(state, action, reward, next_state, done)

            # every time step do the training
                agent.train_model()
                state = next_state
            # stop training
                if window > max_len:
                    done = True
                #print("end episode @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", e)
                if done:
            #  every episode update the target model to be same with model
            #  print("update target model")
                    agent.update_target_model()

                # adding +100K to reward because initially we subtracted 100K as a penalty
           #      state = (WMA, no_of_lb_stock, no_of_hb_stock, total_portfolio_amt, cash),
                    #do_nothing_reward = (state_dono[1] * data.Close_lb_stock[i_dono] + state_dono[2] * data.Close_hb_stock[i_dono] ) - state_dono[4]
                    rewards.append(reward + reward * env.penalty)
                    ##do_nothing_rewards.append(do_nothing_reward)
                    episodes.append(e)
#                   print("plot graph")
                    #pylab.plot(episodes, rewards, 'b').title("Profit accross Epocs")
                    sns.lineplot(x=episodes,y=rewards, dashes=False, ).set(title='Profit accross Epocs')
                    #sns.lineplot(x=episodes, y=do_nothing_rewards, dashes=False, ).set(title='Profit accross Epocs')
                    plt.savefig("./save_graph/PO_dqn.png")
#                   print("save graph")
                    #pylab.savefig("./save_graph/PO_dqn.png")
                    #pylab.plot(episodes, do_nothing_rewards, 'b').title("Profit accross Epocs")
            #                  print("save graph")
                    #pylab.savefig("./save_graph/PO_dqn_dono.png")
                    ##print("episode:", e, "  profit:", reward, " cash:", env.total_cash,
                    ##"do nothing profit:", do_nothing_reward ,"memory length:",
                    ##len(agent.memory), "  epsilon:", agent.epsilon)
 #                  print("wrapup episode")
        # save the model
        if e % 50 == 0:
        # print("save model")
            agent.model.save_weights("./save_model/PO_dqn.h5")


#Test your Algorithm
#make directories
make_dir()
#get the data
data = load_data()
#select the data
data = transform_data(data)
#create the datafrane
#do_eda(data)
#run the episodes
main(data)
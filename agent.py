'''
If you are using this framework, you need to fill the following to complete the following code block:

State and Action Size

state s = (WMA for [2, 7, 30] days, VPT for [2, 7, 30] days, no of low beta stock, no of high beta stock, total portfolio amt , cash left)
action at ∈ [−0.25,−0.1,0.05,0,0.05,0.1,0.25]

Hyperparameters
Create a neural-network model in function 'build_model()'
Define epsilon-greedy strategy in function 'get_action()'
Complete the function 'append_sample()'. This function appends the recent experience tuple <state, action, reward, new-state> to the memory
Complete the 'train_model()' function with following logic:
If the memory size is greater than mini-batch size, you randomly sample experiences from memory as per the mini-batch size and do the following:
Initialise your input and output batch for training the model
Calculate the target Q value for each sample: reward + gamma*max(Q(s'a,))
Get Q(s', a) values from the last trained model
Update the input batch as your encoded state and output batch as your Q-values
Then fit your DQN model using the updated input and output batch.
[ ]


'''

import numpy as np
import random
import collections
from collections import deque
# for building the DQN model
from keras import layers
from keras import Sequential
from keras.layers import Dense
from keras.optimizer_v1 import adam

from env import POEnv


class DQNAgent:

    def __init__(self, state_size, action_size):

        self.load_model = False

        # Define size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyperparameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

        self.batch_size = 32
        self.train_start = 500

        # create replay memory using deque
        self.memory = deque(maxlen=1000)
        #self.env = POEnv()
        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        # self.save_model_graph()

        if self.load_model:
            self.model.load_weights("./save_model/PO_dqn.h5")
            self.epsilon = 0.0

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        '''
        TODO:
        Build multilayer perceptron to train the Q(s,a) function. In this neural network, the input will be states and the output
        will be Q(s,a) for each (state,action).
        Note: Since the ouput Q(s,a) is not restricted from 0 to 1, we use 'linear activation' as output layer.

        Loss Function:
        Loss=1/2 * (R_t + γ∗max Q_t (S_{t+1},a)−Q_t(S_t,a)^2
               which is 'mean squared error'

        '''

        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer='adam')
        return model

    def save_model_graph(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("./save_model/PO_dqn_model.json", "w") as json_file:
            json_file.write(model_json)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        '''
        TODO:
        Update the target Q-value network to current Q-value network after training for a episode. This means that weights an
        biases of target Q-value network will become same as current Q-value network.
        '''
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        '''
        Select action
        Args:
            state: At any given state, choose action

        TODO:
        Choose action according to ε-greedy policy. We generate a random number over [0, 1) from uniform distribution.
        If the generated number is less than ε, we will explore, otherwise we will exploit the policy by choosing the
        action which has maximum Q-value.

        More the ε value, more will be exploration and less exploitation.

        '''
        # choose random action if generated random number is less than ε.
        # Action is represented by index, 0-Number of actions, like (0,1,2,3) for 4 actions
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # if generated random number is greater than ε, choose the action which has max Q-value
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        '''
        Save sample in memory and decay ε after we generate each sample from environment.

        Args:
            (state, action, reward, next_state, done)- <s,a,r,s',done>

        TODO:
            We are saving each sample  (state, action, reward, next_state, done) of the episode, in a memory. Memory can be
            defined by queue. We will dequeue sample of batch size from the memory and use it to train the neural network.

            ε-decay:
            With ε, we explore and with 1-ε, we exploit. Initially we want to explore more, but at later point, after training
            the model, we have good policy to choose better action. So, at that point, we want to expoit more and explore less.
            So, we want to decrease the value of ε, by which we explore.

            self.epsilon_min:
            Minimum value of ε, by which we want to explore. If the current value of ε is greater then
            minimum value to ε, we will decay ε gradually, when generating samples.

            Note: The rate by which we will decrease ε should be slow, otherwise we will not explore much and instead settle
            for suboptimal policy instead of optiomal policy.

        '''
        # Adding sample to the memory.
        self.memory.append((state, action, reward, next_state, done))

        # Decay in ε after we generate each sample from the environment
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size) and train the network
    def train_model(self):
        '''
        Train the neural network to find the best policy

        TODO:
        1. Sample <s,a,r,s',done> of batch size from the memory
        2. Set the target as R_t + γ∗max Q_t(S_{t+1},a)−Q_t(S_t,a)
        3. Set the target only for the action we took in the environment. For the other actions, we don't wan't to
        update the network.
        4. Remember that we already the actions that we took when generating sample from environment
        4. To find the Q_t(S_{t+1},a), we input the next state s' to the model, and we get Q-value for all the actions
        5. To find the Q_t(S_t,a), we input the current state s to the model, and we get Q-value for all the actions
        6. Train the model

        Note:
        We use 2 different neural network for Q_t(S_t,a) and target Q_t(S_{t+1},a). This is so because we are
        constantly updating the current Q-value network at each and every timestep in a episode. Therefore, the target
        Q-value will change subsequently. The network can become destabilized by falling into feedback loops between the
        target and current Q-values.
        We update the target Q-value network only after completion of a batch. We update the target Q-value with the
        current Q-value network.

        '''
        # We start the training only when we have sufficient sample in the memory. We set the number of samples required
        # start training in variable train_start
        if len(self.memory) < self.train_start:
            return

        # Sample batch from the memory
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        # Initialise the variables update_input and update_target for a batch for storing the s and s'.
        # Later, we will use it to store Q_t(S_t,a_t) and Q_t(S_{t+1},a)
        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        # Set the values of input, action, reward, target and done using memory
        # Note the order of <s,a,r,s',done>
        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        # Set the target as Q values predicted from the current state and next state
        # store Q_t(S_t,a_t) and Q_t(S_{t+1},a) in target and target_val
        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        # Update the target value according to the update policy of Q-learning
        # R_t + γ ∗ max Q_t(S_{t+1},a)−Q_t(S_t,a_t)
        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

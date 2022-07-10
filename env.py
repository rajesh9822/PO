import numpy as np
import math
import random
import collections
from collections import deque
# for building the DQN model
import pandas as pd
from keras import layers
from keras import Sequential
from keras.layers import Dense
#from keras.optimizers import Adam
from datetime import datetime
from itertools import product


class POEnv:
    def __init__(self, total_amt=5, fees=0.01, reward=9):
        self.hyperparameters = self.initialize_hyperparameters(total_amt, fees, reward)
        self.action_space = self.initialize_action_space()
        self.state_space = self.initialize_state_space()
        #self.state_init = self.set_init_state(data)
        #self.reset_state()

    ## Initialize environment hyperparameters, total action space and total state space

    def initialize_hyperparameters(self, total_amt, fees, reward):

            self.total_amt = 1000000, #intial cash
            self.fees = 0.01,  # per transaction fees
            self.reward = 0  # profit

            return {self.total_amt, self.fees, self.reward}

    def initialize_action_space(self):

        """ An action is represented by a tuple (buy_sell_percent).
        Depending on the current state of the portfolio stock is represented by

        agent will select the most appropriate action which maximizes reward, buy or sell, high or low beta stock

        at ∈[−0.25, −0.1, 0.05, 0, 0.05, 0.1, 0.25]
        """
        self.action_space = [-0.25, -0.1, 0.05, 0, 0.05, 0.1, 0.25]

        return self.action_space

    def initialize_state_space(self):
        """ Current state of  env is represented by
        state = (WMA, no_of_lb_stock, no_of_hb_stock, total_portfolio_amt, cash),
        """
        self.wma_lb_stock = 0
        self.wma_hb_stock = 0
        self.no_of_lb_stock = 0
        self.no_of_hb_stock = 0
        self.total_portfolio_amt = 0
        self.total_cash = 0
        self.state_space = [self.wma_lb_stock, self.wma_hb_stock, self.no_of_lb_stock, self.no_of_hb_stock, self.total_portfolio_amt, self.total_cash]
        return self.state_space

    ## Set / Reset initial portfolio state

    def set_init_state(self, data):
        """ Select the first date from which the portfolio will start.
        """
       # state = (WMA, VPT, no_of_lb_stock, no_of_hb_stock, total_portfolio_amt, cash),
        for i in range(len(data)):
             self.wma_lb_stock = self.wma_lb_stock + data.Close_lb_stock[i] * i
             self.wma_hb_stock = self.wma_hb_stock + data.Close_hb_stock[i] * i
            # we will start with 50-50 allocation

        self.total_portfolio_amt = 1000000
        self.no_of_lb_stock  = (self.total_portfolio_amt  * .5 )/ data.Close_lb_stock[i]
        self.no_of_hb_stock = (self.total_portfolio_amt * .5) / data.Close_hb_stock[i]
        self.total_cash = 1000000
        self.state = [self.wma_lb_stock, self.wma_hb_stock, self.no_of_lb_stock, self.no_of_hb_stock, self.total_portfolio_amt, self.total_cash]
        return self.state

    def get_next_state(self, state, action, data):
        #Calculate next state, reward and total ride time for a given
        #    state and action
        self.done = False
        window = 7
        for i in range(len(data)):
            self.wma_lb_stock = self.wma_lb_stock + data.Close_lb_stock[i] * i
            self.wma_hb_stock = self.wma_hb_stock + data.Close_hb_stock[i] * i
            print("data.Close_lb_stock[i] ", data.Close_lb_stock[i],i)
            # we will start with 50-50 allocation
        window = window + 7
        print("self.wma_lb_stock ", self.wma_lb_stock, i)
        print("self.wma_hb_stock ", self.wma_hb_stock, i)
        self.total_portfolio_amt_cur = state[:4]
        buy_sel_per = self.action_space[action]
        print(buy_sel_per)

        if buy_sel_per < 0:
            #sell lb and buy corresponding hb stock
                    #Calculate no shares to buy and Sell
                    no_of_lb_stock_to_sell =  self.no_of_lb_stock * buy_sel_per

                    no_of_hb_stock_to_buy = self.no_of_hb_stock * buy_sel_per * -1

                    #Calculate total no of shares after agent action
                    self.no_of_lb_stock = self.no_of_lb_stock + no_of_lb_stock_to_sell
                    self.no_of_hb_stock = self.no_of_hb_stock + no_of_hb_stock_to_buy

                    #calculate cost of buy and sale based on the action

                    cost_of_lb_stock_to_sell = no_of_lb_stock_to_sell * data.Close_lb_stock[i] + 0.01 * no_of_lb_stock_to_sell
                    cost_of_hb_stock_to_buy  = no_of_hb_stock_to_buy * data.Close_hb_stock[i] + 0.01 * no_of_hb_stock_to_buy

                    if abs(cost_of_lb_stock_to_sell) - abs(cost_of_hb_stock_to_buy) > 0:

                        self.total_cash = self.total_cash + (abs(cost_of_lb_stock_to_sell) - abs(cost_of_hb_stock_to_buy))
                    else:
                        self.total_cash = self.total_cash + (abs(cost_of_lb_stock_to_sell) - abs(cost_of_hb_stock_to_buy))

        else:
            # sell lb and buy corresponding hb stock
            # Calculate no shares to buy and Sell
                    no_of_lb_stock_to_buy = self.no_of_lb_stock * buy_sel_per

                    no_of_hb_stock_to_sell = self.no_of_hb_stock * buy_sel_per * -1

                    # Calculate total no of shares after agent action
                    self.no_of_lb_stock = self.no_of_lb_stock + no_of_lb_stock_to_buy
                    self.no_of_hb_stock = self.no_of_hb_stock + no_of_hb_stock_to_sell

                    # calculate cost of buy and sale based on the action

                    cost_of_lb_stock_to_buy = no_of_lb_stock_to_buy * data.Close_lb_stock[i] + .01 * no_of_lb_stock_to_buy
                    cost_of_hb_stock_to_sell = no_of_hb_stock_to_sell * data.Close_hb_stock[i] + .01 * no_of_hb_stock_to_sell

                    if abs(cost_of_hb_stock_to_sell) - abs(cost_of_lb_stock_to_buy) > 0:
                        self.total_cash = self.total_cash + (abs(cost_of_hb_stock_to_sell) - abs(cost_of_lb_stock_to_buy))
                    else:
                        self.total_cash = self.total_cash - (abs(cost_of_hb_stock_to_sell) - abs(cost_of_lb_stock_to_buy))

        self.total_portfolio_amt_nxt = self.no_of_lb_stock * data.Close_lb_stock[i] + self.no_of_hb_stock * data.Close_hb_stock[i]
        print("data.Close_lb_stock[i]",data.Close_lb_stock[i])
        print("self.total_portfolio_amt_nxt", self.total_portfolio_amt_nxt)
        print("self.total_cash",self.total_cash)
        self.next_state = [self.wma_lb_stock, self.wma_hb_stock, self.no_of_lb_stock, self.no_of_hb_stock,
                                      self.total_portfolio_amt_nxt, self.total_cash]
        print(self.next_state)


        return self.next_state, self.reward, self.done
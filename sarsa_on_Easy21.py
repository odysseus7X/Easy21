import numpy as np
import random
import MC_control as env
import matplotlib.pyplot as plt

class TD_learning():
    def __init__(self, episodes, counter, lamda, Q_optimal):
        super().__init__()
        self.episodes = episodes
        self.N_o = counter
        self.lamda = lamda
        self.Q_optimal = Q_optimal

    def epsilon_greedy_policy(self, N_state, Q_state):

        rand_numb = random.random()
        epsilon = self.N_o/(self.N_o + N_state)
        if rand_numb <= epsilon:
            action_1 = random.randint(0,1)
        else:
            action_1 = np.argmax(Q_state)

        return action_1
    
    def Mean_squared_error(self, Q):
        mse = np.mean((Q - self.Q_optimal) ** 2)
        return mse


    def train(self):
        Q = np.zeros((11,22,2))
        E = np.zeros((11,22,2))
        N = np.zeros((11,22,2))
        N_s= np.zeros((11,22))
        environment = env.Easy21_env()
        MSE_list = []

        for ep in range(self.episodes):
            player_sum1 = random.randint(1, 10)
            dealer_card_1 = random.randint(1, 10)
            terminated = False
            state = (dealer_card_1, player_sum1, terminated)
            action = self.epsilon_greedy_policy(N_s[dealer_card_1, player_sum1], Q[dealer_card_1, player_sum1])
            N[dealer_card_1, player_sum1, action]+=1
            N_s[dealer_card_1, player_sum1]+=1
            E[:]=0

            while terminated == False:

                E = E * self.lamda
                E[dealer_card_1, player_sum1, action] += 1

                Q_value = Q[dealer_card_1, player_sum1, action]
                N_value = N[dealer_card_1, player_sum1, action]

                Next_state, Reward = environment.step(state, action)
                dealer_card_1, player_sum1, terminated = Next_state

                if terminated == False:
                    Next_action = self.epsilon_greedy_policy(N_s[dealer_card_1, player_sum1], Q[dealer_card_1, player_sum1])
                    
                    N[dealer_card_1, player_sum1, Next_action]+=1
                    N_s[dealer_card_1, player_sum1]+=1
                    
                    delta = Reward + Q[dealer_card_1, player_sum1, Next_action] - Q_value
                    Q += E * delta/N_value

                else:
                    Next_action=0
                    delta = Reward + 0 - Q_value
                    Q += E * delta/N_value

                action = Next_action
                state = Next_state

                
            if self.lamda==0 or self.lamda == 1:
                MSE = self.Mean_squared_error(Q)
                MSE_list.append(MSE)
                
        MSE = self.Mean_squared_error(Q)
        print("Done for:"," ", self.lamda," ", "mse:", MSE)

        return MSE_list, MSE

agent = env.Monte_carlo_control(1000000, 100)
Q_MC = agent.train()

if __name__ == "__main__":
    list_l = np.arange(0, 1.001, 0.1)
    ep_list = np.arange(0, 1000000, 1)
    MSE_list_x =[]
    for l in list_l:
        agent_TD = TD_learning(1000000, 100, l, Q_MC)
        MSE_list_, MSE1 = agent_TD.train()
        MSE_list_x.append(MSE1)
        if l==0:
            MSE_list_0 = MSE_list_
        elif l==1:
            MSE_list_1 = MSE_list_

    plt.figure()
    plt.plot(list_l, MSE_list_x)
    plt.title('mean-squared error for various lambdas')
    
    plt.figure()
    plt.plot(ep_list, MSE_list_0)
    plt.xlabel('episodes')
    plt.title('mean-squared error for lambda = 0')
    
    plt.figure()
    plt.plot(ep_list, MSE_list_1)
    plt.xlabel('episodes')
    plt.title('mean-squared error for lambda = 1')
    
    plt.show()

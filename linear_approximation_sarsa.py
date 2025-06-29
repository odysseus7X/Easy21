import numpy as np
import random
import MC_control as env
import matplotlib.pyplot as plt

class linear_approximation_sarsa():
    def __init__(self, episodes, lamda, Q_optimal, gamma):
        super().__init__()
        self.episodes = episodes
        self.lamda = lamda
        self.Q_optimal = Q_optimal
        self.feature_vector = np.zeros((36))
        self.dc_feature = np.zeros((3))
        self.ps_feature = np.zeros((6))
        self.a_feature = np.zeros((2))
        self.alpha = 0.05
        self.epsilon = 0.05
        self.gamma = gamma

    def epsilon_greedy_policy(self, Q_state):

        rand_numb = random.random()
        epsilon = self.epsilon
        if rand_numb <= epsilon:
            action_1 = random.randint(0,1)
        else:
            action_1 = np.argmax(Q_state)

        return action_1
    
    def Mean_squared_error(self, Q):
        mse = np.mean((Q - self.Q_optimal) ** 2)
        return mse
    
    def feature_vector_convertor(self, dealer_card, player_sum, action):
        self.dc_feature[:] = 0
        self.ps_feature[:] = 0
        self.a_feature[:] = 0
        self.feature_vector[:] = 0
        
        if 1<=dealer_card<=4:
                self.dc_feature[0]=1
        if 4<=dealer_card<=7:
                self.dc_feature[1]=1
        if 7<=dealer_card<=10:
                self.dc_feature[2]=1

        x=np.array([1, 4, 7, 10, 13, 16])
        for i in range(6):
            if x[i]<=player_sum<=(x[i]+5):
                self.ps_feature[i]=1
        
        self.a_feature[action-1] = 1

        for i in range(3):
             for j in range(6):
                  for k in range(2):
                       self.feature_vector[(i+1)*(j+1)*(k+1)-1] = (self.dc_feature[i])*(self.ps_feature[j])*(self.a_feature[k])

        return self.feature_vector

    def train(self):
        Q = np.zeros((11,22,2))
        E = np.zeros((36))
        weight = np.zeros((36))
        environment = env.Easy21_env()
        MSE_list = []

        for ep in range(self.episodes):
            player_sum1 = random.randint(1, 10)
            dealer_card_1 = random.randint(1, 10)
            terminated = False
            state = (dealer_card_1, player_sum1, terminated)
            action = self.epsilon_greedy_policy(Q[dealer_card_1, player_sum1])
            feature = self.feature_vector_convertor(dealer_card_1, player_sum1, action)

            E[:]=0

            while terminated == False:

                E = E * self.lamda * self.gamma
                E += feature

                Q_value = np.sum((feature.T)*weight)

                Next_state, Reward = environment.step(state, action)
                dealer_card_1, player_sum1, terminated = Next_state

                if terminated == False:
                    Next_action = self.epsilon_greedy_policy(Q[dealer_card_1, player_sum1])
                    Next_feature = self.feature_vector_convertor(dealer_card_1, player_sum1, Next_action)
                    
                    delta = Reward + np.sum((Next_feature.T)*weight) - Q_value
                    weight += E * delta * self.alpha

                else:
                    Next_action=0
                    Next_feature = np.zeros((36))
                    delta = Reward + 0 - Q_value
                    weight += E * delta * self.alpha

                action = Next_action
                state = Next_state
                feature = Next_feature

                
            if self.lamda==0 or self.lamda==1:
                for i in range(11):
                     for j in range(22):
                          for k in range(2):
                               feature_vec = self.feature_vector_convertor(i+1, j+1, k+1)
                               Q[i,j,k] =  np.sum((feature_vec.T)*weight)
                MSE = self.Mean_squared_error(Q)
                MSE_list.append(MSE)
        
        for i in range(11):
             for j in range(22):
                  for k in range(2):
                       feature_vec = self.feature_vector_convertor(i+1, j+1, k+1)
                       Q[i,j,k] =  np.sum((feature_vec.T)*weight)
                
        MSE = self.Mean_squared_error(Q)
        print("Done for:"," ", self.lamda," ", "mse:", MSE)

        return MSE_list, MSE

agent = env.Monte_carlo_control(1000000, 100)
Q_MC = agent.train()

if __name__ == "__main__":
    list_l = np.arange(0, 1.01, 0.1)
    ep_list = np.arange(0, 1000, 1)
    MSE_list_x =[]
    for l in list_l:
        agent_TD = linear_approximation_sarsa(1000, l, Q_MC, 1)
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
    plt.plot(ep_list, MSE_list_0, color = 'red', label = 'mean-squared error for lambda = 0')
    plt.xlabel('episodes')
    plt.ylabel('MSE')
    plt.title('mean-squared error for lambda = 0 and 1')
    
    plt.plot(ep_list, MSE_list_1, color = 'blue', label = 'mean-squared error for lambda = 1')
    plt.xlabel('episodes')
    plt.ylabel('MSE')
    
    plt.show()

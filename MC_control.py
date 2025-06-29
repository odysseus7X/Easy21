import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#hyperparameters
episodes = 1000000
N_o = 100

class Easy21_env():
    def __init__(self):
        super().__init__()
        self.stick = 0
        self.hit = 1
        self.dealer_sum = 0

    def step(self, state, action):
        dealer_card, player_sum, terminated = state
        self.dealer_sum = dealer_card
        reward = 0
        next_state = state

        if action==self.stick:
            while self.dealer_sum < 17:
                card_drawn_d = self.card_drawer()
                self.dealer_sum += card_drawn_d

                if self.dealer_sum > 21 or self.dealer_sum < 1:
                    reward = 1
                    terminated = True
                    next_state = (dealer_card, player_sum, terminated)
                    return next_state, reward
                
            terminated = True
            next_state = (dealer_card, player_sum, terminated)
            if self.dealer_sum > player_sum :
                reward = -1
            elif self.dealer_sum== player_sum:
                reward = 0
            else:
                reward = 1

            return next_state, reward
        
        else:
            card_drawn_p = self.card_drawer()
            player_sum += card_drawn_p
            
            if player_sum > 21 or player_sum < 1:
                reward = -1
                terminated = True

        next_state = (dealer_card, player_sum, terminated)

        return next_state, reward
    
    def card_drawer(self, zeta=1/3):
        rand_num = random.random()
        if rand_num <= zeta:
            pick = random.randint(-10, -1)
        else:
            pick = random.randint(1, 10)
        
        return pick

class Monte_carlo_control():
    def __init__(self, episodes, N_o):
        super().__init__()
        self.episodes = episodes
        self.N_o = N_o

    def train(self):
        Q = np.zeros((11,22,2))
        N = np.zeros((11,22,2))
        N_s = np.zeros((11,22))
        env = Easy21_env()

        for ep in range(self.episodes):
            History = {}
            player_sum1 = random.randint(1,10)
            dealer_card_1 = random.randint(1,10)
            Return = 0
            terminated = 0
            state = (dealer_card_1, player_sum1, terminated)

            while terminated==False:
                rand_numb = random.random()
                epsilon = self.N_o/(self.N_o + N_s[dealer_card_1, player_sum1])
                if rand_numb <= epsilon:
                    action = random.randint(0,1)
                else:
                    action = np.argmax(Q[dealer_card_1, player_sum1])

                History[state] = History.get(state, action)
                next_state, Reward = env.step(state, action)

                Return += Reward
                state = next_state
                dealer_card_1, player_sum1, terminated = state


            for state, action in History.items():
                dealer_card_1, player_sum1, terminated = state
                N[dealer_card_1, player_sum1, action]+=1
                N_s[dealer_card_1, player_sum1]+=1

                Q[dealer_card_1, player_sum1, action] += (Return-Q[dealer_card_1, player_sum1, action])/N[dealer_card_1, player_sum1, action]

            if ep%500 == 0:
                sum = np.sum(Q)
                print("average value at episode", ep, ":", sum/420 )
        
        return Q


if __name__ == "__main__":
    agent = Monte_carlo_control(episodes, N_o)
    Q = agent.train()

    x = np.arange(0, 11)
    y = np.arange(0, 22)
    X, Y = np.meshgrid(x, y)
    Z_max = np.max(Q, axis=2).T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z_max, cmap='plasma')

    ax.set_xlabel("Dealer Card")
    ax.set_ylabel("Player Sum")
    ax.set_zlabel("Value")
    ax.set_title("State-Value Function")

    plt.show()

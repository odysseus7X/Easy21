# Easy21 – Assignment Solution (David Silver's RL Course)

This repository contains my solution to the **Easy21 assignment** from [David Silver's Reinforcement Learning course at UCL](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html).

The assignment explores Reinforcement Learning concepts such as:
- Monte Carlo control
- Temporal Difference (TD) learning
- Function approximation
- Eligibility traces (SARSA(λ))

---

## 1. Implementation of Easy21

The Easy21 environment was implemented following the problem definition provided in the assignment. The environment is similar to Blackjack, with some key differences (e.g., cards can be red or black, no aces, no splitting, and the deck is infinite).

---

## 2. Monte-Carlo Control in Easy21

Monte Carlo Control was used to learn the optimal policy through episodic sampling without bootstrapping. The action-value function is updated only at the end of each episode.

### 🔍 Result:

![Monte Carlo Value Function](/easy21_plots/MC_value.png)

---

## 3. TD Learning in Easy21

Temporal Difference (TD) learning methods were applied to compare their performance with Monte Carlo.

### 🔍 Results:

- MSE vs lambda
  
  ![.](/easy21_plots/MSEvslambda_sarsa.png)

- MSE vs episodes for TD(0)

  ![.](/easy21_plots/MSE_0_sarsa.png)

- MSE vs episodes for Monte carlo(λ=1)

  ![.](/easy21_plots/MSE_1_sarsa.png)

---

## 4. Linear Function Approximation in Easy21

A linear function approximator with coarse coding to generalize value estimates across similar states.

### 🔍 Results:

- MSE vs λ (Linear Approximation)

  ![MSE vs Lambda](/easy21_plots/MSEvslambda_linear_approx.png)

- MSE vs episodes for linear approximated sarsa (λ=0 and 1)
  Red - λ=0
  Blue- λ=1
  
  ![.](/easy21_plots/MSE_0_1_linear_approx.png)

---

## 5. Discussion

### • What are the pros and cons of bootstrapping in Easy21?

Pros:-  
Helps in reducing variance.  
Better credit assignment as value is updated after each step.  
Can be done online as no need to wait till the end of the episode.  

Cons:-  
Increased bias.
More dependent on the initial values.

---

### • Would you expect bootstrapping to help more in Blackjack or Easy21? Why?

Easy21. It has longer episodes on average as compared to that of blackjack. Bootstrapping shows greater effect when it is done on episodes with larger no. of steps.

---

### • What are the pros and cons of function approximation in Easy21?

Pros:-  
Less memory usage.  
Faster calculation as it has lesser no. of parameters.  

Cons:-  
Cannot attain true optimum value.

---

### • How would you modify the function approximator suggested in this section to get better results in Easy21?

Can use better appproximators like neural networks. Increasing the no. of parameters will also lead to better approximation.

# Easy21 – Assignment Solution (David Silver's RL Course)

This repository contains my solution to the **Easy21 assignment** from [David Silver's Reinforcement Learning course at UCL](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html).

The assignment explores Reinforcement Learning concepts such as:
- Monte Carlo control
- Temporal Difference (TD) learning
- Function approximation
- Eligibility traces (SARSA(λ))

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



---

### • Would you expect bootstrapping to help more in Blackjack or Easy21? Why?



---

### • What are the pros and cons of function approximation in Easy21?



---

### • How would you modify the function approximator suggested in this section to get better results in Easy21?



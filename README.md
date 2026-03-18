# VIA

Many decisions can be reframed into a **Markov Decision Process**, where there is a set of states, a set of actions, probabilities of moving between states when performing an action, and a set of rewards from making actions. A common way of determining the best set of actions that maximise the rewards, known as the *optimal policy*, is using *value iteration*. 

This package implements a simple value iteration algorithm with the `value_iteration_function` in the `via.py` file. It includes $2$ examples: 
* Sam who is trying to decide whether to party or rest when he is healthy and sick,
* somone who is trying to leave a 2x2 grid in a Grid World game.

Both of these examples have a python file with the required transition dynamics and reward functions, and an additional jupyter notebook with more details on the problem set up and how to implement value iteration using these functions. These are known as `happy_sick_functions.py` and `happy_sick_example.ipynb`, and `grid_world_functions.py` and `grid_world_example.ipynb` respectively.

## Value iteration algorithm - how was it implemented?

```text
Algorithm: Value Iteration

Input: states S, actions A, transition function P(s'|s,a), reward R(s,a,s'), discount gamma, threshold epsilon
Output: Value function V(s), Policy pi(s)

1. Initialize V(s) = 0 for all s in S
2. Repeat:
3.     delta = 0
4.     For each state s in S:
5.         v_old = V(s)
6.         V(s) = max_a sum_{s'} P(s'|s,a) * (R(s,a,s') + gamma * V(s'))
7.         delta = max(delta, |V(s) - v_old|)
8. Until delta < epsilon
9. For each state s in S:
10.    pi(s) = argmax_a sum_{s'} P(s'|s,a) * (R(s,a,s') + gamma * V(s'))
11. Return V, pi
```

### Comparison to Chapter 9.5.2's pseudocode


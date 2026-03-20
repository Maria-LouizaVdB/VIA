# Value Iteration Algorithm (VIA)

Many decisions can be reframed into a **Markov Decision Process**, where there is a set of states, a set of actions, probabilities of moving between states when performing an action, and a set of rewards from making actions. A common way of determining the best set of actions that maximise the rewards, known as the *optimal policy*, is using *value iteration*. 

This package implements a simple value iteration algorithm with the `value_iteration_function` in the `via.py` file. It includes $2$ examples: 
* Sam who is trying to decide whether to party or rest when he is healthy and sick,
* somone who is trying to leave a 2x2 grid in a Grid World game.

Both of these examples have a python file with the required transition dynamics and reward functions, and an additional jupyter notebook with more details on the problem set up and how to implement value iteration using these functions. These are known as `happy_sick_functions.py` and `happy_sick_example.ipynb`, and `grid_world_functions.py` and `grid_world_example.ipynb` respectively.

## Installation

This package, known as *value_iteration*, is installed from the GitHub repository using: `python -m pip install 'git+https://github.com/Maria-LouizaVdB/VIA' `. It is uninstalled with `python -m pip uninstall value_iteration`.

## Value iteration algorithm - how was it implemented?

```markdown
# Algorithm: Value Iteration

## Input:
* states S, 
* actions A, 
* transition function P(s'|s,a), 
* reward R(s,a,s'), 
* discount gamma, 
* threshold epsilon, 
* maximum iteration max_k

## Output:
* value function V(s), 
* policy pi(s),
* number of iterations k

## Pseudocode
1. Initialize V_0(s) = 0 for all s in S
2. k = 0
2. Repeat until delta < epsilon or k > max_k 
   a. Increase k by 1, k = k+1
   b. delta = 0
   c. For each state s in S:
      - V_k(s) = max_a sum_{s'} P(s'|s,a) * (R(s,a,s') + gamma * V_{k-1}(s'))
      - delta = max(delta, |V_k(s) - V_{k-1}(s)|)
3. Initialise pi(s) = 0 for all s in S
4. For each state s in S:
   - pi(s) = argmax_a sum_{s'} P(s'|s,a) * (R(s,a,s') + gamma * V_k(s'))
5. Return V, pi, k
```

### Comparison to Chapter 9.5.2's pseudocode

This algorithm was based on the pseudocode in Chapter 9.5.2 of Artifical Integlligence: Foundations of Computational Agents by David L. Poole and Alan K. Mackworth (https://artint.info/2e/html2e/ArtInt2e.Ch9.S5.SS2.html). However, it expands on the `until termination` requirement. In this version, there are $2$ possible **termination events**:
* the value function has been deemed to **converge** when for every state, the difference between consecutive value functions is small. That is, when $\forall s \in \text{States}, |V_{k}(s)-V_{k-1}(s)|<\epsilon$,
* or if the algorithm has **run for a long time** and not yet converged. By changing the `max_k` argument, it is possible to change what a 'long time' is per implementation. This can be used to limit the run time and to explore the convergence rate over time.
As a result of this change, we also return the number of iterations and print a warning when the value functions did not converge. 

This package's function also assumes that the value function should be initiallised at $0$ for all states. However, this may not necessarily be the case for Poole and Mackworth's algorithm. 

### Transition and reward functions

The transition and reward functions are coded to accept $3$ arguments: the current state, the next state and the action. They should return a singular value that is either a probability (float) or reward (numeric). 

When coding the value iteration function, the arguments for transition and rewards are encoded as functions rather than a dictionary. This is so that it can be more versitile than a look up table. If it is beneficial to consider these functions as dictionaries, these can be included inside a function (as was done in both examples for this case). 
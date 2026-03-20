from typing import Callable, List, Any
import numpy

def value_iteration_function(states: List[Any],
                             actions: List[Any],
                             trans_fun: Callable[ [Any,Any,Any], float ],  
                             reward_fun: Callable[ [Any,Any,Any], float ],
                             epsilon: float = 0.05,
                             gamma: float = 0.95,
                             max_iteration: int = 100
                            ) -> tuple[dict, dict, int]:

    """ 
    Perform value interation
    
    input: 
        states - list of states
        actions - list of actions between states
        trans_fun - function that inputs 2 states and an action and returns probability that the action occurs 
        reward_fun - function that takes in 2 states and an action, and returns a reward
        gamma - discount factor, default 0.95
        epsilon - treshold to check if value function has stabilised, default 0.05
        max_iteration - maximum number of iterations before the function ends to avoid infinite loop, default 100

    output:
        pi - approximate optimal policy
        v - value function
        k - number of iteration steps before convergence
    """

    # sanity check inputs
    if len(states) == 0: raise ValueError("Warning: no states inputted.")
    if len(actions) == 0: raise ValueError("Warning: no actions inputted.")
    if epsilon < 0: raise ValueError("Warning: epsilon is negative.")
    if gamma < 0 or gamma > 1: raise ValueError("Warning: gamma is outside of [0,1].")
    if max_iteration <= 0: raise ValueError("Warning: max_iteration is not positive.")
    
    # calculate value function values until stabalises
    v = {s: 0.0 for s in states}
    k = 0

    while True:
        k += 1
        delta = 0
        # calculate value function for each state
        for s in states:
            v_old = v[s]
            v[s] = max( sum(trans_fun(s,s_next,a) * reward_fun(s, s_next, a) + 
                            trans_fun(s,s_next,a) * gamma * v[s_next] for s_next in states) for a in actions)
            
            delta = max( delta, abs(v[s] - v_old) )
            
        # check stabalised
        if delta < epsilon:
            break
            
        # escape infinite loop
        if k > max_iteration: 
            print("Warning: did not converge within", max_iteration, "iteration steps!")
            break
        
    # determine policy
    pi = {s: 0.0 for s in states}
    
    for s in states:
        pi[s] = actions[numpy.argmax( [ sum(trans_fun(s,s_next,a) * reward_fun(s,s_next, a) + 
                                          trans_fun(s,s_next,a) * gamma * v[s_next] for s_next in states) for a in actions ] )]
        # numpy.argmax takes list input!
    
    return v, pi, k-1
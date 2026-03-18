from typing import Callable, List, Any
import numpy

def value_iteration_function(states: List[Any],
                             actions: List[Any],
                             trans_fun: Callable[ [Any,Any,Any], float ],  
                             reward_fun: Callable[ [Any,Any,Any], float ],
                             epsilon: float = 0.05,
                             gamma: float = 0.95
                            ) -> tuple[dict, dict]:

    """ 
    Perform value interation
    
    input: 
        states - list of states
        actions - list of actions between states
        trans_fun - function that inputs 2 states and an action and returns probability that the action occurs 
        reward_fun - function that takes in 2 states and an action, and returns a reward
        epsilon - treshold to check if value function has stabilised
        gamma - discount factor

    output:
        pi - approximate optimal policy
        v - value function
    """
    
    # calculate value function values until stabalises
    v = {s: 0.0 for s in states}
    
    while True:
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

    # determine policy
    pi = {s: 0.0 for s in states}
    
    for s in states:
        pi[s] = actions[numpy.argmax( [ sum(trans_fun(s,s_next,a) * reward_fun(s,s_next, a) + 
                                          trans_fun(s,s_next,a) * gamma * v[s_next] for s_next in states) for a in actions ] )]
        # numpy.argmax takes list input!
    
    return v, pi
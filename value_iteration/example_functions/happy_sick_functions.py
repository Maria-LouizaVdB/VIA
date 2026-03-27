from typing import Literal

def prob_27(states_current: Literal["h","s"], 
            states_next: Literal["h","s"], 
            actions: Literal["r","p"] ) -> float:

    """
    transition probabilities of exercise 9.27 from Artificial Intelligence: Foundations of Computational Agents
    https://artint.info/2e/html2e/ArtInt2e.Ch9.S5.html

    input: 
        states_current - initial state, h (healthy) or s (sick)
        states_next - next state, h (healthy) or s (sick)
        action - r (relax) or p (party)

    output:
        probability that the proposed move and action is successful
    """

    # sense checking
    if states_current not in  ["h","s"]: raise ValueError("Warning: incorrect state.")
    if states_next not in  ["h","s"]: raise ValueError("Warning: incorrect state.")
    if actions not in ["r","p"]: raise ValueError("Warning: incorrect action.")

    # determining transition probability
    trans_prob_dict = {"hhr": 0.95,
                       "hsr": 0.05,
                       "hhp": 0.7,
                       "hsp": 0.3,
                       "shp": 0.1,
                       "ssp": 0.9,
                       "ssr": 0.5,
                       "shr": 0.5}

    dict_look_up = states_current + states_next + actions

    return trans_prob_dict[dict_look_up]

def reward_27(states: Literal["h","s"],
              states_next: Literal["h","s"],
              actions: Literal["r","p"] ) -> float:

    """
    reward outcome of exercise 9.27 from Artificial Intelligence: Foundations of Computational Agents
    https://artint.info/2e/html2e/ArtInt2e.Ch9.S5.html

    input: 
        states - initial state, h (healthy) or s (sick)
        states_next - next state, h (healthy) or s (sick), this is not used when calculating results but the input is needed for the via function
        action - r (relax) or p (party)

    output:
        reward - reward from moving from initial state to final state with 
    """

    # sense checking
    if states not in  ["h","s"]: raise ValueError("Warning: incorrect state.")
    if states_next not in  ["h","s"]: raise ValueError("Warning: incorrect state.")
    if actions not in ["r","p"]: raise ValueError("Warning: incorrect action.")

    # determining reward
    reward_dict = {"hr": 7,
                   "hp": 10,
                   "sr": 0,
                   "sp": 2}

    dict_look_up = states + actions

    return reward_dict[dict_look_up]
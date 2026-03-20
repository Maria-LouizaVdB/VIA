def prob_grid_world(states_current: ["tl","tr","bl","br"],
                    states_next: ["tl","tr","bl","br"],
                    actions: ["r","l","u","d"]) -> float:

    """
    transition probabilities of Grid World example

    input: 
        states_current - initial state, tl (top left), tr (top right), bl (bottom left), br (bottom right)
        states_next - next state, tl (top left), tr (top right), bl (bottom left), br (bottom right)
        action - r (move right), l (move left), u (move up), d (move down)

    output:
        probability that the proposed move and action is successful
    """

    # possible states and actions
    states_set = ["tl","tr","bl","br"]
    actions_set = ["r","l","u","d"]
    
    # sense checking
    if states_current not in states_set: return print("Warning: incorrect state.")
    if states_next not in  states_set: return print("Warning: incorrect state.")
    if actions not in actions_set: return print("Warning: incorrect action.")

    # determining transition probability
    trans_prob_dict = {f"{s}_{s_n}_{a}": 0.0 for s in states_set for s_n in states_set for a in actions_set} # every possible transition has value 0
    
    update_prob_dict = {"tl_tr_r": 0.9,
                        "tl_bl_r": 0.1,
                        "tl_bl_d": 0.9,
                        "tl_tr_d": 0.1,
                        "tr_tl_l":0.9,
                        "tr_br_l":0.1,
                        "tr_br_d": 0.8,
                        "tr_tl_d":0.2,
                        "bl_br_r":0.9,
                        "bl_tl_r":0.1,
                        "bl_tl_u": 0.8,
                        "bl_br_u": 0.2,
                        "br_br_r": 1, # bottom right is terminal
                        "br_br_l": 1,
                        "br_br_u": 1,
                        "br_br_d": 1
                        }

    trans_prob_dict.update(update_prob_dict)

    dict_look_up = states_current + "_" + states_next + "_" + actions

    return trans_prob_dict[dict_look_up]

def reward_grid_world(states_current: ["tl","tr","bl","br"],
                      states_next: ["tl","tr","bl","br"],
                      actions: ["r","l","u","d"]) -> float:

    """
    rewards after one move of Grid World example

    input: 
        states_current - initial state, tl (top left), tr (top right), bl (bottom left), br (bottom right)
        states_next - next state, tl (top left), tr (top right), bl (bottom left), br (bottom right)
        action - r (move right), l (move left), u (move up), d (move down)

    output:
        probability that the proposed move and action is successful
    """

    # possible states and actions
    states_set = ["tl","tr","bl","br"]
    actions_set = ["r","l","u","d"]
    
    # sense checking
    if states_current not in states_set: return print("Warning: incorrect state.")
    if states_next not in  states_set: return print("Warning: incorrect state.")
    if actions not in actions_set: return print("Warning: incorrect action.")

    # determining reward probability
    reward_dict = {f"{s}_{s_n}_{a}": 0.0 for s in states_set for s_n in states_set for a in actions_set} # every possible transition has value 0
    
    update_reward_dict = {"tl_tr_r": -1,
                        "tl_bl_r": -2,
                        "tl_bl_d": -2,
                        "tl_tr_d": -1,
                        "tr_tl_l":-1.5,
                        "tr_br_l": 10,
                        "tr_br_d": 15,
                        "tr_tl_d": -1,
                        "bl_br_r": 20,
                        "bl_tl_r": -2.5,
                        "bl_tl_u": -0.5,
                        "bl_br_u": 5
                        }

    reward_dict.update(update_reward_dict)

    dict_look_up = states_current + "_" + states_next + "_" + actions

    return reward_dict[dict_look_up]
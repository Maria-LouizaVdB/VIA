### Original functions
from value_iteration import value_iteration_function, prob_27, reward_27

### unit testing
import pytest
from pytest import approx

### testing value_iteration_function when using prob_27 and reward_27 for various epsilon and max_iteration

@pytest.mark.parametrize("epsilon, max_iteration, expected", [
    (0.05, 1, ({'h': 10, 's': 2}, {'h': 'p', 's': 'r'}, 1)),
    (0.05, 2, ({'h': 16.84, 's': 5.4}, {'h': 'p', 's': 'r'}, 2)),
    (0.05, 3, ({'h': 22.067199999999996, 's': 10.008000000000001}, {'h': 'p', 's': 'r'}, 3)),
    (0.05, 50, ({'h': 66.63720895148133, 's': 54.442087000261814}, {'h': 'p', 's': 'r'}, 47)),
    (0.1, 50, ({'h': 66.25283172845768, 's': 54.057709777238166}, {'h': 'p', 's': 'r'}, 41))
])

def test_via_27(epsilon,max_iteration,expected):
    assert value_iteration_function(states = ["h","s"], actions = ["p","r"], 
                                    trans_fun = prob_27, reward_fun = reward_27,
                                    epsilon = epsilon, gamma = 0.9, max_iteration = max_iteration) == approx(expected)

### check warning for value_iteration function when using prob_27 and reward_27

@pytest.mark.parametrize("states,actions, epsilon, gamma, max_iteration,err_type, err_msg", [
    ([],["r","p"], 0.05, 0.9, 50, ValueError, "Warning: no states inputted."),
    (["h","s"],[], 0.05, 0.9, 50, ValueError, "Warning: no actions inputted."),
    (["h","s"],["r","p"], -1, 0.9, 50, ValueError, "Warning: epsilon is negative."),
    (["h","s"],["r","p"], 0.05, -1, 50, ValueError, "Warning: gamma is outside of [0,1]."),
    (["h","s"],["r","p"], 0.05, 10, 50, ValueError, "Warning: gamma is outside of [0,1]."),
    (["h","s"],["r","p"], 0.05, 0.9, 0, ValueError, "Warning: max_iteration is not positive."),
    (["h","s"],["r","p"], 0.05, 0.9, -1, ValueError, "Warning: max_iteration is not positive.")
])
def test_via_27_invalid(states,actions, epsilon, gamma, max_iteration, err_type, err_msg):
    with pytest.raises(err_type) as e:
        value_iteration_function(states = states, actions = actions, 
                                    trans_fun = prob_27, reward_fun = reward_27,
                                    epsilon = epsilon, gamma = gamma, max_iteration = max_iteration)
    assert str(e.value) == err_msg
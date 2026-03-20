### Original functions
from value_iteration import value_iteration_function, prob_27, reward_27

### unit testing
import pytest
from pytest import approx

### testing value_iteration_function when using prob_27 and reward_27 for various epsilon and max_iteration

@pytest.mark.parametrize("epsilon, max_iteration, expected", [
    (0.05, 2, ({'h': 23.7092725, 's': 15.127210125000001}, {'h': 'p', 's': 'r'}, 2)),
    (0.05, 50, ({'h': 66.75194627324974, 's': 54.58185655196772}, {'h': 'p', 's': 'r'}, 40)),
    (0.1, 50, ({'h': 66.37655187874134, 's': 54.23571571693515}, {'h': 'p', 's': 'r'}, 34))
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
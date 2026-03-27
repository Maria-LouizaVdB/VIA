### Original functions
from value_iteration import value_iteration_function, prob_27, reward_27

### unit testing
import pytest
from pytest import approx

### One step iteration check
def test_via_one_step():
    v,pi,k = value_iteration_function(states = ["h","s"],
                                actions = ["p","r"],
                                trans_fun = prob_27,
                                reward_fun = reward_27,
                                epsilon = 0.05,
                                gamma = 0.9,
                                max_iteration = 1)

    assert k == 1
    assert pi == {'h': 'p', 's': 'r'}
    assert v["h"] == approx(10.0)
    assert v["s"] == approx(2.0)

### Checking one step, two step and three step iterations, and dependency on epsilon
@pytest.mark.parametrize("epsilon, max_iteration, expected", [
    (0.05, 1, ({'h': 10, 's': 2}, {'h': 'p', 's': 'r'}, 1)),
    (0.05, 2, ({'h': 16.84, 's': 5.4}, {'h': 'p', 's': 'r'}, 2)),
    (0.05, 3, ({'h': 22.0672, 's': 10.008}, {'h': 'p', 's': 'r'}, 3)),
    (0.05, 50, ({'h': 66.63720895, 's': 54.442087}, {'h': 'p', 's': 'r'}, 47)),
    (0.1, 50, ({'h': 66.25283173, 's': 54.05770977}, {'h': 'p', 's': 'r'}, 41))
])
def test_via_27_iterations(epsilon, max_iteration, expected):
    expected_v, expected_pi, expected_k = expected
    v, pi, k = value_iteration_function(
        states=["h","s"], actions=["p","r"],
        trans_fun=prob_27, reward_fun=reward_27,
        epsilon=epsilon, gamma=0.9, max_iteration=max_iteration
    )
    for s in expected_v:
        assert v[s] == approx(expected_v[s], rel=1e-5)
    assert pi == expected_pi
    assert k == expected_k

### Checking error messages are correctly triggered
@pytest.mark.parametrize("states,actions, epsilon, gamma, max_iteration, err_type, err_msg", [
    ([], ["r","p"], 0.05, 0.9, 50, ValueError, "Warning: no states inputted."),
    (["h","s"], [], 0.05, 0.9, 50, ValueError, "Warning: no actions inputted."),
    (["h","s"], ["r","p"], -1, 0.9, 50, ValueError, "Warning: epsilon is negative."),
    (["h","s"], ["r","p"], 0.05, -1, 50, ValueError, "Warning: gamma is outside of [0,1]."),
    (["h","s"], ["r","p"], 0.05, 10, 50, ValueError, "Warning: gamma is outside of [0,1]."),
    (["h","s"], ["r","p"], 0.05, 0.9, 0, ValueError, "Warning: max_iteration is not positive."),
    (["h","s"], ["r","p"], 0.05, 0.9, -1, ValueError, "Warning: max_iteration is not positive.")
])
def test_via_27_invalid(states, actions, epsilon, gamma, max_iteration, err_type, err_msg):
    with pytest.raises(err_type) as e:
        value_iteration_function(
            states=states, actions=actions,
            trans_fun=prob_27, reward_fun=reward_27,
            epsilon=epsilon, gamma=gamma, max_iteration=max_iteration
        )
    assert err_msg in str(e.value)
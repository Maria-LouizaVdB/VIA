### Original functions
from value_iteration import value_iteration_function, prob_27, reward_27

### unit testing
import pytest
from pytest import approx

### check correct output for prob_27

@pytest.mark.parametrize("s,s_next,a,expected", [
    ("h","h","r",0.95),
    ("h","h","p",0.7),
    ("h","s","p",0.3),
    ("h","s","r",0.05),
    ("s","s","p",0.9),
    ("s","s","r",0.5),
    ("s","h","p",0.1),
    ("s","h","r",0.5)
])

def test_prob_27(s,s_next,a,expected):
    assert prob_27(s,s_next,a) == approx(expected)

### check warning for prob_27

@pytest.mark.parametrize("s,s_next,a,err_type,err_msg", [
    ("A","h","p", ValueError, "Warning: incorrect state."),
    ("h","A","p", ValueError, "Warning: incorrect state."),
    ("h","h","A", ValueError, "Warning: incorrect action."),
])
def test_prob_27_invalid(s, s_next, a, err_type, err_msg):
    with pytest.raises(err_type) as e:
        prob_27(s, s_next, a)
    assert str(e.value) == err_msg

### check correct output from reward_27

@pytest.mark.parametrize("s,s_next,a,expected", [
    ("h","h","r",7), # middle value is removed immediately by reward_27 function
    ("h","h","p",10),
    ("s","h","r",0),
    ("s","h","p",2)
])

def test_reward_27(s,s_next,a,expected):
    assert reward_27(s,s_next,a) == approx(expected)

### check warning from reward_27

@pytest.mark.parametrize("s,s_next,a,err_type,err_msg", [
    ("A","h","p", ValueError, "Warning: incorrect state."),  # invalid current state
    ("h","A","p", ValueError, "Warning: incorrect state."),  # invalid next state
    ("h","h","A", ValueError, "Warning: incorrect action."), # invalid action
])

def test_reward_27_invalid(s, s_next, a, err_type, err_msg):
    with pytest.raises(err_type) as e:
        reward_27(s, s_next, a)
    assert str(e.value) == err_msg
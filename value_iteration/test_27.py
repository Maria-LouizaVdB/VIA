from via_function import value_iteration_function
from happy_sick_example/happy_sick_functions import prob_27, reward_27

import pytest
from pytest import approx

@pytest.mark.parametrize("s,s_next,a,expected", [
    ("h","h","r",0.95),
    ("h","h","p",0.7),
    ("h","s","p",0.3),
    ("h","s","r",0.05),
    ("s","s","p",0.9),
    ("s","s","r",0.5),
    ("s","h","p",0.1),
    ("s","h","r",0.5),
    ("A","h","p",{"error":"Warning: incorrect state."}), # check incorrect state
    ("h","A","p",{"error":"Warning: incorrect state."}), # check incorrect state
    ("h","h","A",{"error":"Warning: incorrect action."}) # check incorrect state
])

def test_prob_27(s,s_next,a,expected):
    assert prob_27(s,s_next,a) == approx(expected)

@pytest.mark.parametrize("s,s_next,a,expected", [
    ("h","h","r",7), # middle value is removed immediately by reward_27 function
    ("h","h","p",10),
    ("s","h","r",0),
    ("s","h","p",2),
    ("A","h","p",{"error":"Warning: incorrect state."}), # check incorrect state
    ("h","A","p",{"error":"Warning: incorrect state."}), # check incorrect state
    ("h","h","A",{"error":"Warning: incorrect action."}) # check incorrect state
])

def test_reward_27(s,s_next,a,expected):
    assert reward_27(s,s_next,a) == approx(expected)
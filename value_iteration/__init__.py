### Required to find the original functions
import sys
import os

parent_dir = os.path.abspath("..")  
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir) 

from via_function import value_iteration_function
from happy_sick_example.happy_sick_functions import prob_27, reward_27
from grid_world_example.grid_world_functions import prob_grid_world, reward_grid_world
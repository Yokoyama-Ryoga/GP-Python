from settings import *
import linear_GP
import tiny_gp_plus

if structure == "linear":
    linear_GP.linear()
elif structure == "tree":
    tiny_gp_plus.tree()
else:
    print("Data structure is not found.")
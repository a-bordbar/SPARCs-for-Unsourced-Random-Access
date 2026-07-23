import numpy as np
import matplotlib.pyplot as plt


# This script reproduces fig.3 in the paper

L = 8 
B = 100 
J_vec = np.array([12, 15, 20]) # Number of bits in each section
p = 2.0 ** (-J_vec) # Probability of a user selecting an index

Ka_vec = np.arange(25, 301, 5) # Number of active users

# I want to first plot the entropy bound. The rate is R <= (2^J / Ka) * H_2*(1-p0)
def H2(p):
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

R_bound = np.zeros((len(J_vec), len(Ka_vec)))
for j_idx, J in enumerate(J_vec):
    p0 = (1-p[j_idx]) ** Ka_vec
    R_bound[j_idx, :] = (2**J / Ka_vec) * H2(p0) / J


 



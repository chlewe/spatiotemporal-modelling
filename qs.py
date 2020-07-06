"""
Global variables for quorum sensing (QS)
"""

d1 = 0.25      # diffusion rate of AHL from outside to inside the cell
d2 = 2.5       # diffusion rate of AHL from inside to outside the cell
alpha = 1      # low production rate of AHL
beta = 100     # increase of production rate of AHL
u_thresh = 2   # threshold AHL concentration between low and increased activity
n = 10         # polymerisation degree
gamma_e = 0.5  # decay rate of extracellular AHL
gamma_c = 0.5  # decay rate of intracellular AHL

bacteria_particles = []

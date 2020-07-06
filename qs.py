"""
Global variables for quorum sensing (QS)
"""

D = 1
domain_lower_bound = 0
domain_upper_bound = 50
particle_number_per_dim = 51
h = (domain_upper_bound - domain_lower_bound) / (particle_number_per_dim - 1)
epsilon = h
volume_p = h ** 2
cutoff = 3 * epsilon
cell_side = cutoff
t_max = 20
dt = h ** 2 / (4 * D)

d1 = 0.25      # diffusion rate of AHL from outside to inside the cell
d2 = 2.5       # diffusion rate of AHL from inside to outside the cell
alpha = 1      # low production rate of AHL
beta = 100     # increase of production rate of AHL
u_thresh = 2   # threshold AHL concentration between low and increased activity
n = 10         # polymerisation degree
gamma_e = 0.5  # decay rate of extracellular AHL
gamma_c = 0.5  # decay rate of intracellular AHL

bacteria_particles = []


def update_h():
    global h
    h = (domain_upper_bound - domain_lower_bound) / (particle_number_per_dim - 1)

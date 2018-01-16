"""Python 3 script for simulating p=2 spherical model and neural model. """ 
from numpy import sqrt, identity, outer, empty, array, isclose
from numpy import zeros, tanh
from scipy.stats import multivariate_normal, norm as normal
from scipy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

def gen_gaussian_initial_state(N, mean=0, variance=1):
    """gen_gaussian_initial_state: Generate a gaussian initial state

    :param N: size of state
    :param mean: mean of gaussian 
    :param cov: variance of gaussian
    """
    X = normal.rvs(size=N, loc=mean, scale=variance)
    return X

def gen_gaussian_spherical_initial_state(N, mean=0, variance=1):
    """gen_gaussian_spherical_initial_state: Generate a gaussian initial
    state with \sum_i x_i^2 = N

    :param N: size of state
    :param mean: mean of gaussian 
    :param cov: variance of gaussian
    """
    X = normal.rvs(size=N, loc=mean, scale=variance)
    return sqrt(N) * X/norm(X)

def gen_gaussian_J(N, J, eta, symmetry_threshold=0.99):
    """gen_gaussian_J: generate a matrix J with elements drawn from 
     a gaussian. Anything above symmetry_threshold (0.99)
     generate a symmetric matrix.

    :param N: Size of matrix
    :param J: coupling strength
    :param eta: asymmetry
    """
    

    if abs(eta) > symmetry_threshold:
        symmetric = True
    else:
        symmetric=False

    if not symmetric:
        # require asymmetric
        gaussian_J = zeros([N, N])
        # joint distribution of pairs on either side of diagonal 
        # (eta is the covariance)
        cov = J**2/N*array([[1, eta],
                               [eta, 1]])
        # distribution object with zero mean, given covariance. 
        # Each sample is a correlated pair
        pair_dist = multivariate_normal([0, 0], cov)
        # put the pairs in the matrix 
        for i, j in [(i, j) for i in range(N) for j in range(N)]:
            if i > j:
                gaussian_J[i][j], gaussian_J[j][i] = pair_dist.rvs()
    else:
        # require symmetric: use a different method
        A = normal.rvs(
            loc=0,
            scale=J**2 / N,
            size=(N, N))
        gaussian_J = (A + A.T)/2  # symmetrise

    return gaussian_J

def spherical_next_state(state, delta, J, T, N):
    """spherical_next_state: given previous state, J, and timestep,
    return next state in spherical model

    :param state: previous state
    :param J: interaction matrix
    :param T: temperature of the noise
    :param N: system size
    :param delta: timestep
    """

    return state + delta * \
            (-state/N * (state.T @ J @ state) + J @ state) + \
            normal(0, sqrt(2*T*delta)).rvs(N) @ (identity(N) - 1/N*outer(state, state))

def neural_next_state(state, delta, J, T, N, g=tanh):
    """neural_next_state: given previous state, J, and timestep,
    return next state in neural model

    :param state: previous state
    :param J: interaction matrix
    :param T: temperature of the noise
    :param N: system size
    :param delta: timestep
    """
    return state + delta *  (-state + J @ g(state)) + \
                   normal(0, sqrt(2*T*delta)).rvs(N)

def gen_trajectory(initial_state, steps, delta, J, T, N, system='spherical'):
    """gen_trajectory: generate a trajectory of length steps 

    :param initial_state: initial state of the simulation
    :param n_steps: number of steps to iterate
    :param delta: timestep
    :param traj_type: 'spherical'/'neural'
    """
    if system is 'spherical':
        next_state = spherical_next_state
    elif system is 'neural': 
        next_state = neural_next_state

    traj = empty((steps+1, N)) # Allocate empty trajectory
    traj[0] = initial_state
    for t in range(steps):
        traj[t+1] = array(next_state(traj[t], delta, J, T, N))

    return traj

### EXAMPLE

# define parameters
N = 20 
eta = 0.2
coupling = 4 # J
J = gen_gaussian_J(N, coupling, eta)
T = 0.01

# define initial state
X = gen_gaussian_spherical_initial_state(N)
assert isclose(norm(X)**2, N) # Spherical constraint

# start both trajectories from spherical initial condition
spherical_traj = gen_trajectory(X, 100, 0.01, J, T, N, 'spherical')
neural_traj    = gen_trajectory(X, 100, 0.01, J, T, N, 'neural')

fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(spherical_traj)
ax[0].set_title('spherical', loc='right')
ax[0].set_ylabel('$x_i$')

ax[1].plot(neural_traj)
ax[1].set_title('neural', loc='right')
ax[1].set_ylabel('$x_i$')
ax[1].set_xlabel('t')

fig.tight_layout()
plt.show()

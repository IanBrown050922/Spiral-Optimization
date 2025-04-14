import numpy as np

class SpiralOptimizer:
    '''
    Initializes the spiral optimizer.
    
    Parameters:
    objective_function (callable): The function to minimize. It accepts a scalar or numpy array and returns a scalar.
    dim (int): The dimensionality of the search space.
    bounds (tuple): A tuple (lower_bound, upper_bound) to randomly initialize all points.
    num_points (int): The number of candidate points used in the algorithm.
    r (float): Convergence rate. Each iteration, the distance from the center is scaled by r (0 < r < 1 for contraction).
    theta (float): Rotation rate in radians.
    rand_theta (boolean): Determines whether the rotation rates along each coordinate plane are random, or all equal.
    '''
    def __init__(self, objective_function, dim, bounds, num_points, r, theta, rand_theta=False):
        self.objective = objective_function
        self.dim = dim
        self.lower, self.upper = bounds
        self.num_points = num_points
        self.r = r

        # Store theta as a matrix, where theta_(i,j), the angle of rotation on the (i,j) plane,
        # is in position i,j (for simpler access).

        # The simplest implementation (used here and in the original paper) sets every plane's
        # rotation rate equal, so for all i, j: theta_(i,j) = theta.

        # We could also use random initialization of each theta_(i,j) if rand_theta=True:
        self.theta_matrix = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                if rand_theta:
                    self.theta_matrix[i, j] = np.random.uniform(-theta, theta)
                else:
                    self.theta_matrix[i, j] = theta

        # randomly initialize candidate points uniformly in (self.lower, self.upper)^self.dim
        self.points = np.random.uniform(self.lower, self.upper, (num_points, dim))
        # evaluate all initial points
        self.values = np.apply_along_axis(self.objective, 1, self.points)
        # Determine the current best candidate (which will be used as the first center of rotation)
        # and its function value.
        self.best_index = np.argmin(self.values)
        self.center = self.points[self.best_index].copy()
        self.best_value = self.values[self.best_index]


    '''
    Returns a matrix describing a rotation on the plane formed by two axes.

    Parameters:
    angle (float): angle of rotation.
    i (int): index of first axis
    j (int): index of second axis
    '''
    def basic_rotation_matrix(self, angle, i, j):
        # start with the identity matrix
        R = np.eye(self.dim)
        # modify the 2x2 submatrix corresponding to the rotation plane
        R[i, i] = np.cos(angle)
        R[i, j] = -np.sin(angle)
        R[j, i] = np.sin(angle)
        R[j, j] = np.cos(angle)
        return R

    '''
    Returns the rotation component of the Spiral Matrix described in Section III B. of
    'The Spiral Optimization and its stability analysis' (Kenichi Tamura, Keiichiro Yasuda)
    https://ieeexplore.ieee.org/document/6557686
    '''
    def spiral_matrix(self):
        R = np.eye(self.dim)
        for i in range(1, self.dim): # Product from i = 1 to n - 1
            for j in range(1, i + 1): # Product from j = 1 to i
                # axes defining the plane of rotation
                axis_1 = self.dim-i-1
                axis_2 = self.dim-j
                S = self.basic_rotation_matrix(self.theta_matrix[axis_1, axis_2], axis_1, axis_2)
                R = np.matmul(R, S)
        
        return R
                

    '''
    Performs Stochastic Spiral Optimization (SPO)
    https://ieeexplore.ieee.org/document/8261609

    Parameters:
    max_iterations: number of iterations to execute.
    '''
    def run(self, max_iterations):
        for _ in range(max_iterations):
            S = self.spiral_matrix() # compute the Spiral Matrix
            center = self.points[self.best_index] # center = x*
            new_points = np.empty_like(self.points) # create an array to store the new candidate points
            
            # update every point except the current best candidate (the center of rotation)
            for idx, point in enumerate(self.points): # point = x
                if idx == self.best_index:
                    new_points[idx] = point
                else:
                    # Apply the spiral transformation to point:
                    # 1. translate so that the center is at the origin
                    # 2. rotate and contract the point about the center
                    # 3. translate so that the center is in it's original location
                    
                    # Stochastic SPO uses a randomized convergence rate each iteration.
                    # r should be chosen from (r_l, 1) where 0 < r_l < 1.
                    # Here, we use r_l = r - (1 - r) = 2r - 1 so that the unform distribution
                    # for r is centered on self.r, where self.r is assumed to be the best value
                    # of r for the deterministic version of SPO.
                    rS = np.random.uniform(2*self.r - 1, 1) * S
                    # Stochastic SPO does elementwise product of (rS - I)x* with random vector u
                    u = np.random.uniform(0.0, 1.0, size=self.dim)
                    new_points[idx] = np.matmul(rS, point) - u * np.matmul((rS - np.eye(self.dim)), center)
            
            # evaluate the new candidate positions
            new_values = np.apply_along_axis(self.objective, 1, new_points)
            
            # update the set of points
            self.points = new_points
            self.values = new_values
            
            # update center to new best candidate point if applicable
            current_best_index = np.argmin(new_values)
            if new_values[current_best_index] < self.best_value:
                self.best_value = new_values[current_best_index]
                self.best_index = current_best_index
                self.center = new_points[current_best_index].copy()

        print(f'Minimizer found: {self.center}')
        print(f'Objective value: {self.best_value}')
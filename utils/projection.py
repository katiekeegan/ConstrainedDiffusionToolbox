import torch
import unittest

class SimpleConstraintProjector:
    def __init__(self):
        self.linear_equalities = []
        self.nonlinear_equalities = []
        self.linear_inequalities = []
        self.nonlinear_inequalities = []

    def add_linear_equality(self, A_eq, b_eq):
        """
        Add a linear equality constraint of the form Ax = b.
        A_eq: Tensor of shape (p, n)
        b_eq: Tensor of shape (p,)
        """
        self.linear_equalities.append((A_eq, b_eq))
    
    def add_nonlinear_equality(self, equality_func):
        """
        Add a nonlinear equality constraint.
        equality_func: A function that takes x and returns the value of the equality (should be 0 for feasible region).
        """
        self.nonlinear_equalities.append(equality_func)
    
    def add_linear_inequality(self, A_ineq, b_ineq):
        """
        Add a linear inequality constraint of the form Ax <= b.
        A_ineq: Tensor of shape (m, n)
        b_ineq: Tensor of shape (m,)
        """
        self.linear_inequalities.append((A_ineq, b_ineq))
    
    def add_nonlinear_inequality(self, inequality_func):
        """
        Add a nonlinear inequality constraint.
        inequality_func: A function that takes x and returns the value of the inequality (should be >= 0 for feasible region).
        """
        self.nonlinear_inequalities.append(inequality_func)

    def project_linear_inequality(self, x, A, b):
        """
        Projects a point onto the feasible region defined by Ax <= b.
        """
        direction = torch.linalg.lstsq(A, b - A @ x).solution
        return x + direction
    
    def project_nonlinear_inequality(self, x, constraint_func, step_size=1e-3, max_iter=100):
        """
        Projects a point onto the feasible region defined by a nonlinear inequality constraint using gradient descent.
        """
        x_proj = x.clone()
        for _ in range(max_iter):
            constraint_value = constraint_func(x_proj)
            if constraint_value >= 0:  # Already feasible
                return x_proj
            
            constraint_grad = torch.autograd.grad(constraint_value, x_proj)[0]
            x_proj = x_proj - step_size * constraint_grad
        return x_proj
    
    def project_linear_equality(self, x, A_eq, b_eq):
        """
        Projects a point onto the feasible region defined by Ax = b.
        """
        y = torch.linalg.lstsq(A_eq, b_eq).solution
        return y
    
    def project_nonlinear_equality(self, x, equality_func, step_size=1e-3, max_iter=100):
        """
        Projects a point onto the feasible region defined by a nonlinear equality constraint using gradient descent.
        """
        x_proj = x.clone()
        for _ in range(max_iter):
            equality_value = equality_func(x_proj)
            if torch.abs(equality_value) <= 1e-6:  # Feasible
                return x_proj
            
            equality_grad = torch.autograd.grad(equality_value, x_proj)[0]
            x_proj = x_proj - step_size * equality_grad
        return x_proj

    def project(self, x, step_size=1e-3, max_iter=100):
        """
        Projects a point x onto the feasible region defined by all linear and nonlinear constraints.
        """
        # Project onto all linear inequality constraints
        for A_ineq, b_ineq in self.linear_inequalities:
            x = self.project_linear_inequality(x, A_ineq, b_ineq)
        
        # Project onto all nonlinear inequality constraints
        for nonlinear_ineq_func in self.nonlinear_inequalities:
            x = self.project_nonlinear_inequality(x, nonlinear_ineq_func, step_size, max_iter)
        
        # Project onto all linear equality constraints
        for A_eq, b_eq in self.linear_equalities:
            x = self.project_linear_equality(x, A_eq, b_eq)
        
        # Project onto all nonlinear equality constraints
        for nonlinear_eq_func in self.nonlinear_equalities:
            x = self.project_nonlinear_equality(x, nonlinear_eq_func, step_size, max_iter)
        
        return x

class RiemannianManifoldConstraintOptimizer:
    def __init__(self, manifold_type, lr=1e-3, max_iter=100, tol=1e-6):
        """
        Riemannian Gradient Descent for manifold-constrained optimization
        Arguments:
            manifold_type: 'sphere' or 'stiefel'
            lr: learning rate
            max_iter: maximum iterations
            tol: tolerance for convergence
        """
        self.manifold = manifold_type
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

    def project_sphere(self, x):
        """
        Project a point onto the unit sphere.
        """
        norm = torch.norm(x)
        return x / norm  # Project onto the sphere (unit norm)

    def project_stiefel(self, X):
        """
        Project a matrix onto the Stiefel manifold (set of orthonormal matrices).
        Uses the Riemannian gradient descent technique with SVD.
        """
        U, _, V = torch.svd(X)
        return U @ V.t()

    def riemannian_gradient(self, f, x):
        """
        Compute the Riemannian gradient of function f at point x.
        For simplicity, this is done in Euclidean space for now.
        A more advanced version would require the specific manifold's geometry.
        """
        epsilon = 1e-6
        grad = torch.zeros_like(x)
        for i in range(len(x)):
            x_pos = x.clone()
            x_pos[i] += epsilon
            grad[i] = (f(x_pos) - f(x)) / epsilon
        return grad

    def exponential_map(self, x, v):
        """
        Perform the Riemannian exponential map to move along the manifold.
        """
        return x + self.lr * v  # Simple update rule (this could be manifold-specific)

    def riemannian_step(self, x, grad):
        """
        Perform one step of Riemannian gradient descent.
        """
        if self.manifold == "sphere":
            # Project gradient onto tangent space and apply the exponential map
            proj_grad = grad - torch.dot(grad, x) * x  # Tangent space projection
            return self.exponential_map(x, proj_grad)
        elif self.manifold == "stiefel":
            # Project gradient onto the Stiefel manifold (orthonormal matrices)
            proj_grad = grad - 2 * (grad @ grad.t()) @ x
            return self.exponential_map(x, proj_grad)
        else:
            raise ValueError(f"Unsupported manifold: {self.manifold}")

    def optimize(self, f, x_init):
        """
        Perform Riemannian gradient descent optimization.
        Arguments:
            f: objective function to minimize
            x_init: initial point (tensor)
        """
        x = x_init.clone().detach()
        for _ in range(self.max_iter):
            grad = self.riemannian_gradient(f, x)
            x_new = self.riemannian_step(x, grad)
            if torch.norm(x_new - x) < self.tol:
                print("Converged!")
                break
            x = x_new

        return x

# Example objective function: minimize the distance to a fixed point.
# For simplicity, this is just the squared Euclidean distance to the origin.
def objective_function(x):
    return torch.sum(x**2)

# Unit test
class TestRiemannianOptimizer(unittest.TestCase):
    def test_sphere_optimization(self):
        optimizer = RiemannianOptimizer(manifold_type='sphere', lr=1e-3)

        # Start at a random point on the sphere
        x_init = torch.randn(5)
        x_init = optimizer.project_sphere(x_init)  # Ensure initial point is on the sphere

        # Perform optimization
        x_opt = optimizer.optimize(objective_function, x_init)

        # Check if the optimized point is close to the origin (i.e., norm should be small)
        self.assertTrue(torch.norm(x_opt) < 1e-2, "Optimization did not converge to the correct solution.")

    def test_stiefel_optimization(self):
        optimizer = RiemannianOptimizer(manifold_type='stiefel', lr=1e-3)

        # Start with a random matrix
        X_init = torch.randn(5, 5)
        X_init = optimizer.project_stiefel(X_init)  # Ensure the initial matrix is orthonormal

        # Perform optimization
        X_opt = optimizer.optimize(objective_function, X_init)

        # Check if the optimized matrix is still orthonormal
        U, _, V = torch.svd(X_opt)
        self.assertTrue(torch.allclose(X_opt, U @ V.t()), "Optimization did not maintain orthonormality.")

class TestSimpleConstraintProjector(unittest.TestCase):
    def setUp(self):
        # Initialize the projector
        self.projector = SimpleConstraintProjector()

    def test_linear_equality(self):
        # Define a linear equality constraint Ax = b
        A_eq = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b_eq = torch.tensor([5.0, 6.0])
        
        # Add the linear equality constraint
        self.projector.add_linear_equality(A_eq, b_eq)
        
        # Initial point x
        x_init = torch.tensor([1.0, 1.0], requires_grad=True)
        
        # Project the point onto the feasible region of the linear equality constraint
        x_proj = self.projector.project(x_init)
        
        # Check if the projected point satisfies the equality constraint
        self.assertTrue(torch.allclose(A_eq @ x_proj, b_eq), "Linear equality projection failed.")
    
    def test_nonlinear_equality(self):
        # Define a nonlinear equality constraint f(x) = 0
        def equality_func(x):
            return x[0]**2 + x[1]**2 - 1  # Constraint: x lies on the unit circle
        
        # Add the nonlinear equality constraint
        self.projector.add_nonlinear_equality(equality_func)
        
        # Initial point x (off the unit circle)
        x_init = torch.tensor([1.5, 0.5], requires_grad=True)
        
        # Project the point onto the feasible region of the nonlinear equality constraint
        x_proj = self.projector.project(x_init)
        
        # Check if the projected point lies on the unit circle
        self.assertTrue(torch.abs(x_proj[0]**2 + x_proj[1]**2 - 1) < 1e-3, "Nonlinear equality projection failed.")
    
    def test_linear_inequality(self):
        # Define a linear inequality constraint Ax <= b
        A_ineq = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b_ineq = torch.tensor([1.0, 1.0])
        
        # Add the linear inequality constraint
        self.projector.add_linear_inequality(A_ineq, b_ineq)
        
        # Initial point x (outside the feasible region)
        x_init = torch.tensor([2.0, 2.0], requires_grad=True)
        
        # Project the point onto the feasible region of the linear inequality constraint
        x_proj = self.projector.project(x_init)
        
        # Check if the projected point satisfies the inequality constraint
        self.assertTrue(torch.all(x_proj <= b_ineq), "Linear inequality projection failed.")
    
    def test_nonlinear_inequality(self):
        # Define a nonlinear inequality constraint f(x) >= 0
        def inequality_func(x):
            return x[0]**2 + x[1]**2 - 1  # Constraint: x lies outside or on the unit circle
        
        # Add the nonlinear inequality constraint
        self.projector.add_nonlinear_inequality(inequality_func)
        
        # Initial point x (inside the feasible region)
        x_init = torch.tensor([0.5, 0.5], requires_grad=True)
        
        # Project the point onto the feasible region of the nonlinear inequality constraint
        x_proj = self.projector.project(x_init)
        
        # Check if the projected point satisfies the inequality constraint
        self.assertTrue(x_proj[0]**2 + x_proj[1]**2 - 1 >= 0, "Nonlinear inequality projection failed.")
    
    def test_combined_constraints(self):
        # Define some combined constraints (both linear and nonlinear)
        
        # Linear equality constraint: x1 + x2 = 1
        A_eq = torch.tensor([[1.0, 1.0]])
        b_eq = torch.tensor([1.0])
        self.projector.add_linear_equality(A_eq, b_eq)
        
        # Nonlinear inequality constraint: x1^2 + x2^2 >= 1 (x lies outside or on the unit circle)
        def inequality_func(x):
            return x[0]**2 + x[1]**2 - 1
        self.projector.add_nonlinear_inequality(inequality_func)
        
        # Initial point x (which violates both constraints)
        x_init = torch.tensor([0.5, 0.5], requires_grad=True)
        
        # Project the point onto the feasible region defined by all constraints
        x_proj = self.projector.project(x_init)
        
        # Check if the projected point satisfies both constraints
        self.assertTrue(torch.allclose(A_eq @ x_proj, b_eq), "Linear equality constraint not satisfied.")
        self.assertTrue(x_proj[0]**2 + x_proj[1]**2 - 1 >= 0, "Nonlinear inequality constraint not satisfied.")

if __name__ == "__main__":
    unittest.main() 
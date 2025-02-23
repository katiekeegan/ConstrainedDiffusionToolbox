import torch
import unittest
import numpy as np


class SimpleConstraintProjector:
    def __init__(self):
        self.linear_equalities = []
        self.nonlinear_equalities = []
        self.linear_inequalities = []
        self.nonlinear_inequalities = []

    def add_linear_equality(self, A_eq, b_eq):
        self.linear_equalities.append((A_eq, b_eq))

    def add_nonlinear_equality(self, equality_func):
        self.nonlinear_equalities.append(equality_func)

    def add_linear_inequality(self, A_ineq, b_ineq):
        self.linear_inequalities.append((A_ineq, b_ineq))

    def add_nonlinear_inequality(self, inequality_func):
        self.nonlinear_inequalities.append(inequality_func)

    def add_constraints_from_dict(self, constraints_dict):
        """
        Add constraints from a dictionary.

        Parameters:
        -----------
        constraints_dict : dict
            A dictionary where the keys are constraint types and the values are lists of constraints.
            The keys should be one of the following:
                - "linear_equality"
                - "nonlinear_equality"
                - "linear_inequality"
                - "nonlinear_inequality"
            The values should be lists of tuples or functions, depending on the constraint type:
                - For linear constraints: tuples of (A, b)
                - For nonlinear constraints: functions
        """
        for constraint_type, constraints in constraints_dict.items():
            if constraint_type == "linear_equality":
                for A_eq, b_eq in constraints:
                    self.add_linear_equality(A_eq, b_eq)
            elif constraint_type == "nonlinear_equality":
                for equality_func in constraints:
                    self.add_nonlinear_equality(equality_func)
            elif constraint_type == "linear_inequality":
                for A_ineq, b_ineq in constraints:
                    self.add_linear_inequality(A_ineq, b_ineq)
            elif constraint_type == "nonlinear_inequality":
                for inequality_func in constraints:
                    self.add_nonlinear_inequality(inequality_func)
            else:
                raise ValueError(f"Unknown constraint type: {constraint_type}")

    def project_linear_inequality(self, x, A, b):
        direction = torch.linalg.lstsq(A, b - A @ x).solution
        normal = A  # Normal is the rows of A
        return x + direction, torch.norm(direction), normal

    def project_nonlinear_equality(
        self, x, equality_func, step_size=1e-3, max_iter=10000, tol=1e-3
    ):
        x_proj = x.clone().requires_grad_(True)  # Enable gradient tracking
        prev_residual = float("inf")  # Track previous residual for convergence check

        for _ in range(max_iter):
            equality_value = equality_func(x_proj)
            residual = torch.abs(equality_value)

            # Check for convergence
            if residual <= tol:
                return x_proj.detach(), residual.item(), None

            # Check if residual is increasing (diverging)
            if residual >= prev_residual:
                step_size *= 0.5  # Reduce step size if not converging
                if step_size < 1e-10:  # Prevent step size from becoming too small
                    break

            prev_residual = residual

            # Compute gradient and update x_proj
            equality_grad = torch.autograd.grad(
                equality_value, x_proj, create_graph=True
            )[0]
            x_proj = x_proj - step_size * equality_grad

        # If max_iter is reached, return the best solution found
        residual = torch.abs(equality_func(x_proj))
        normal = torch.autograd.grad(equality_func(x_proj), x_proj)[
            0
        ]  # Gradient as normal
        return x_proj.detach(), residual.item(), normal

    def project_linear_equality(self, x, A_eq, b_eq):
        y = torch.linalg.lstsq(A_eq, b_eq).solution
        residual = torch.norm(A_eq @ y - b_eq)
        normal = A_eq  # Normal is the rows of A_eq
        return y, residual, normal

    def project_nonlinear_inequality(
        self, x, constraint_func, step_size=1e-3, max_iter=1000
    ):
        x_proj = x.clone().requires_grad_(True)  # Enable gradient tracking
        for _ in range(max_iter):
            constraint_value = constraint_func(x_proj)  # Evaluate g(x)

            # Check if constraint is satisfied
            if constraint_value <= 0:  # Correct condition: g(x) <= 0
                return x_proj.detach(), constraint_value.item(), None

            # If constraint is violated, adjust x_proj using gradient descent
            constraint_grad = torch.autograd.grad(
                constraint_value, x_proj, create_graph=True
            )[0]
            x_proj = x_proj - step_size * constraint_grad

        # If max_iter is reached, return the final projection
        residual = constraint_func(x_proj)
        normal = torch.autograd.grad(constraint_func(x_proj), x_proj)[
            0
        ]  # Gradient as normal
        return x_proj.detach(), residual.item(), normal

    def project(self, x, step_size=1e-3, max_iter=100, return_residual=True):
        norm_residual = 0
        normals = []

        for A_ineq, b_ineq in self.linear_inequalities:
            x, residual, normal = self.project_linear_inequality(x, A_ineq, b_ineq)
            norm_residual += residual
            if normal is not None:
                normals.append(normal)

        for nonlinear_ineq_func in self.nonlinear_inequalities:
            x, residual, normal = self.project_nonlinear_inequality(
                x, nonlinear_ineq_func, step_size, max_iter
            )
            norm_residual += residual
            if normal is not None:
                normals.append(normal)

        for A_eq, b_eq in self.linear_equalities:
            x, residual, normal = self.project_linear_equality(x, A_eq, b_eq)
            norm_residual += residual
            if normal is not None:
                normals.append(normal)

        for nonlinear_eq_func in self.nonlinear_equalities:
            x, residual, normal = self.project_nonlinear_equality(
                x, nonlinear_eq_func, step_size, max_iter
            )
            norm_residual += residual
            if normal is not None:
                normals.append(normal)

        if return_residual:
            return x, norm_residual, normals
        return x


class TestSimpleConstraintProjector(unittest.TestCase):
    def setUp(self):
        self.projector = SimpleConstraintProjector()

    def test_add_linear_equality(self):
        A_eq = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b_eq = torch.tensor([1.0, 1.0])
        self.projector.add_linear_equality(A_eq, b_eq)
        self.assertEqual(len(self.projector.linear_equalities), 1)
        self.assertTrue(torch.equal(self.projector.linear_equalities[0][0], A_eq))
        self.assertTrue(torch.equal(self.projector.linear_equalities[0][1], b_eq))

    def test_add_nonlinear_equality(self):
        def equality_func(x):
            return torch.sum(x**2) - 1.0

        self.projector.add_nonlinear_equality(equality_func)
        self.assertEqual(len(self.projector.nonlinear_equalities), 1)

    def test_add_linear_inequality(self):
        A_ineq = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b_ineq = torch.tensor([1.0, 1.0])
        self.projector.add_linear_inequality(A_ineq, b_ineq)
        self.assertEqual(len(self.projector.linear_inequalities), 1)
        self.assertTrue(torch.equal(self.projector.linear_inequalities[0][0], A_ineq))
        self.assertTrue(torch.equal(self.projector.linear_inequalities[0][1], b_ineq))

    def test_add_nonlinear_inequality(self):
        def inequality_func(x):
            return torch.sum(x**2) - 1.0

        self.projector.add_nonlinear_inequality(inequality_func)
        self.assertEqual(len(self.projector.nonlinear_inequalities), 1)

    def test_project_linear_inequality(self):
        A_ineq = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b_ineq = torch.tensor([1.0, 1.0])
        x = torch.tensor([2.0, 2.0])
        x_proj, residual, normal = self.projector.project_linear_inequality(
            x, A_ineq, b_ineq
        )
        self.assertTrue(torch.allclose(A_ineq @ x_proj, b_ineq, atol=1e-6))

    def test_project_nonlinear_inequality(self):
        def inequality_func(x):
            return torch.sum(x**2) - 1.0

        x = torch.tensor([2.0, 2.0])
        x_proj, residual, normal = self.projector.project_nonlinear_inequality(
            x, inequality_func
        )
        self.assertLessEqual(inequality_func(x_proj).item(), 1e-6)

    def test_project_linear_equality(self):
        A_eq = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b_eq = torch.tensor([1.0, 1.0])
        x = torch.tensor([2.0, 2.0])
        x_proj, residual, normal = self.projector.project_linear_equality(x, A_eq, b_eq)
        self.assertTrue(torch.allclose(A_eq @ x_proj, b_eq, atol=1e-6))

    def test_project_nonlinear_equality(self):
        def equality_func(x):
            return torch.sum(x**2) - 1.0

        x = torch.tensor([1.0, 1.0])
        x_proj, residual, normal = self.projector.project_nonlinear_equality(
            x, equality_func
        )
        self.assertLessEqual(torch.abs(equality_func(x_proj)).item(), 1e-3)

    def test_project(self):
        A_eq = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b_eq = torch.tensor([1.0, 1.0])
        self.projector.add_linear_equality(A_eq, b_eq)
        x = torch.tensor([2.0, 2.0])
        x_proj = self.projector.project(x)
        self.assertTrue(torch.allclose(A_eq @ x_proj, b_eq, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

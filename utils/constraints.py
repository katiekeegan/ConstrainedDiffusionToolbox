import torch
import unittest
import numpy as np

class SimpleConstraintProjector:
    def __init__(self):
        self.linear_equalities = []
        self.nonlinear_equalities = []
        self.linear_inequalities = []
        self.nonlinear_inequalities = []
        self.sphere_constraints = []  # New list for sphere constraints

    def add_sphere_constraint(self, center, radius):
        self.sphere_constraints.append((center, radius))

    def add_linear_equality(self, A_eq, b_eq):
        self.linear_equalities.append((A_eq, b_eq))

    def add_nonlinear_equality(self, equality_func):
        self.nonlinear_equalities.append(equality_func)

    def add_linear_inequality(self, A_ineq, b_ineq):
        self.linear_inequalities.append((A_ineq, b_ineq))

    def add_nonlinear_inequality(self, inequality_func):
        self.nonlinear_inequalities.append(inequality_func)

    def add_constraints_from_dict(self, constraints_dict):
        for constraint_type, constraints in constraints_dict.items():
            if constraint_type == "linear_equality":
                A_eq, b_eq = constraints
                self.add_linear_equality(A_eq, b_eq)
            elif constraint_type == "nonlinear_equality":
                equality_func = constraints
                self.add_nonlinear_equality(equality_func)
            elif constraint_type == "linear_inequality":
                A_ineq, b_ineq = constraints
                self.add_linear_inequality(A_ineq, b_ineq)
            elif constraint_type == "nonlinear_inequality":
                inequality_func = constraints
                self.add_nonlinear_inequality(inequality_func)
            else:
                raise ValueError(f"Unknown constraint type: {constraint_type}")

    def project_linear_inequality(self, x, A, b):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Solve for direction to satisfy A(x + dir) >= b
        solution = torch.linalg.lstsq(A, (b - A @ x.T).T).solution  # (batch_size, d)
        x_proj = x + solution
        residual = torch.norm(A @ x_proj.T - b.unsqueeze(-1), dim=0)
        return x_proj, residual.mean().item(), A

    def project_nonlinear_equality(
        self, x, equality_func, step_size=1e-3, max_iter=10000, tol=1e-3
    ):
        x_proj = x.clone().requires_grad_(True)
        prev_residual = float("inf")

        for _ in range(max_iter):
            equality_value = equality_func(x_proj)
            residual = torch.abs(equality_value)

            if torch.all(residual <= tol):
                return x_proj.detach(), residual.mean().item(), None

            if torch.any(residual >= prev_residual):
                step_size *= 0.5
                if step_size < 1e-10:
                    break

            prev_residual = residual

            equality_grad = torch.autograd.grad(
                equality_value.sum(), x_proj, create_graph=True
            )[0]
            x_proj = x_proj - step_size * equality_grad

        residual = torch.abs(equality_func(x_proj))
        normal = torch.autograd.grad(equality_func(x_proj).sum(), x_proj)[0]
        return x_proj.squeeze().detach(), residual.mean().item(), normal


    def project_linear_equality(self, x, A_eq, b_eq):
        x = x.squeeze()
        # Ensure x is 2D: (batch_size, d)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        # A_eq: (m, d), b_eq: (m,)
        # Compute direction: minimal norm adjustment for Ax = b
        A_broadcasted = A_eq.unsqueeze(0).expand(x.size(0),-1,-1)
        b_broadcasted = b_eq.unsqueeze(0).expand(x.size(0), -1).unsqueeze(-1)
        print(f"A broadcasted: {A_broadcasted.size()}")
        print(f"b broadcasted: {b_broadcasted.size()}")
        # Compute (Ax - b)
        Ax_minus_b = torch.matmul(A_broadcasted, x) - b_broadcasted  # Shape: (batch_size, 1)

        # Compute (A A^T)^(-1)
        AA_T = torch.matmul(A_broadcasted, torch.transpose(A_broadcasted,1,-1))  # Shape: (1, 1)
        AA_T_inv = torch.linalg.pinv(AA_T)  # Shape: (1, 1)

        print(f'AA_T_inv {AA_T_inv.size()}')
        print(f'Ax_minus_b {Ax_minus_b.size()}')
        # Compute the correction term: (A A^T)^(-1) (Ax - b)
        correction = torch.matmul(AA_T_inv, Ax_minus_b)  # Shape: (batch_size, 1)
        term = torch.matmul(torch.transpose(A_broadcasted,1,-1), correction)
        x_proj = x - term
        residual = torch.norm(term)
        return x_proj, residual.mean().item(), A_eq

    def project_nonlinear_inequality(
        self, x, constraint_func, step_size=1e-3, max_iter=1000
    ):
        x_proj = x.clone().requires_grad_(True)
        for _ in range(max_iter):
            constraint_value = constraint_func(x_proj)

            if torch.all(constraint_value <= 0):
                return x_proj.detach(), constraint_value.mean().item(), None

            constraint_grad = torch.autograd.grad(
                constraint_value.sum(), x_proj, create_graph=True
            )[0]
            x_proj = x_proj - step_size * constraint_grad

        residual = constraint_func(x_proj)
        normal = torch.autograd.grad(constraint_func(x_proj).sum(), x_proj)[0]
        return x_proj.detach(), residual.mean().item(), normal

    def project(self, x, step_size=1e-3, max_iter=100, return_residual=True):
        norm_residual = 0
        normals = []

        if x.dim() == 1:
            x = x.unsqueeze(0)
            return_single = True
        else:
            return_single = False

        # Process linear inequalities
        for A_ineq, b_ineq in self.linear_inequalities:
            x, residual, normal = self.project_linear_inequality(x, A_ineq, b_ineq)
            norm_residual += residual.mean().item()
            if normal is not None:
                normals.append(normal)

        # Process nonlinear inequalities
        for nonlinear_ineq_func in self.nonlinear_inequalities:
            x, residual, normal = self.project_nonlinear_inequality(
                x, nonlinear_ineq_func, step_size, max_iter
            )
            norm_residual += residual
            if normal is not None:
                normals.append(normal)

        # Process sphere constraints
        for center, radius in self.sphere_constraints:
            delta = x - center.unsqueeze(0)  # (batch_size, d)
            norm = torch.norm(delta, dim=1, keepdim=True)  # (batch_size, 1)
            if torch.any(norm == 0):
                if radius == 0:
                    x_proj = center.unsqueeze(0).expand_as(x)
                else:
                    delta = torch.ones_like(x)
                    norm_delta = torch.norm(delta, dim=1, keepdim=True)
                    if torch.any(norm_delta == 0):
                        delta = torch.ones_like(x)
                        norm_delta = torch.sqrt(torch.tensor(x.size(1), dtype=delta.dtype))
                    delta = delta / norm_delta * radius
                    x_proj = center.unsqueeze(0) + delta
            else:
                x_proj = center.unsqueeze(0) + delta * (radius / norm)
            residual = torch.abs(torch.norm(x_proj - center.unsqueeze(0), dim=1) - radius)
            norm_residual += residual.mean().item()
            normal = (x_proj - center.unsqueeze(0)) / radius if radius != 0 else torch.zeros_like(x)
            normals.append(normal)
            x = x_proj

        # Process linear equalities
        for A_eq, b_eq in self.linear_equalities:
            x, residual, normal = self.project_linear_equality(x, A_eq, b_eq)
            norm_residual += residual
            if normal is not None:
                normals.append(normal)

        # Process nonlinear equalities
        for nonlinear_eq_func in self.nonlinear_equalities:
            x, residual, normal = self.project_nonlinear_equality(
                x, nonlinear_eq_func, step_size, max_iter
            )
            norm_residual += residual
            if normal is not None:
                normals.append(normal)
        if return_single:
            x = x.squeeze(0)
        if return_residual:
            return x, norm_residual, normals
        return x


class TestSimpleConstraintProjector(unittest.TestCase):
    def setUp(self):
        self.projector = SimpleConstraintProjector()

    def test_add_sphere_constraint(self):
        center = torch.tensor([1.0, 1.0])
        radius = 3.0
        self.projector.add_sphere_constraint(center, radius)
        self.assertEqual(len(self.projector.sphere_constraints), 1)
        self.assertTrue(torch.equal(self.projector.sphere_constraints[0][0], center))
        self.assertEqual(self.projector.sphere_constraints[0][1], radius)

    def test_project_sphere_constraint(self):
        center = torch.tensor([0.0, 0.0])
        radius = 2.0
        self.projector.add_sphere_constraint(center, radius)
        x = torch.tensor([[3.0, 4.0], [1.0, 1.0]])  # Norm 5 and sqrt(2)
        x_proj = self.projector.project(x, return_residual=False)
        expected = torch.tensor([[3.0/5*2, 4.0/5*2], [1.0/np.sqrt(2)*2, 1.0/np.sqrt(2)*2]])
        self.assertTrue(torch.allclose(x_proj, expected, atol=1e-6))

    def test_project_sphere_constraint_at_center(self):
        center = torch.tensor([1.0, 1.0])
        radius = 3.0
        self.projector.add_sphere_constraint(center, radius)
        x = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        x_proj = self.projector.project(x, return_residual=False)
        expected_norm = radius
        computed_norm = torch.norm(x_proj - center.unsqueeze(0), dim=1)
        self.assertTrue(torch.allclose(computed_norm, torch.tensor([expected_norm, expected_norm]), delta=1e-6))


if __name__ == "__main__":
    unittest.main()
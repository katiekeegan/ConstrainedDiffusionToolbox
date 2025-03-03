import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.constraints import SimpleConstraintProjector
import unittest

class PointDataset(Dataset):
    """ Custom dataset for 2D points. """
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SmileyFaceDataset(Dataset):
    def __init__(self, num_samples=1000, constraint_projector=None, A=None, b=0.0, 
                 sphere_center=None, sphere_radius=None, noise_level=1, example=False, 
                 projection_step_size=1e-3, projection_max_iter=100, projection_type="plane"):
        self.num_samples = num_samples
        self.noise_level = noise_level
        self.example = example
        self.projection_step_size = projection_step_size
        self.projection_max_iter = projection_max_iter
        self.projection_type = projection_type  # Option for plane or sphere

        # Set up the constraint projector
        if constraint_projector is not None:
            self.constraint_projector = constraint_projector
        else:
            self.constraint_projector = SimpleConstraintProjector()
            if A is None:
                A = np.array([1.0, 2.0, 3.0])  # Default normal vector
            A_norm = A / np.linalg.norm(A)
            A_tensor = torch.tensor(A_norm, dtype=torch.float32)
            b_tensor = torch.tensor([b], dtype=torch.float32)
            self.constraint_projector.add_linear_equality(A_tensor, b_tensor)

        # Optionally handle sphere parameters
        self.sphere_center = sphere_center if sphere_center is not None else np.array([0.0, 0.0, 0.0])
        self.sphere_radius = sphere_radius if sphere_radius is not None else 2.0
        
        self.data = self._generate_smiley()

    def _project_onto_sphere(self, point):
        """ Project a point onto the surface of the sphere. """
        point_normalized = point / np.linalg.norm(point)
        return self.sphere_center + self.sphere_radius * point_normalized

    def _generate_smiley(self):
        samples = []

        # Generate circle outline (face)
        angles = np.linspace(0, 2 * np.pi, self.num_samples // 2)
        face_x = 1.5 * np.cos(angles)
        face_y = 1.5 * np.sin(angles)

        # Generate eyes
        eye_x = np.concatenate([
            np.random.normal(-0.8, 0.1, self.num_samples // 8),
            np.random.normal(0.8, 0.1, self.num_samples // 8)
        ])
        eye_y = np.concatenate([
            np.random.normal(0.8, 0.1, self.num_samples // 8),
            np.random.normal(0.8, 0.1, self.num_samples // 8)
        ])

        # Generate mouth (smile arc)
        mouth_angles = np.linspace(-np.pi / 4, np.pi / 4, self.num_samples // 4)
        mouth_x = 0.8 * np.cos(mouth_angles)
        mouth_y = 0.6 * np.sin(mouth_angles)

        # Rotate mouth 90 degrees clockwise
        mouth_x, mouth_y = mouth_y, -mouth_x

        # Combine all 2D points
        all_x = np.concatenate([face_x, eye_x, mouth_x])
        all_y = np.concatenate([face_y, eye_y, mouth_y])

        # if self.projection_type == "sphere":
        #     # Shift the smiley face upwards along the y-axis (to avoid points going inside the sphere)
        #     shift_y = 3.0  # Shift value, can be adjusted for better positioning
        #     all_y += shift_y

        # Generate 3D points and project
        for x, y in zip(all_x, all_y):

            if self.projection_type == "sphere":
                # Convert 2D face coordinates to spherical coordinates (θ, φ)
                theta = np.arctan2(y, x)  # Azimuthal angle in xy-plane
                r = np.sqrt(x**2 + y**2)  # Radial distance in xy-plane (within face)
                phi = np.arcsin(np.clip(r / 2.0, -1, 1))  # Polar angle from the z-axis

                # Convert spherical to Cartesian coordinates
                sample_3d = np.array([
                    self.sphere_radius * np.sin(phi) * np.cos(theta),
                    self.sphere_radius * np.sin(phi) * np.sin(theta),
                    self.sphere_radius * np.cos(phi)
                ])

            # If projecting onto a plane, use the constraint projector
            if self.projection_type == "plane":
                sample_3d = np.array([x, y, 0])
                sample_tensor = torch.tensor(sample_3d, dtype=torch.float32)
                sample_3d = self.constraint_projector.project(
                    sample_tensor,
                    step_size=self.projection_step_size,
                    max_iter=self.projection_max_iter,
                    return_residual=False
                ).squeeze().numpy()

            if self.example:
                # Add Gaussian noise uniformly to all points strictly in the orthogonal direction to the constraint
                normal = None
                if self.constraint_projector.linear_equalities:
                    # Linear equality constraint: normal is A_eq
                    A_eq = self.constraint_projector.linear_equalities[0][0].numpy()#.flatten()
                    normal = A_eq
                    # Generate a random noise vector in 3D
                    noise = self.noise_level * np.random.randn((3))
                    
                    noise_orth = np.dot(noise, normal) * normal  # Radial noise component
                    sample_3d += noise_orth
                elif self.constraint_projector.sphere_constraints:
                    normal = sample_3d/np.linalg.norm(sample_3d)
                    # Generate a random noise vector in 3D
                    noise = self.noise_level * (2*np.random.randn(3)-1)  # Uniform random noise
                    
                    # The noise should be in the direction of the radial vector, so add this noise along the normal direction
                    noise_orth = np.dot(noise, normal) * normal  # Radial noise component

                    # Add the orthogonal (radial) noise to the sample
                    sample_3d += noise_orth


            samples.append(sample_3d)

        return torch.tensor(samples, dtype=torch.float32).squeeze()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

class TestSmileyFaceDataset(unittest.TestCase):
    def setUp(self):
        # Set up a constraint projector with a simple linear equality constraint
        self.constraint_projector = SimpleConstraintProjector()
        A = np.array([1.0, 2.0, 3.0])  # Normal vector for the constraint
        A_norm = A / np.linalg.norm(A)
        A_tensor = torch.tensor(A_norm.reshape(1, -1), dtype=torch.float32)
        b_tensor = torch.tensor([0.0], dtype=torch.float32)
        self.constraint_projector.add_linear_equality(A_tensor, b_tensor)

        # Initialize the dataset with the constraint projector
        self.dataset = SmileyFaceDataset(
            num_samples=1000,
            constraint_projector=self.constraint_projector,
            noise_level=10,
            example=False,
            projection_step_size=1e-3,
            projection_max_iter=100,
            projection_type="sphere"  # Change this to "plane" for plane projection
        )

    def test_dataset_length(self):
        self.assertEqual(len(self.dataset), 1000)

    def test_dataset_item_shape(self):
        sample = self.dataset[0]
        self.assertEqual(sample.shape, torch.Size([3]))

    # def test_constraint_satisfaction(self):
    #     A = self.constraint_projector.linear_equalities[0][0].numpy().flatten()
    #     b = self.constraint_projector.linear_equalities[0][1].numpy()

    #     for i in range(len(self.dataset)):
    #         sample = self.dataset[i].numpy()

    #         # Ensure sample is a 1D vector
    #         if sample.ndim == 1:
    #             dot_product = np.dot(A, sample)
                
    #             # Use a small tolerance to compare the dot product with b
    #             self.assertTrue(np.abs(dot_product - b) < 1e-4, 
    #                             f"Dot product {dot_product} differs from b={b} at sample {i}")
    #         else:
    #             raise ValueError(f"Sample {i} is not a 1D vector.")

    def test_data_loader(self):
        data_loader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        for batch in data_loader:
            self.assertEqual(batch.shape, torch.Size([32, 3]))
            break

    def test_example_mode(self):
        example_dataset = SmileyFaceDataset(
            num_samples=1000,
            constraint_projector=self.constraint_projector,
            noise_level=10,
            example=True,
            projection_step_size=1e-3,
            projection_max_iter=100,
            projection_type="sphere"
        )
        self.assertEqual(len(example_dataset), 1000)
        sample = example_dataset[0]
        self.assertEqual(sample.shape, torch.Size([3]))

if __name__ == '__main__':
    unittest.main()

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from scipy.stats import gaussian_kde


def plot_points_3d(points_list, A=None, b=None, sphere_center=None, sphere_radius=None,resolution = 100,  plot_density=False,labels=['Generated Points', 'Data']):
    """
    Plot the points in 3D and include the plane defined by Ax = b as a visualization using Plotly.
    """
    fig = go.Figure()

    for i, points in enumerate(points_list):
        # Plot the generated points
        points_density = None
        if plot_density:
            points_density = gaussian_kde(np.transpose(points))(np.transpose(points))
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=5,  color=points_density if points_density else 'blue', colorscale='Viridis', opacity=0.7),
            name=labels[i]
        ))
    # Plot the plane
    if A is not None and b is not None:
        xx, yy = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
        zz = (b - A[0] * xx - A[1] * yy) / A[2]  # Solve for z in Ax + By + Cz = b

        fig.add_trace(go.Surface(
            x=xx, y=yy, z=zz,
            colorscale='Reds',
            opacity=0.5,
            name='Plane'
        ))
    if sphere_center is not None and sphere_radius is not None:
        u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
        xx = sphere_radius * np.cos(u)*np.sin(v) + sphere_center[0]
        yy = sphere_radius * np.sin(u)*np.sin(v) + sphere_center[1]
        zz = sphere_radius * np.cos(v) + sphere_center[2]
        fig.add_trace(go.Surface(
            x=xx, y=yy, z=zz,
            colorscale='Reds',
            opacity=0.5,
            name='Sphere'
        ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    return fig

def combine_figures(figures, rows, cols, labels):
    """
    Combines multiple Plotly figures into a single figure with subplots.
    """
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"{labels[i]}" for i in range(len(figures))],
                        specs=[[{'type': 'surface'}]*cols]*rows)

    for i, f in enumerate(figures):
        row = i // cols + 1
        col = i % cols + 1
        for trace in f.data:
          if hasattr(trace, 'showscale'):
              trace.update(showscale=False)
          fig.add_trace(trace, row=row, col=col)

    fig.update_layout(height=600, width=900, showlegend=False)

    return fig
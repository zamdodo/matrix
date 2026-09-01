"""Unit tests for SurfaceMesher."""

import numpy as np
import pytest

from src.reconstruction.geometry.mesher import SurfaceMesher
from src.reconstruction.models.s3_output import MeshData, PointCloudData


def test_surface_mesher_basic_triangle():
    # 3 non-collinear points in XY plane
    points = np.array([
        [0.0, 0.0, 1.0],
        [5.0, 0.0, 2.0],
        [0.0, 5.0, 1.5],
    ], dtype=np.float64)
    colors = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
    ], dtype=np.uint8)

    cloud = PointCloudData(points=points, colors=colors)
    mesher = SurfaceMesher()
    mesh = mesher.generate_mesh(cloud)

    assert isinstance(mesh, MeshData)
    assert mesh.num_vertices == 3
    assert mesh.num_faces >= 1
    assert mesh.normals is not None
    assert mesh.normals.shape == (3, 3)
    # Check that normal Z component is positive (pointing up)
    assert np.all(mesh.normals[:, 2] >= 0.0)


def test_surface_mesher_insufficient_points():
    points = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    cloud = PointCloudData(points=points)
    mesher = SurfaceMesher()
    mesh = mesher.generate_mesh(cloud)
    assert mesh is None


"""
S3 Surface Reconstruction and Meshing

Generates 3D triangular surface meshes from point clouds with normal estimation
and edge-length filtering.
"""

from typing import Optional, Tuple
import numpy as np

from ..models.s3_output import MeshData, PointCloudData


class SurfaceMesher:
    """
    Constructs 3D surface meshes from point cloud geometry.
    """

    def __init__(
        self,
        max_edge_length: Optional[float] = None,
        edge_length_factor: float = 3.0,
    ) -> None:
        """
        Initialize surface mesher.

        Parameters:
            max_edge_length: Hard cutoff distance (m) above which triangle edges are pruned.
            edge_length_factor: Multiplier of median nearest-neighbor distance for adaptive edge pruning.
        """
        self.max_edge_length = max_edge_length
        self.edge_length_factor = float(edge_length_factor)

    def generate_mesh(self, point_cloud: PointCloudData) -> Optional[MeshData]:
        """
        Generate a 3D surface mesh from a point cloud.

        Parameters:
            point_cloud: Reconstructed PointCloudData.

        Returns:
            MeshData if mesh generation succeeds; None if points are insufficient (< 3).
        """
        pts = point_cloud.points
        n_pts = pts.shape[0]

        if n_pts < 3:
            return None

        # 1. 2D Delaunay Triangulation on the primary spatial projection plane (XY for aerial UAV)
        faces = self._delaunay_2d(pts[:, :2])
        if len(faces) == 0:
            return None

        # 2. Filter triangles with excessively long edges
        v0 = pts[faces[:, 0]]
        v1 = pts[faces[:, 1]]
        v2 = pts[faces[:, 2]]

        e01 = np.linalg.norm(v1 - v0, axis=1)
        e12 = np.linalg.norm(v2 - v1, axis=1)
        e20 = np.linalg.norm(v0 - v2, axis=1)
        max_edges = np.maximum(np.maximum(e01, e12), e20)

        cutoff = self.max_edge_length
        if cutoff is None:
            med_edge = float(np.median(max_edges))
            cutoff = max(med_edge * self.edge_length_factor, 1.0)

        valid_face_mask = max_edges <= cutoff
        filtered_faces = faces[valid_face_mask]

        if len(filtered_faces) == 0:
            # Fallback: keep all faces if filter is too aggressive
            filtered_faces = faces

        # 3. Compute surface normals per vertex
        normals = self._compute_vertex_normals(pts, filtered_faces)

        # 4. Color assignment
        vertex_colors = point_cloud.colors

        return MeshData(
            vertices=pts,
            faces=filtered_faces,
            normals=normals,
            vertex_colors=vertex_colors,
        )

    @staticmethod
    def _delaunay_2d(points_2d: np.ndarray) -> np.ndarray:
        """
        Compute 2D Delaunay triangulation for (N, 2) points.
        Uses scipy.spatial.Delaunay if available; otherwise uses a robust Bowyer-Watson / incremental Delaunay.
        """
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(points_2d)
            return np.asarray(tri.simplices, dtype=np.int64)
        except ImportError:
            return SurfaceMesher._delaunay_pure_numpy(points_2d)

    @staticmethod
    def _delaunay_pure_numpy(points: np.ndarray) -> np.ndarray:
        """
        Pure NumPy incremental 2D Delaunay Triangulation fallback.
        """
        n_points = len(points)
        if n_points < 3:
            return np.empty((0, 3), dtype=np.int64)

        # Super-triangle bounding all points
        min_xy = np.min(points, axis=0) - 100.0
        max_xy = np.max(points, axis=0) + 100.0
        d_xy = max_xy - min_xy
        delta = max(d_xy[0], d_xy[1]) * 10.0

        p1 = [min_xy[0] - delta, min_xy[1] - delta]
        p2 = [min_xy[0] + delta * 2, min_xy[1] - delta]
        p3 = [min_xy[0] - delta, min_xy[1] + delta * 2]

        all_pts = np.vstack([points, [p1, p2, p3]])
        st_idx = [n_points, n_points + 1, n_points + 2]

        triangles = [st_idx]

        for i in range(n_points):
            pt = all_pts[i]
            bad_triangles = []

            for tri in triangles:
                # Check if point is in circumcircle
                a, b, c = all_pts[tri[0]], all_pts[tri[1]], all_pts[tri[2]]
                ax, ay = a[0] - pt[0], a[1] - pt[1]
                bx, by = b[0] - pt[0], b[1] - pt[1]
                cx, cy = c[0] - pt[0], c[1] - pt[1]

                det = (
                    (ax * ax + ay * ay) * (bx * cy - cx * by)
                    - (bx * bx + by * by) * (ax * cy - cx * ay)
                    + (cx * cx + cy * cy) * (ax * by - bx * ay)
                )
                orient = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                if (orient > 0 and det > 0) or (orient < 0 and det < 0):
                    bad_triangles.append(tri)

            polygon = []
            for tri in bad_triangles:
                edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
                for edge in edges:
                    shared = False
                    for other in bad_triangles:
                        if other == tri:
                            continue
                        other_edges = [
                            (other[0], other[1]), (other[1], other[0]),
                            (other[1], other[2]), (other[2], other[1]),
                            (other[2], other[0]), (other[0], other[2]),
                        ]
                        if edge in other_edges:
                            shared = True
                            break
                    if not shared:
                        polygon.append(edge)

            triangles = [t for t in triangles if t not in bad_triangles]
            for edge in polygon:
                triangles.append([edge[0], edge[1], i])

        # Remove triangles connected to the super-triangle vertices
        valid_triangles = []
        for tri in triangles:
            if not any(v >= n_points for v in tri):
                valid_triangles.append(tri)

        if len(valid_triangles) == 0:
            return np.empty((0, 3), dtype=np.int64)

        return np.asarray(valid_triangles, dtype=np.int64)

    @staticmethod
    def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Compute area-weighted surface normals for each vertex."""
        n_vertices = vertices.shape[0]
        normals = np.zeros((n_vertices, 3), dtype=np.float64)

        if len(faces) == 0:
            normals[:, 2] = 1.0
            return normals

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)

        # Accumulate face normals to vertices
        for i in range(len(faces)):
            fn = face_normals[i]
            normals[faces[i, 0]] += fn
            normals[faces[i, 1]] += fn
            normals[faces[i, 2]] += fn

        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normals = normals / norms

        # Ensure normals point predominantly upward (+Z)
        up_dots = normals[:, 2]
        if np.mean(up_dots) < 0:
            normals = -normals

        return normals


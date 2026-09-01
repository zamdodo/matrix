"""S3 - 3D Reconstruction."""

import numpy as np


class Reconstructor:
    """Generate 3D reconstruction from visual observations and camera poses."""

    def __init__(self, visual_data, localization_data):
        """
        Initialize the S3 reconstructor.

        Parameters
        ----------
        visual_data:
            Visual observations supplied by S1.
        localization_data:
            Camera localization / pose information supplied by S2.
        """
        self.visual_data = visual_data
        self.localization_data = localization_data

    @staticmethod
    def _triangulate(point_a, point_b, projection_a, projection_b):
        """Triangulate one 3D point from two camera observations."""
        point_a = np.asarray(point_a, dtype=float)
        point_b = np.asarray(point_b, dtype=float)
        projection_a = np.asarray(projection_a, dtype=float)
        projection_b = np.asarray(projection_b, dtype=float)

        if point_a.shape != (2,) or point_b.shape != (2,):
            raise ValueError("Image observations must contain two coordinates.")

        if projection_a.shape != (3, 4) or projection_b.shape != (3, 4):
            raise ValueError("Projection matrices must have shape (3, 4).")

        x1, y1 = point_a
        x2, y2 = point_b

        matrix = np.vstack(
            [
                x1 * projection_a[2] - projection_a[0],
                y1 * projection_a[2] - projection_a[1],
                x2 * projection_b[2] - projection_b[0],
                y2 * projection_b[2] - projection_b[1],
            ]
        )

        _, _, vh = np.linalg.svd(matrix)
        point_4d = vh[-1]

        if np.isclose(point_4d[3], 0.0):
            raise ValueError("Triangulation produced a point at infinity.")

        return (point_4d[:3] / point_4d[3]).tolist()

    @staticmethod
    def _projection_from_pose(pose):
        """
        Convert a 4x4 camera pose into a simple 3x4 projection matrix.

        The pose is expected to describe the camera coordinate frame.
        """
        pose = np.asarray(pose, dtype=float)

        if pose.shape != (4, 4):
            raise ValueError("Camera pose must have shape (4, 4).")

        return pose[:3, :]

    def _get_projection_matrix(self, obs):
        """Extract or compute projection matrix from an observation."""
        if "projection_matrix" in obs:
            return obs["projection_matrix"]
        if "pose" in obs:
            return self._projection_from_pose(obs["pose"])
        return None

    def _build_point_cloud(self):
        """Build a point cloud from explicit 3D points or pairs of 2D observations."""
        if self.visual_data is None or not isinstance(self.visual_data, (list, tuple)):
            return []

        points_3d = []

        # Process explicit 3D points
        for observation in self.visual_data:
            if isinstance(observation, dict) and "point_3d" in observation:
                point = np.asarray(observation["point_3d"], dtype=float)
                if point.shape == (3,) and np.all(np.isfinite(point)):
                    points_3d.append(point.tolist())

        # Process pairs of 2D observations with projection matrices or poses
        if len(self.visual_data) >= 2:
            for i in range(len(self.visual_data) - 1):
                obs1 = self.visual_data[i]
                obs2 = self.visual_data[i + 1]

                if isinstance(obs1, dict) and isinstance(obs2, dict) and "point" in obs1 and "point" in obs2:
                    proj1 = self._get_projection_matrix(obs1)
                    proj2 = self._get_projection_matrix(obs2)

                    if proj1 is not None and proj2 is not None:
                        try:
                            point_3d = self._triangulate(
                                obs1["point"],
                                obs2["point"],
                                proj1,
                                proj2,
                            )
                            points_3d.append(point_3d)
                        except ValueError:
                            continue

        return points_3d

    def reconstruct(self):
        """Generate the S4-compatible 3D reconstruction."""
        # Check if visual_data is an S2Payload or dict with observations
        from .models.schema import S2Payload
        from .pipeline import S3ReconstructionPipeline

        if isinstance(self.visual_data, S2Payload) or (isinstance(self.visual_data, dict) and "observations" in self.visual_data):
            pipeline = S3ReconstructionPipeline()
            result = pipeline.run(self.visual_data)
            return {
                "point_cloud": result.point_cloud.points.tolist(),
                "mesh": result.mesh.to_dict() if result.mesh is not None else None,
                "metadata": result.to_metadata_dict(),
            }

        point_cloud = self._build_point_cloud()
        mesh_dict = None

        if len(point_cloud) >= 3:
            from .geometry.mesher import SurfaceMesher
            from .models.s3_output import PointCloudData
            cloud_obj = PointCloudData(points=np.array(point_cloud, dtype=float))
            mesh_obj = SurfaceMesher().generate_mesh(cloud_obj)
            if mesh_obj is not None:
                mesh_dict = mesh_obj.to_dict()

        return {
            "point_cloud": point_cloud,
            "mesh": mesh_dict,
            "metadata": {
                "num_points": len(point_cloud),
                "source": "S1 visual observations + S2 camera localization",
                "reconstruction_method": "triangulation-ready",
            },
        }


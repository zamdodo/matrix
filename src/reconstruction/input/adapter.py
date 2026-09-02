from typing import Dict, Any, List

class S2ToS3Adapter:
    """Adapts S2 localization payload format into S3 reconstruction input."""

    @staticmethod
    def adapt(s2_payload: Dict[str, Any]) -> Dict[str, Any]:
        observations = s2_payload.get("observations", [])
        camera_poses = []
        feature_tracks = []

        for obs in observations:
            if isinstance(obs, dict):
                if "pose" in obs:
                    camera_poses.append(obs["pose"])
                if "features" in obs:
                    feature_tracks.extend(obs["features"])
            elif hasattr(obs, "pose"):
                camera_poses.append(getattr(obs, "pose"))
                if hasattr(obs, "features"):
                    feature_tracks.extend(getattr(obs, "features"))

        return {
            "camera_poses": camera_poses,
            "feature_tracks": feature_tracks,
            "metadata": s2_payload.get("metadata", {}),
        }

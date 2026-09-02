# Subsystem 3: 3D Reconstruction Interface Contract (v0.1)

## Input Contract (S2 Payload)
Consumes `S2Payload` containing camera poses and 2D feature tracks.

## Output Contract (S3 Payload)
Produces `S3Output` containing 3D point cloud coordinates $(X, Y, Z, \text{RGB})$, surface mesh, and reprojection error metadata.

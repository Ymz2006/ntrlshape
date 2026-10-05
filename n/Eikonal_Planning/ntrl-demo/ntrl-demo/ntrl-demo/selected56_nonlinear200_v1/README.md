# selected56_nonlinear200_v1

200 cases TOTAL, covering all 56 tasks: 100 planar + 100 spatial; each task contributes 3 or 4. Source preset is selected56_testcases_v2 / env34_all_shapes: 2D env3/env4 use tight001 (all 7 shapes), other tasks use original_1k. No source query is changed, and original stable query_id/source_row are retained.

## Nonlinear definition

Both endpoints pass the existing mesh validity check. Direct linear translation plus shortest quaternion SLERP has at least one invalid interior pose under sampling steps <=0.002 translation and <=0.01 rad rotation. Invalidity is classified as mesh collision or workspace-bound violation. Each selected_cases.json row stores a concrete failing pose and its interpolation fraction. This is a non-direct-connection benchmark; it is not a test of arbitrary curve curvature, nor a proof that a feasible path exists. Tight cases have no RRT-solvability guarantee. No cases were selected based on MPNet success/failure.

## Usage

Load tasks/<task>/queries.npz: start_xyz_xyzw and goal_xyz_xyzw are N×7 normalized xyz + qx,qy,qz,qw. query_ids and source_row retain original source identity; subset_row in selected_cases.json indexes this smaller package. CSV and original-format N×12 sampled_points.npy are included. Planar tasks also have queries_se2.npz (x,y,theta radians). All auxiliary per-query arrays are subset with the same row selection; env.npy is unchanged.

The local robot mesh geometry/<task>/robot_normalized.ply is already centered/scaled/oriented. Apply R,t directly. The environment_normalized.ply is already in the same normalized world frame. Do not reapply source_meta normalization. 2D motion is XY translation and Z rotation. Queries are not interpolated or modified by this export.

Use all supplied IDs and report every failure; do not silently replace unsolved cases. Endpoint masks describe this reference checker only, not a universal collision truth. Since task contributions differ (3/4), report both per-task macro SR and all-case pooled SR. Metrics should follow the full benchmark README: successful raw-path cumulative translation/rotation and explicitly defined planning time.

Selection is deterministic: task-specific SHA256-seeded source permutations, then balanced quotas using seed 20260913. First 12 qualifying candidates per task were screened; the selected first 3/4 were freshly rechecked at export. Original full package remains intact. checksums.json covers the frozen package files.

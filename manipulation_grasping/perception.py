"""
3D Vision & Point Cloud Perception Pipeline:
Voxel Grid Filtering, RANSAC Table Plane Segmentation,
and Euclidean Cluster Extraction with PCA Oriented Bounding Boxes.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class PointCloud:
    """3D Point Cloud container."""
    points: np.ndarray  # (N, 3) float32/float64
    colors: Optional[np.ndarray] = None  # (N, 3) float [0, 1]
    normals: Optional[np.ndarray] = None  # (N, 3) float unit vectors


@dataclass
class SegmentedObject:
    """Detected and isolated 3D tabletop object."""
    cluster_id: int
    points: np.ndarray  # (M, 3)
    centroid: np.ndarray  # (3,)
    dimensions: np.ndarray  # (3,) [extent_x, extent_y, extent_z]
    principal_axes: np.ndarray  # (3, 3) rotation matrix from PCA
    surface_normals: np.ndarray  # (M, 3)
    bounding_box_corners: np.ndarray  # (8, 3)


class PerceptionPipeline:
    """3D point cloud segmentation and geometric feature extraction."""

    def __init__(
        self,
        voxel_size: float = 0.005,
        ransac_distance_threshold: float = 0.012,
        ransac_max_iterations: int = 150,
        cluster_tolerance: float = 0.035,
        min_cluster_size: int = 30,
        max_cluster_size: int = 15000,
    ):
        self.voxel_size = voxel_size
        self.ransac_distance_threshold = ransac_distance_threshold
        self.ransac_max_iterations = ransac_max_iterations
        self.cluster_tolerance = cluster_tolerance
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

    def pass_through_filter(
        self,
        cloud: PointCloud,
        x_bounds: Tuple[float, float] = (0.2, 0.85),
        y_bounds: Tuple[float, float] = (-0.45, 0.45),
        z_bounds: Tuple[float, float] = (-0.1, 0.8),
    ) -> PointCloud:
        """Filter points outside workspace ROI bounds."""
        pts = cloud.points
        mask = (
            (pts[:, 0] >= x_bounds[0]) & (pts[:, 0] <= x_bounds[1]) &
            (pts[:, 1] >= y_bounds[0]) & (pts[:, 1] <= y_bounds[1]) &
            (pts[:, 2] >= z_bounds[0]) & (pts[:, 2] <= z_bounds[1])
        )
        filtered_pts = pts[mask]
        filtered_colors = cloud.colors[mask] if cloud.colors is not None else None
        return PointCloud(points=filtered_pts, colors=filtered_colors)

    def voxel_downsample(self, cloud: PointCloud) -> PointCloud:
        """Voxel grid downsampling using spatial hash dictionary."""
        if len(cloud.points) == 0:
            return cloud

        inv_voxel = 1.0 / self.voxel_size
        voxel_indices = np.floor(cloud.points * inv_voxel).astype(np.int64)

        # Dictionary of voxel index -> list of point indices
        voxels = {}
        for idx, v in enumerate(voxel_indices):
            key = (int(v[0]), int(v[1]), int(v[2]))
            if key not in voxels:
                voxels[key] = []
            voxels[key].append(idx)

        downsampled_pts = []
        downsampled_colors = [] if cloud.colors is not None else None

        for idx_list in voxels.values():
            pts_in_voxel = cloud.points[idx_list]
            mean_pt = np.mean(pts_in_voxel, axis=0)
            downsampled_pts.append(mean_pt)
            if cloud.colors is not None:
                mean_col = np.mean(cloud.colors[idx_list], axis=0)
                downsampled_colors.append(mean_col)

        pts_array = np.array(downsampled_pts, dtype=np.float64)
        col_array = np.array(downsampled_colors, dtype=np.float64) if downsampled_colors else None
        return PointCloud(points=pts_array, colors=col_array)

    def ransac_plane_segmentation(
        self, cloud: PointCloud
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment the dominant planar surface (e.g. tabletop) via RANSAC.
        Returns:
            plane_model: [a, b, c, d] such that a*x + b*y + c*z + d = 0, with ||[a,b,c]|| = 1.
            inlier_indices: Indices of points belonging to table.
            outlier_indices: Indices of remaining points (objects on table).
        """
        pts = cloud.points
        N = len(pts)
        if N < 3:
            return np.array([0.0, 0.0, 1.0, 0.0]), np.array([]), np.arange(N)

        best_inliers = np.array([], dtype=int)
        best_plane = np.array([0.0, 0.0, 1.0, 0.0])

        rng = np.random.default_rng(42)

        for _ in range(self.ransac_max_iterations):
            # Sample 3 random points
            sample_idx = rng.choice(N, size=3, replace=False)
            p1, p2, p3 = pts[sample_idx[0]], pts[sample_idx[1]], pts[sample_idx[2]]

            # Normal vector: (p2 - p1) x (p3 - p1)
            v12 = p2 - p1
            v13 = p3 - p1
            normal = np.cross(v12, v13)
            norm_val = np.linalg.norm(normal)
            if norm_val < 1e-6:
                continue

            normal = normal / norm_val
            # Enforce positive z component for upward table normal
            if normal[2] < 0.0:
                normal = -normal

            # Prefer planes roughly parallel to horizontal ground (normal.z > 0.8)
            if normal[2] < 0.7:
                continue

            d = -np.dot(normal, p1)

            # Distance of all points to plane: |a*x + b*y + c*z + d|
            distances = np.abs(pts @ normal + d)
            inliers = np.where(distances < self.ransac_distance_threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = np.array([normal[0], normal[1], normal[2], d])

        # If a valid plane is found, refine using all inliers (PCA on inliers)
        if len(best_inliers) >= 3:
            inlier_pts = pts[best_inliers]
            centroid = np.mean(inlier_pts, axis=0)
            cov = np.cov((inlier_pts - centroid).T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            refined_normal = eigvecs[:, 0]  # Smallest eigenvalue corresponds to normal
            if refined_normal[2] < 0.0:
                refined_normal = -refined_normal
            refined_d = -np.dot(refined_normal, centroid)
            best_plane = np.array([refined_normal[0], refined_normal[1], refined_normal[2], refined_d])

            # Recalculate inliers with refined plane
            distances = np.abs(pts @ best_plane[:3] + best_plane[3])
            best_inliers = np.where(distances < self.ransac_distance_threshold)[0]

        all_indices = np.arange(N)
        outlier_mask = np.ones(N, dtype=bool)
        outlier_mask[best_inliers] = False
        outlier_indices = all_indices[outlier_mask]

        # Only keep outliers that are strictly ABOVE the table plane
        plane_normal = best_plane[:3]
        plane_d = best_plane[3]
        signed_heights = pts[outlier_indices] @ plane_normal + plane_d
        above_table_mask = signed_heights > (self.ransac_distance_threshold * 0.8)
        outlier_indices = outlier_indices[above_table_mask]

        return best_plane, best_inliers, outlier_indices

    def euclidean_clustering(self, points: np.ndarray) -> List[np.ndarray]:
        """
        Segment points into disjoint clusters using spatial proximity.
        """
        N = len(points)
        if N < self.min_cluster_size:
            return []

        visited = np.zeros(N, dtype=bool)
        clusters = []

        # Simple grid-based neighbor search
        cell_size = self.cluster_tolerance
        grid = {}
        for i in range(N):
            cell = tuple(np.floor(points[i] / cell_size).astype(int))
            if cell not in grid:
                grid[cell] = []
            grid[cell].append(i)

        neighbor_offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ]

        for i in range(N):
            if visited[i]:
                continue

            cluster_indices = []
            queue = [i]
            visited[i] = True

            while queue:
                curr = queue.pop(0)
                cluster_indices.append(curr)

                pt_curr = points[curr]
                cell_curr = tuple(np.floor(pt_curr / cell_size).astype(int))

                # Query neighboring cells
                for offset in neighbor_offsets:
                    neighbor_cell = (
                        cell_curr[0] + offset[0],
                        cell_curr[1] + offset[1],
                        cell_curr[2] + offset[2],
                    )
                    if neighbor_cell in grid:
                        for neighbor_idx in grid[neighbor_cell]:
                            if not visited[neighbor_idx]:
                                dist = np.linalg.norm(pt_curr - points[neighbor_idx])
                                if dist <= self.cluster_tolerance:
                                    visited[neighbor_idx] = True
                                    queue.append(neighbor_idx)

            if self.min_cluster_size <= len(cluster_indices) <= self.max_cluster_size:
                clusters.append(np.array(cluster_indices, dtype=int))

        return clusters

    @staticmethod
    def estimate_surface_normals(points: np.ndarray, k_neighbors: int = 10) -> np.ndarray:
        """Estimate per-point surface normals via local PCA."""
        N = len(points)
        normals = np.zeros((N, 3))
        if N < 3:
            return normals

        for i in range(N):
            dists = np.linalg.norm(points - points[i], axis=1)
            nearest_idx = np.argsort(dists)[:min(k_neighbors, N)]
            neighbors = points[nearest_idx]

            centroid = np.mean(neighbors, axis=0)
            cov = np.cov((neighbors - centroid).T)
            if cov.shape != (3, 3):
                normals[i] = np.array([0.0, 0.0, 1.0])
                continue

            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, 0]
            # Orient normal outward relative to centroid of entire object
            obj_centroid = np.mean(points, axis=0)
            if np.dot(normal, points[i] - obj_centroid) < 0.0:
                normal = -normal

            normals[i] = normal

        return normals

    def extract_segmented_objects(
        self, cloud: PointCloud, outlier_indices: np.ndarray
    ) -> List[SegmentedObject]:
        """Convert clusters into structured SegmentedObject representations."""
        if len(outlier_indices) == 0:
            return []

        obj_points = cloud.points[outlier_indices]
        cluster_list = self.euclidean_clustering(obj_points)

        segmented_objects = []

        for cid, cluster_idx in enumerate(cluster_list):
            c_pts = obj_points[cluster_idx]
            centroid = np.mean(c_pts, axis=0)

            # PCA for 3D Oriented Bounding Box
            centered_pts = c_pts - centroid
            cov = np.cov(centered_pts.T)
            eigvals, eigvecs = np.linalg.eigh(cov)

            # Sort eigenvalues descending (principal directions)
            order = np.argsort(eigvals)[::-1]
            principal_axes = eigvecs[:, order]

            # Ensure right-handed coordinate frame: det(R) = +1
            if np.linalg.det(principal_axes) < 0.0:
                principal_axes[:, 2] = -principal_axes[:, 2]

            # Project points onto principal axes to compute bounding box extents
            projected = centered_pts @ principal_axes
            mins = np.min(projected, axis=0)
            maxs = np.max(projected, axis=0)
            dimensions = maxs - mins

            # Compute 8 corners of OBB
            corners_local = np.array([
                [mins[0], mins[1], mins[2]],
                [maxs[0], mins[1], mins[2]],
                [maxs[0], maxs[1], mins[2]],
                [mins[0], maxs[1], mins[2]],
                [mins[0], mins[1], maxs[2]],
                [maxs[0], mins[1], maxs[2]],
                [maxs[0], maxs[1], maxs[2]],
                [mins[0], maxs[1], maxs[2]],
            ])
            corners_world = (corners_local @ principal_axes.T) + centroid

            # Estimate normals for grasp evaluation
            normals = self.estimate_surface_normals(c_pts)

            segmented_objects.append(
                SegmentedObject(
                    cluster_id=cid,
                    points=c_pts,
                    centroid=centroid,
                    dimensions=dimensions,
                    principal_axes=principal_axes,
                    surface_normals=normals,
                    bounding_box_corners=corners_world,
                )
            )

        return segmented_objects

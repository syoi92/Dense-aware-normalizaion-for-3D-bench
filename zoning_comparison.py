import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.cluster import DBSCAN
import os

def estimate_density_convex_hull(points):
    if len(points) >= 4:
        hull = ConvexHull(points)
        volume = hull.volume
        density = len(points) / volume if volume != 0 else 0
        return density
    return 0

def estimate_density_knn(points, k=10):
    if len(points) > k:
        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        distances, _ = nbrs.kneighbors(points)
        r = np.mean(distances[:, 1:], axis=1)  # Exclude the distance to itself
        local_density = k / ((4/3) * np.pi * r**3)
        return np.mean(local_density)
    return 0

def visualize_and_save_zones(points, labels, output_folder, file_name):
    num_zones = len(np.unique(labels))
    colors = np.random.rand(num_zones, 3)  # Generate random colors for each zone
    
    # Assign colors based on labels
    colored_points = np.zeros((points.shape[0], 3))
    for i in range(num_zones):
        if i == -1:  # Skip noise
            continue
        colored_points[labels == i] = colors[i]
    
    # Create a point cloud with colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colored_points)
    
    # Save as PLY file
    output_path = os.path.join(output_folder, file_name)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved: {output_path}")

def cluster_and_estimate_density(point_cloud, output_folder):
    labels = DBSCAN(eps=0.05, min_samples=10).fit_predict(point_cloud)
    unique_labels = set(labels)

    densities = []
    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue  # Skip noise

        cluster_points = point_cloud[labels == cluster_id]

        # Density estimation using Convex Hull
        ch_density = estimate_density_convex_hull(cluster_points)
        
        # Density estimation using kNN
        knn_density = estimate_density_knn(cluster_points)

        densities.append({
            "cluster_id": cluster_id,
            "convex_hull_density": ch_density,
            "knn_density": knn_density,
            "num_points": len(cluster_points)
        })

    # Save densities to CSV
    df = pd.DataFrame(densities)
    df.to_csv(os.path.join(output_folder, "density_info.csv"), index=False)

    # Visualize and save the colored zones
    visualize_and_save_zones(point_cloud, labels, output_folder, "colored_clusters.ply")

def main(input_file, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the point cloud using Open3D
    pcd = o3d.io.read_point_cloud(input_file)
    point_cloud = np.asarray(pcd.points)
    
    # Cluster the point cloud and estimate densities
    cluster_and_estimate_density(point_cloud, output_folder)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_point_cloud_file> <output_folder>")
    else:
        input_file = sys.argv[1]
        output_folder = sys.argv[2]
        main(input_file, output_folder)

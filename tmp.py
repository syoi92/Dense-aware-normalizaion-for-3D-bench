import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, QhullError
from sklearn.neighbors import NearestNeighbors
import time
import os

def estimate_density_convex_hull(points):
    try:
        if len(points) >= 4:
            hull = ConvexHull(points)
            volume = hull.volume
            density = len(points) / volume if volume != 0 else 0
            return density, volume
    except QhullError as e:
        print("Convex Hull computation error:", e)
    return 0, 0

def estimate_density_knn(points, k=10):
    if len(points) > k:
        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        distances, _ = nbrs.kneighbors(points)
        r = np.mean(distances[:, 1:], axis=1)  # Exclude the distance to itself
        local_density = k / ((4/3) * np.pi * r**3)
        return np.mean(local_density)
    return 0

def visualize_and_save_zones(points, labels, output_folder, file_name):
    unique_labels = np.unique(labels)
    num_zones = len(unique_labels)
    colors = np.random.rand(num_zones, 3)  # Generate random colors for each zone
    
    # Create a mapping from label to color, assigning grey to noise points (label == -1)
    color_map = {label: colors[i] for i, label in enumerate(unique_labels) if label != -1}
    color_map[-1] = np.array([0.75, 0.75, 0.75])  # Grey color for noise
    
    # Assign colors based on labels
    colored_points = np.array([color_map[label] for label in labels])
    
    # Create a point cloud with colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colored_points)
    
    # Save as PLY file
    output_path = os.path.join(output_folder, file_name)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Visualization saved: {output_path}")

def cluster_and_estimate_density(point_cloud, output_folder):
    print("Starting DBSCAN clustering...")
    labels = DBSCAN(eps=0.05, min_samples=10).fit_predict(point_cloud)
    unique_labels = set(labels)

    densities = []
    total_time_ch = 0
    total_time_knn = 0
    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue  # Skip noise

        cluster_points = point_cloud[labels == cluster_id]

        # Density estimation using Convex Hull
        print(f"Calculating density for cluster {cluster_id} using Convex Hull...")
        start_time = time.time()
        ch_density, volume = estimate_density_convex_hull(cluster_points)
        ch_time = time.time() - start_time
        total_time_ch += ch_time
        
        # Density estimation using kNN
        print(f"Calculating density for cluster {cluster_id} using kNN...")
        start_time = time.time()
        knn_density = estimate_density_knn(cluster_points)
        knn_time = time.time() - start_time
        total_time_knn += knn_time

        densities.append({
            "cluster_id": cluster_id,
            "convex_hull_density": ch_density,
            "knn_density": knn_density,
            "num_points": len(cluster_points)
        })

    # Save densities to CSV
    df = pd.DataFrame(densities)
    df.to_csv(os.path.join(output_folder, "density_info.csv"), index=False)
    print(f"Density information saved: {os.path.join(output_folder, 'density_info.csv')}")
    print(f"Total time for Convex Hull density calculations: {total_time_ch:.6f}s")
    print(f"Total time for kNN density calculations: {total_time_knn:.6f}s")

    # Visualize and save the colored zones
    visualize_and_save_zones(point_cloud, labels, output_folder, "colored_clusters.ply")

def main(input_file, output_folder):
    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud(input_file)
    point_cloud = np.asarray(pcd.points)
    print("Point cloud loaded.")
    
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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

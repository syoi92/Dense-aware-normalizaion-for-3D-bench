import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import os
import time

# Load point cloud from PLY file
def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return pcd, points

# Octree-based density zoning
def octree_density_zones(points, max_depth=4):
    # octree = o3d.geometry.Octree(max_depth=max_depth)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # octree.convert_from_point_cloud(pcd, size_expand=0.01)
    # return octree

    octree = o3d.geometry.Octree(max_depth=max_depth)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    labels = np.zeros(len(points), dtype=int)
    zone_id = 0

    def assign_zone(node, node_info):
        nonlocal zone_id
        if isinstance(node, o3d.geometry.OctreeLeafNode):
            for idx in node.indices:
                labels[idx] = zone_id
            zone_id += 1
    
    octree.traverse(assign_zone)
    return labels

# DBSCAN-based zoning
def dbscan_density_zones(points, eps=0.05, min_samples=10):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return db.labels_

# GMM-based zoning
def gmm_density_zones(points, n_components=5):
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(points)
    return gmm.predict(points)

# KNN Density Estimation
def knn_density_zones(points, n_neighbors=10):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    distances, _ = nbrs.kneighbors(points)
    densities = 1 / (np.mean(distances, axis=1) + 1e-10)
    return np.digitize(densities, np.linspace(np.min(densities), np.max(densities), 6)) - 1

# Visualization function: Assign colors to zones and save PLY files
def visualize_and_save_zones(points, labels, output_folder, file_name):
    num_zones = len(np.unique(labels))
    colors = np.random.rand(num_zones, 3)  # Generate random colors for each zone
    
    # Assign colors based on labels
    colored_points = np.zeros((points.shape[0], 3))
    for i in range(num_zones):
        colored_points[labels == i] = colors[i]
    
    # Create a point cloud with colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colored_points)
    
    # Save as PLY file
    output_path = os.path.join(output_folder, file_name)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved: {output_path}")

# Function to compare methods and save results
def compare_methods(input_file, output_folder):
    pcd, points = load_point_cloud(input_file)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Octree method
    start_time = time.time()
    octree_labels = octree_density_zones(points, max_depth=4)
    octree_time = time.time() - start_time
    visualize_and_save_zones(points, octree_labels, output_folder, "octree_zones.ply")

    # DBSCAN method
    start_time = time.time()
    dbscan_labels = dbscan_density_zones(points, eps=0.05, min_samples=10)
    dbscan_time = time.time() - start_time
    visualize_and_save_zones(points, dbscan_labels, output_folder, "dbscan_zones.ply")

    # GMM method
    start_time = time.time()
    gmm_labels = gmm_density_zones(points, n_components=5)
    gmm_time = time.time() - start_time
    visualize_and_save_zones(points, gmm_labels, output_folder, "gmm_zones.ply")

    # KNN method
    start_time = time.time()
    knn_zones = knn_density_zones(points, n_neighbors=10)
    knn_time = time.time() - start_time
    visualize_and_save_zones(points, knn_zones, output_folder, "knn_zones.ply")

    # Print time results
    print("\nExecution Times:")
    print(f"Octree method time: {octree_time:.4f} seconds")
    print(f"DBSCAN method time: {dbscan_time:.4f} seconds")
    print(f"GMM method time: {gmm_time:.4f} seconds")
    print(f"KNN method time: {knn_time:.4f} seconds")

# Main function
def main():
    input_file = "samples/room-17.ply"  # Replace with your own PLY file path
    output_folder = "samples/output-17"  # Replace with your desired output folder
    compare_methods(input_file, output_folder)

# Run the script
if __name__ == "__main__":
    main()



import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import os

def calDensity(input_file, output_folder):
    # Load a point cloud
    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud(input_file)
    points = np.asarray(pcd.points)
    print(f"Loaded point cloud with {len(points)} points.")

    # Compute density using a k-d tree
    print("Building KD-Tree...")
    kdtree = KDTree(points)

    # Set the radius for local neighborhood density calculation
    radius = 0.1

    # Precompute neighbors for all points
    print("Precomputing neighbors for all points...")
    all_neighbors = kdtree.query_ball_tree(kdtree, radius)

    # Initialize density and gradient vectors
    print("Calculating densities...")
    densities = np.array([len(neighbors) for neighbors in all_neighbors])

    # Compute gradient for each point
    print("Calculating gradients...")
    gradient_vectors = []
    for i, neighbors in enumerate(all_neighbors):
        if i % 1000 == 0:
            print(f"Processing point {i}/{len(points)}...")
        density_differences = [densities[i] - densities[j] for j in neighbors if j != i]
        gradient = np.mean(density_differences) if density_differences else 0
        gradient_vectors.append(gradient)

    # Convert gradients to numpy array
    gradient_magnitudes = np.array(gradient_vectors)

    # Normalize densities and gradient magnitudes for visualization
    print("Normalizing densities and gradients...")
    densities_normalized = (densities - np.min(densities)) / (np.max(densities) - np.min(densities))
    gradient_magnitudes_normalized = (gradient_magnitudes - np.min(gradient_magnitudes)) / (np.max(gradient_magnitudes) - np.min(gradient_magnitudes))

    # Map densities to colors and save density-colored point cloud
    print("Mapping densities to colors and saving point cloud...")
    # density_colors = cm.viridis(densities_normalized)[:, :3]
    density_colors = cm.inferno(densities_normalized)[:, :3]
    density_pcd = o3d.geometry.PointCloud()
    density_pcd.points = o3d.utility.Vector3dVector(points)
    density_pcd.colors = o3d.utility.Vector3dVector(density_colors)
    density_output_path = os.path.join(output_folder, os.path.split(input_file)[-1].replace('.ply', '_density.ply'))
    o3d.io.write_point_cloud(density_output_path, density_pcd)

    # Map gradient magnitudes to colors and save gradient-colored point cloud
    print("Mapping gradients to colors and saving point cloud...")
    # gradient_colors = cm.viridis(gradient_magnitudes_normalized)[:, :3]
    gradient_colors = cm.inferno(gradient_magnitudes_normalized)[:, :3]
    gradient_pcd = o3d.geometry.PointCloud()
    gradient_pcd.points = o3d.utility.Vector3dVector(points)
    gradient_pcd.colors = o3d.utility.Vector3dVector(gradient_colors)
    gradient_output_path = os.path.join(output_folder, os.path.split(input_file)[-1].replace('.ply', '_gradient.ply'))
    o3d.io.write_point_cloud(gradient_output_path, gradient_pcd)

    # print("Visualization of results...")
    # # Visualize both point clouds
    # o3d.visualization.draw_geometries([density_pcd], window_name="Density Visualization")
    # o3d.visualization.draw_geometries([gradient_pcd], window_name="Gradient Visualization")


def pcd_convert_viridis_to_inferno(input_file):
    """
    Convert RGB colors from Viridis to Inferno by reverse-mapping.
    
    Parameters:
        viridis_colors (ndarray): Array of RGB values mapped with Viridis.
    
    Returns:
        inferno_colors (ndarray): Array of RGB values mapped with Inferno.
    """
    pcd = o3d.io.read_point_cloud(input_file)
    points = np.asarray(pcd.points)
    viridis_colors = np.asarray(pcd.colors)
    
    # Initialize the Viridis and Inferno colormaps
    viridis_cmap = cm.get_cmap('viridis', 256)  # 256 levels for smooth mapping
    inferno_cmap = cm.get_cmap('inferno', 256)

    # Generate a colormap lookup table (LUT) for reverse mapping
    viridis_lut = np.array([viridis_cmap(i)[:3] for i in range(256)])
        
    # Reverse-map Viridis RGB values to normalized values (0 to 1)
    viridis_norm_values = [
        np.argmin(np.sum((viridis_lut - rgb[:3])**2, axis=1)) / 255
        for rgb in viridis_colors
    ]

    # Map normalized values to Inferno colormap
    inferno_colors = inferno_cmap(viridis_norm_values)[:, :3]

    n_pcd = o3d.geometry.PointCloud()
    n_pcd.points = o3d.utility.Vector3dVector(points)
    n_pcd.colors = o3d.utility.Vector3dVector(inferno_colors)
    density_output_path = input_file.replace('.ply', '_r.ply')
    o3d.io.write_point_cloud(density_output_path, n_pcd)
    
    return n_pcd


def density_cdf(points): # points with xyz coordinates
    
    resol = 0.1
    bndry = 20

    dist2 = points[:,0]**2 + points[:,1]**2
    p_num = len(points)
    pdf= []

    radius = resol
    for _ in range(int(bndry/ resol)):

        pdf.append(((dist2 < radius**2).sum() - (dist2 < (radius-resol)**2).sum())/ p_num)
        radius += resol
    pdf.append(1 - sum(pdf))

    # Toward center
    cdf = []
    cdf.append(pdf[-1])
    for i in range(1,len(pdf)):
        cdf.append(cdf[-1]+ pdf[-i])
    cdf = cdf[::-1]

    # #from center
    # cdf = []
    # cdf.append(pdf[0])
    # for i in range(1,len(pdf)):
    #     cdf.append(cdf[-1]+ pdf[i])
    

    r = 2
    rr = (dist2 < r**2).sum() / p_num
    rr

    distance_intervals = [i * resol for i in range(len(pdf))]


    # Plot PDF
    plt.figure()
    plt.plot(distance_intervals, pdf, label='PDF', color='blue')
    plt.xlabel('Distance from Center')
    plt.ylabel('Probability')
    plt.title('PDF of Point Density')
    plt.legend()
    plt.grid()
    plt.savefig('pdf_plot.png', dpi=300, bbox_inches='tight')

    # Plot CDF
    plt.figure()
    plt.plot(distance_intervals, cdf, label='CDF', color='green')
    plt.axhline(y=0.5, color='red', linestyle='--', label='0.5')
    plt.xlabel('Distance from Center')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Point Density (Toward Center)')
    plt.legend()
    plt.grid()
    plt.savefig('cdf_plot.png', dpi=300, bbox_inches='tight')

    plt.show()
        






def save_colormap_legend(colormap_name, output_file):
    """
    Save a legend for a given colormap.
    
    Parameters:
        colormap_name (str): Name of the colormap (e.g., 'viridis', 'inferno').
        output_file (str): Path to save the legend image.
    """
    cmap = cm.get_cmap(colormap_name, 256)
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack([gradient, gradient])

    # Plot and save the colormap legend
    plt.figure(figsize=(8, 2))
    plt.imshow(gradient, aspect="auto", cmap=cmap)
    plt.gca().set_axis_off()
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()


# Main function
def main():
    # # pcd for density representation
    # input_file = "samples/room-17.ply"  # Replace with your own PLY file path
    # output_folder = "samples/VisDen"  # Replace with your desired output folder
    # calDensity(input_file, output_folder)

    # # color conversion
    # input_file = "samples/VisDen/room-17_density.ply"
    # pcd_convert_viridis_to_inferno(input_file)

    # density cdf
    input_file = "samples/VisDen/room-17_density.ply"
    pcd = o3d.io.read_point_cloud(input_file)
    points = np.asarray(pcd.points)
    density_cdf(points)    


# Run the script
if __name__ == "__main__":
    main()

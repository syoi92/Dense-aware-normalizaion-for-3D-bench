# import laspy
# import ezdxf
import numpy as np
import open3d as o3d
import json
import struct
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
import torch


# function to read LAS/LAZ file into NxF numpy array
# def readLAS(filename):
#     las = laspy.read(filename)
#     pcd = np.stack([las.x, las.y, las.z, las.red, las.blue, las.green], axis = -1).astype(np.float32)
#     pcd[:, 3:6] /= 65536 #normalize RGB values
#     return pcd

# function to conver pth to ply
def readPTH_batch(folder):
    # folder = "../_data/cnst-zone1/Val"
    paths = glob(folder + '/*.pth')
    for path in paths:
        readPTH(path)


def readPTH(path):
    # path = "../_data/cnst-zone1/Val/in-SC-2021-1223-17_a0.pth"
    pd = torch.load(path)
    xyz = pd['coord']
    rgb = pd['color']
    # cnst = pd['semantic_gt']

    output_cloud = o3d.geometry.PointCloud()
    output_cloud.points = o3d.utility.Vector3dVector(xyz)
    output_cloud.colors = o3d.utility.Vector3dVector(rgb/255)
    o3d.io.write_point_cloud(path.replace('.pth', '.ply'), output_cloud)
    print('Saved',len(xyz),'points to', path.replace('.pth', '.ply'))


# function to read PLY file into NxF numpy array
def readPLY(filename):
    pcd = o3d.io.read_point_cloud(filename)
    pcd = np.hstack([pcd.points, pcd.colors]).astype(np.float32)
    return pcd

# function to write NxF numpy array point cloud to PLY file
# point cloud should have at least 6 columns: XYZ, RGB [0 - 1]
def writePLY(pcd, filename):
    output_cloud = o3d.geometry.PointCloud()
    output_cloud.points = o3d.utility.Vector3dVector(pcd[:, :3])
    output_cloud.colors = o3d.utility.Vector3dVector(pcd[:, 3:6])
    o3d.io.write_point_cloud(filename, output_cloud)
    print('Saved',len(pcd),'points to',filename)

# function to downsample a point cloud
def downsample(cloud, resolution):
    voxel_set = set()
    output_cloud = []
    voxels = [tuple(k) for k in np.round(cloud[:, :3]/resolution).astype(int)]
    for i in range(len(voxels)):
        if not voxels[i] in voxel_set:
            output_cloud.append(cloud[i])
            voxel_set.add(voxels[i])
    return np.array(output_cloud) 

# function to read binary floorplan OBJ file into Python object
def readOBJ(filename):
    f = open(filename, 'rb')
    num_layers = struct.unpack('I', f.read(4))[0]
    num_structures = [i[0] for i in struct.iter_unpack('I', f.read(4 * num_layers))]
    floorplan = {
      "header": {
        "layer number": num_layers,
        "structure number": num_structures,
      }}
    for i in range(num_layers):
        len_layer_name = struct.unpack('I', f.read(4))[0]
        layer_name = struct.unpack(str(len_layer_name)+'s', f.read(len_layer_name))[0].decode('ascii')
        point_array = []
        for j in range(num_structures[i]):
            num_points = struct.unpack('I', f.read(4))[0]
            coordinates = [i[0] for i in struct.iter_unpack('f', f.read(4 * 2 * num_points))]
            point_array.append({
                "point number": num_points,
                "coordinates": coordinates,
            })
        floorplan['layer '+str(i)] = {
            "layer name": layer_name,
            "points": point_array
        }
    return floorplan

# function to read JSON file into Python object
def readJSON(filename):
    return json.load(open(filename, 'r'))

# function to write Python object to a JSON file 
def writeJSON(obj, filename):
    with open(filename, 'w') as outfile:
        json.dump(obj, outfile, indent=2)

# function to write array of cuboids (Nx8x3) to a DXF file
# each cuboid is defined as 8 XYZ points
# def writeDXF(cuboids, filename):
#     doc = ezdxf.new('R2018')
#     msp = doc.modelspace()
#     for layer_name in cuboids:
#         for c in cuboids[layer_name]:
#             if len(c)==8:
#                 e = msp.add_polyface({'layer': layer_name, 'm_close': True, 'n_close': False})
#                 e.append_face([c[0], c[1], c[3], c[2]])
#                 e = msp.add_polyface({'layer': layer_name, 'm_close': True, 'n_close': False})
#                 e.append_face([c[0], c[1], c[5], c[4]])
#                 e = msp.add_polyface({'layer': layer_name, 'm_close': True, 'n_close': False})
#                 e.append_face([c[0], c[2], c[6], c[4]])
#                 e = msp.add_polyface({'layer': layer_name, 'm_close': True, 'n_close': False})
#                 e.append_face([c[1], c[3], c[7], c[5]])
#                 e = msp.add_polyface({'layer': layer_name, 'm_close': True, 'n_close': False})
#                 e.append_face([c[2], c[3], c[7], c[6]])
#                 e = msp.add_polyface({'layer': layer_name, 'm_close': True, 'n_close': False})
#                 e.append_face([c[4], c[5], c[7], c[6]])
# #            else:
# #                centroid = np.mean(c, axis=0)
# #                for i in range(len(c) - 1):
# #                    e = msp.add_polyface({'layer': layer_name, 'm_close': True, 'n_close': False})
# #                    e.append_face([c[i], c[i+1], centroid])
# #                e = msp.add_polyface({'layer': layer_name, 'm_close': True, 'n_close': False})
# #                e.append_face([c[-1], c[0], centroid])
#     doc.saveas(filename)

# function to convert wall  and door line array to Python object in JSON format
# each line is expressed as (x1, y1, x2, y2) endpoints
def linesToJSON(wall_lines, door_lines, stair_lines):
    wall_point_array = []
    for i in range(len(wall_lines)):
        wall_point_array.append({
            "point number": 2,
            "coordinates": [wall_lines[i][0], wall_lines[i][1], wall_lines[i][2], wall_lines[i][3]],
        })
    door_point_array = []
    for i in range(len(door_lines)):
        door_point_array.append({
            "point number": 2,
            "coordinates": [door_lines[i][0], door_lines[i][1], door_lines[i][2], door_lines[i][3]],
        })
    stair_point_array = []
    for i in range(len(stair_lines)):
        stair_point_array.append({
            "point number": 2,
            "coordinates": [stair_lines[i][0], stair_lines[i][1], stair_lines[i][2], stair_lines[i][3]],
        })
    floorplan = {
        "header": {
            "layer number": 3,
            "structure number": [len(wall_lines), len(door_lines), len(stair_lines)],
        }
    }
    floorplan['layer 0'] = {
        "layer name": 'A_WALL',
        "points": wall_point_array,
    }
    floorplan['layer 1'] = {
        "layer name": 'A_DOOR',
        "points": door_point_array,
    }
    floorplan['layer 2'] = {
        "layer name": 'A_FLOR_STRS',
        "points": stair_point_array,
    }
    return floorplan 

# function to plot the floorplan from a Python object
# optionally overlay the ground truth floorplan
# optionally overlay the point cloud
# optionally save the figure to file
def drawFloorplan(floorplan, gt_floorplan=None, cloud=None, savefile=None):
    segs = []
    colors = []
    linewidths = []
    colormap = {
        'A_WALL': (0,0,0,1),
        'A_DOOR': (0,0,1,1),
        'A_FLOR_STRS': (0,1,0,1),
        'A_WALL_CONC': (0,0,0,1),
        'A_WALL_LINK': (0,0,0,0.5),
        'A_WALL_METL': (0,0,0,0.5),
        'A_WALL_MOVE': (0,0,0,0.5),
        'A_WALL_PRHT': (0,0,0,0.5),
        'S_CONC_WALL': (0,0,0,0.5),
    }
    widthmap = {
        'A_WALL': 1.5,
        'A_DOOR': 2.5,
        'A_FLOR_STRS': 1.5,
        'A_WALL_CONC': 1.5,
        'A_WALL_LINK': 1.5,
        'A_WALL_METL': 1.5,
        'A_WALL_MOVE': 1.5,
        'A_WALL_PRHT': 1.5,
        'S_CONC_WALL': 1.5,
    }
    i = 0
    while True:
        if 'layer '+str(i) in floorplan:
            layer = floorplan['layer '+str(i)]
            layer_color = colormap[layer['layer name']]
            layer_width = widthmap[layer['layer name']]
            num_structures = len(layer['points'])
            for j in range(num_structures):
                segs.append(np.array(layer['points'][j]['coordinates']).reshape(-1,2))
                colors.append(layer_color)
                linewidths.append(layer_width)
            i += 1
        else:
            break
    if gt_floorplan is not None:
        i = 0
        while True:
            if 'layer '+str(i) in gt_floorplan:
                layer = gt_floorplan['layer '+str(i)]
                layer_color = colormap[layer['layer name']]
                layer_color = (layer_color[0], layer_color[1], layer_color[2], layer_color[3]*0.5)
                layer_width = widthmap[layer['layer name']] * 0.5
                num_structures = len(layer['points'])
                for j in range(num_structures):
                    segs.append(np.array(layer['points'][j]['coordinates']).reshape(-1,2))
                    colors.append(layer_color)
                    linewidths.append(layer_width)
                i += 1
            else:
                break
    ln_coll = matplotlib.collections.LineCollection(segs, colors=colors, linewidths=linewidths)
    plt.figure(figsize=(10, 10), dpi=100)
    ax = plt.gca()
    ax.add_collection(ln_coll)
    if len(segs) > 0:
        segs_reshaped = np.vstack(segs)
        minX = segs_reshaped[:,0].min()-1
        maxX = segs_reshaped[:,0].max()+1
        minY = segs_reshaped[:,1].min()-1
        maxY = segs_reshaped[:,1].max()+1
    else:
        minX, maxX = ax.get_xlim()
        minY, maxY = ax.get_ylim()

    if cloud is not None:
        pcd = o3d.io.read_point_cloud(cloud)
        #pcd = pcd.voxel_down_sample(1.0)
        xy = np.asarray(pcd.points)[:, :2]
        rgb = np.asarray(pcd.colors)
        sample_idx = np.random.choice(len(xy), int(0.01 * len(xy)), replace=False)
        xy = xy[sample_idx]
        rgb = rgb[sample_idx]
        plt.scatter(xy[:,0], xy[:,1], marker='.', c=rgb)
        minX = min(minX, xy[:,0].min())
        maxX = max(maxX, xy[:,0].max())
        minY = min(minY, xy[:,1].min())
        maxY = max(maxY, xy[:,1].max())

    ax.set_xlim(minX, maxX)
    ax.set_ylim(minY, maxY)
    if savefile is None:
        plt.show()
    else:
        plt.savefig(savefile)

if __name__=="__main__":
    pcd = readLAS('LAS/04_Library_01_F1_s0p01m.LAZ')
    print('Loaded point cloud with', pcd.shape, 'points')
    print('Minimums: ', pcd.min(axis=0))
    print('Maximums: ', pcd.max(axis=0))

    pcd = readPLY('ply/04_Library_01_F1_s0p01m - Cloud.ply')
    print('Loaded point cloud with', pcd.shape, 'points')
    print('Minimums: ', pcd.min(axis=0))
    print('Maximums: ', pcd.max(axis=0))
    print(pcd[0], pcd[100], pcd[1000])

    pcd = readPLY('ply/12_SmallBuilding_03_F1_s0p01m - Cloud.ply')
    print('Loaded point cloud with', pcd.shape, 'points')
    print('Minimums: ', pcd.min(axis=0))
    print('Maximums: ', pcd.max(axis=0))
    print(pcd[0], pcd[100], pcd[1000])

    floorplan = readJSON('json/01_OfficeLab_01_F1_floorplan.txt')
    print('Floorplan:')
    i = 0
    while True:
        if 'layer '+str(i) in floorplan:
            layer = floorplan['layer '+str(i)]
            num_structures = len(layer['points'])
            print(layer['layer name'], num_structures)
            i += 1
        else:
            break

    floorplan = readOBJ('obj/01_OfficeLab_01_F1_floorplan.obj')
    print('Floorplan:')
    i = 0
    while True:
        if 'layer '+str(i) in floorplan:
            layer = floorplan['layer '+str(i)]
            num_structures = len(layer['points'])
            print(layer['layer name'], num_structures)
            i += 1
        else:
            break
    writeJSON(floorplan, 'json.txt')

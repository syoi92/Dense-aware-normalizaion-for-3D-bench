# import laspy
# import ezdxf
import numpy as np
import open3d as o3d
import json, os
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob
import torch


def main():
    folder = '../_data/cnst-zone4/Test'
    results_folder = '../_data/results/tv2-zone4'
    npy2ply(folder, results_folder)


def anno_relocate(wip_folder):
    ndir = os.path.join(os.path.split(wip_folder)[0],'cnstpcim')
    if not os.path.isdir(ndir):
        os.mkdir(ndir)

    rooms = glob(wip_folder + "/*/")
    rooms.sort()
    print("# of Data: ", len(rooms))

    for room in rooms:
        print("Extract from ", room)
        if not os.path.isdir(room.replace('wip_cnst', 'cnstpcim')):
            os.mkdir(room.replace('wip_cnst', 'cnstpcim'))

        if not os.path.isdir(os.path.join(room.replace('wip_cnst', 'cnstpcim'), 'Annotation')):
            os.mkdir(os.path.join(room.replace('wip_cnst', 'cnstpcim'), 'Annotation'))

        annos = glob(room +"/*.txt")

        bans = ['noise', 'merged', 'label']
        for ban in bans:
            annos = [ x for x in annos if ban not in x.lower()]

        for anno_filtered in annos:
            pcd = np.loadtxt(anno_filtered)
            pcd_d = downsample(pcd, resolution=0.001)

            fn = os.path.split(anno_filtered.replace('wip_cnst', 'cnstpcim'))
            np.savetxt(os.path.join(fn[0],'Annotation', fn[-1]), pcd_d[:,:7], fmt='%.3f %.3f %.3f %d %d %d %d')

            # shutil.copyfile(anno_filtered, fn)
    return ndir


def pcd_merged(wip_folder):
    rooms = glob(wip_folder +'/*')
    rooms.sort()
    for room in rooms:
        annos = glob(os.path.join(room, 'Annotation', '*.txt'))

        if not len(annos) == 0:
            pcd = readTXT(annos[0])
            for anno in annos[1:]:
                pcd = np.vstack((pcd, readTXT(anno)))
            np.savetxt(os.path.join(room, os.path.split(room)[-1] + '.txt'), pcd[:,:7], fmt='%.3f %.3f %.3f %d %d %d %d')
        else:
            print('skip-', room)


def readTXT(path):
    if os.path.getsize(path) == 0:
        print(path, ": empty files")
        return 

    pcd = np.loadtxt(path)
    return pcd


def npy2ply(folder, results_folder):
    cnst_label_path = './cnst_label.json'
    with open(cnst_label_path, "r") as f:
        cnst_label = json.load(f)


    paths = glob(folder + '/*.pth')
    for path in tqdm(paths):
        fn = os.path.split(path)[-1].replace('.pth', '')
        
        pd = torch.load(path)
        xyz = pd['coord']
        # rgb = pd['color']
        pred = np.load(os.path.join(results_folder, fn + '_pred.npy')) 

        rgb = np.empty_like(xyz)
        for cc in cnst_label:
            idx = pred == cc[0]
            rgb[idx] = cc[2]


        output_cloud = o3d.geometry.PointCloud()
        output_cloud.points = o3d.utility.Vector3dVector(xyz)
        output_cloud.colors = o3d.utility.Vector3dVector(rgb)

        pred_fn = os.path.join(results_folder, fn + '_pred.ply')
        o3d.io.write_point_cloud(pred_fn, output_cloud)


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


# function to read JSON file into Python object
def readJSON(filename):
    return json.load(open(filename, 'r'))

# function to write Python object to a JSON file 
def writeJSON(obj, filename):
    with open(filename, 'w') as outfile:
        json.dump(obj, outfile, indent=2)



if __name__=="__main__":
    main()

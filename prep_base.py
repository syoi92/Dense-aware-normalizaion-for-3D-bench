import os
import shutil
import numpy as np
import torch
from glob import glob
import pandas as pd
import json
import open3d as o3d


def main():
    # indoor prep-base (11) xyzrgbInsNor
    root = "../data/wip/cnstpcim_indoor"
    output = "../data/wip/cnstpcim_indoor_prep1"
    data_confidential_json = './data_confidential.json'

    anno_prep(root, output, label = "cnst_labell.json")
    data_split(output, data_confidential_json)



def data_split(source_folder, json_file, rename_files=True):
    # Load JSON mapping
    with open(json_file, "r") as f:
        file_mapping = json.load(f)

    # Ensure destination subfolders exist
    for subfolder in ["Train", "Val", "Test"]:
        os.makedirs(os.path.join(source_folder, subfolder), exist_ok=True)

    # Iterate through .pth files in source folder
    for pth_file in glob(os.path.join(source_folder, "*.pth")):
        filename = os.path.basename(pth_file)
        
        # Match the file prefix to an entry in the JSON
        for original_id, info in file_mapping.items():
            if filename.startswith(original_id):
                set_type = info.get("set")

                # Ignore cases where set is "None"
                if set_type is None or set_type.lower() == "none":
                    continue

                # Extract suffix (e.g., "_a1.pth") & Determine new filename
                suffix = filename[len(original_id):]  # Get everything after the original ID
                new_filename = f"{info['confidential']}{suffix}" if rename_files else filename


                set_folder = set_type.capitalize()  # "train" → "Train"
                dest_path = os.path.join(source_folder, set_folder, new_filename)
                shutil.move(pth_file, dest_path)  # Use shutil.copy() if you want to keep the original

                # Also copy to "Val" if the set is "test"
                if set_type.lower() == "test":
                    val_path = os.path.join(source_folder, "Val", new_filename)
                    shutil.copy(dest_path, val_path)

                print(f"Moved: {filename} → {dest_path}")
                break  # Stop checking once a match is found


def anno_prep(root, output, i_intensity = False, i_ins=True, i_normals = True, label = "cnst_labell.json"):
    with open(os.path.join(root, label), "r") as f:
        cnst_label = json.load(f)
        class_num = sum(1 for value in cnst_label.values() if value.get("indexed"))
        print(cnst_label)
    
    if not os.path.isdir(output):
        os.mkdir(output)

    paths = glob(os.path.join(root, "*/"))
    paths.sort()

    for path in paths:
        fn = os.path.basename(os.path.normpath(path))
        annos = glob(os.path.join(path, 'Annotation', '*.txt'))
        
        pcd_list = []
        for anno in annos:
            t_label = os.path.split(anno)[-1].lower()
            t_pcd = readTXT(anno)
            if not i_intensity:
                t_pcd = t_pcd[:, :6]

            sw = 1
            for key, value in list(cnst_label.items())[:-1]:
                if value.get("indexed"):  # Default is False if 'indexed' is missing
                    if value['name'] in t_label:
                        print(f"Index: {key}, {value['name']} === {anno}")
                        t_pcd = np.hstack((t_pcd, np.ones((len(t_pcd),1)) * int(key)))
                        sw = -1
                        break

            if sw > 0:
                print(f"Index: {class_num-1}, clutter === {anno}")
                t_pcd = np.hstack((t_pcd, np.ones((len(t_pcd),1)) * (class_num-1)))

            if i_ins:
                t_pcd = np.hstack((t_pcd, np.ones((len(t_pcd),1)) * -1))

            if i_normals:
                normals = normal_estimate(t_pcd)
                t_pcd = np.hstack((t_pcd, normals))

            pcd_list.append(t_pcd)
        
        pcd = np.vstack(pcd_list)
       

        if pcd.shape[1] == 11:
            ind_A1 = np.logical_and(pcd[:,0] > 0, pcd[:,1] > 0)
            ind_A2 = np.logical_and(pcd[:,0] < 0, pcd[:,1] > 0)
            ind_A3 = np.logical_and(pcd[:,0] < 0, pcd[:,1] < 0)
            ind_A4 = np.logical_and(pcd[:,0] > 0, pcd[:,1] < 0)
            
            pcd_a1 = downsample(pcd[ind_A1, :], resolution=0.01)
            pcd_a2 = downsample(pcd[ind_A2, :], resolution=0.01)
            pcd_a3 = downsample(pcd[ind_A3, :], resolution=0.01)
            pcd_a4 = downsample(pcd[ind_A4, :], resolution=0.01)
           
            writePthDict(pcd_a1, os.path.join(output,fn+'_a1.pth'))
            writePthDict(pcd_a2, os.path.join(output,fn+'_a2.pth'))
            writePthDict(pcd_a3, os.path.join(output,fn+'_a3.pth'))
            writePthDict(pcd_a4, os.path.join(output,fn+'_a4.pth'))
            print(len(pcd[ind_A1, :]), 'a1-done')
            print(len(pcd[ind_A2, :]), 'a2-done')
            print(len(pcd[ind_A3, :]), 'a3-done')
            print(len(pcd[ind_A4, :]), 'a4-done')
        
        print(fn, pcd.shape, '-done')



def writePthDict(pcd, fn):
    if pcd.shape[0] < 100000:
        return None
    
    label_prep = {'coord': pcd[:,:3].astype(np.float64),
                    'color': pcd[:,3:6].astype(np.float64),
                    'semantic_gt': pcd[:,[6]].astype(np.int64),
                    'instance_gt': pcd[:,[7]].astype(np.int64),
                    'normal': pcd[:,-3:].astype(np.float64)}
    return torch.save(label_prep, fn)


def readTXT(path):
    if os.path.getsize(path) == 0:
        print(path, ": empty files")
        return 

    pcd = np.loadtxt(path)
    return pcd


def downsample(pcd, resolution=0.001, decimal_preserved=False):
    cloud = pcd.copy()
    voxel_set = set()
    output_cloud = []

    idx = np.round(cloud[:, :3]/resolution).astype(int)
    voxels = [tuple(k) for k in idx]
    
    if not decimal_preserved:
        cloud[:, :3] = idx * resolution

    for i in range(len(voxels)):
        if not voxels[i] in voxel_set:
            output_cloud.append(cloud[i])
            voxel_set.add(voxels[i])
    return np.array(output_cloud) 


def normal_estimate(pcd):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pcd[:, :3])
    cloud.estimate_normals()
    
    return np.array(cloud.normals)


  


if __name__ == "__main__":
    main()
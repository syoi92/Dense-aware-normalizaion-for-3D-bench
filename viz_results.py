import os, json
import numpy as np
import torch
from tqdm import tqdm
import open3d as o3d
from glob import glob


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

def main():
    folder = '../_data/cnst-zone4/Test'
    results_folder = '../_data/results/tv2-zone4'
    npy2ply(folder, results_folder)


if __name__ == "__main__":
    main()
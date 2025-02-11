import os
import shutil
import numpy as np
import torch
from glob import glob
import pandas as pd
import json

def class_distribution(root, label = "cnst_label.json"):
    with open(os.path.join(root, label), "r") as f:
        cnst_label = json.load(f)
    
    columns = []
    for cls in cnst_label: 
        print(cls)
        columns.append(cls[1])
    class_num = len(cnst_label)

    paths = glob(os.path.join(root, "*/"))
    paths.sort()
    df, df_ = pd.DataFrame(), pd.DataFrame()

    # for path in paths:
    #     fn = os.path.split(path)[-1]
    #     fn = fn.split('.')[0]

    #     with h5py.File(path, 'r') as hf:
    #         seg = np.array(hf['seg'])

    #     values, counts = np.unique(seg, return_counts=True)
    #     distri = np.zeros(class_num)
    #     for idx in range(len(values)):
    #         distri[int(values[idx])] = counts[idx] 
    #     distri_ = distri / distri.sum()

    #     tmp_df = pd.DataFrame(distri.reshape(1,-1), index=[fn])
    #     tmp_df_ = pd.DataFrame(distri_.reshape(1,-1), index=[fn])

    #     df = pd.concat((df,tmp_df))
    #     df_ = pd.concat((df_,tmp_df_))
    #     print(fn, df.shape)
    
    # df.columns = columns
    # df_.columns = columns
    # df.to_csv(os.path.join(out_root, "class_dist.csv"))
    # df_.to_csv(os.path.join(out_root, "class_dist_normalized.csv"))
    
    return None

# root = "/Users/sy/CnstPCIM/wip/cnstpcim"
# output = "/Users/sy/CnstPCIM/wip/cnstpcim_prep1"
def anno_prep(root, output, is_intensity = True,label = "cnst_label.json"):
    with open(os.path.join(root, label), "r") as f:
        cnst_label = json.load(f)
    
    if not os.path.isdir(output):
        os.mkdir(output)

    paths = glob(os.path.join(root, "*/"))
    paths.sort()

    for path in paths:
        fn = os.path.basename(os.path.normpath(path))
        annos = glob(os.path.join(path, 'Annotation', '*.txt'))
        
        pcd = readTXT(annos[0])
        if not is_intensity:
            pcd = pcd[:, :6]

        sw = 1
        for lb in cnst_label:
            if lb[1] in os.path.split(annos[0])[-1].lower():
                print(lb[1], '===', annos[0])
                pcd = np.hstack((pcd, np.ones((len(pcd),1)) * lb[0]))
                sw = -1
                break
        if sw > 0:
            print('clutter ===', annos[0])
            pcd = np.hstack((pcd, np.ones((len(pcd),1)) * 22))

        for anno in annos[1:]:
            t_label = os.path.split(anno)[-1].lower()
            t_pcd = readTXT(anno)
            if not is_intensity:
                t_pcd = t_pcd[:, :6]

            sw = 1         
            for lb in cnst_label:
                if lb[1] in t_label:
                    print(lb[1], '===', anno)
                    t_pcd = np.hstack((t_pcd, np.ones((len(t_pcd),1)) * lb[0]))
                    sw = -1
                    break
            if sw > 0:
                print('clutter ===', anno)
                t_pcd = np.hstack((t_pcd, np.ones((len(t_pcd),1)) * 22))                
            
            pcd = np.vstack((pcd, t_pcd))
        
        print(fn, pcd.shape, '-done')
        torch.save(torch.tensor(pcd), os.path.join(output,fn+'.pth'))



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





    



def main():
    # path = "../data/s3dis/Area_1/conferenceRoom_1.pth"
    # sample = torch.load(path)
    # print(sample.keys())


    # wip >> folder structure relocation
    wip_folder = "/Users/sy/CnstPCIM/wip/wip_cnst"
    # wip_folder = "../data/wip/wip_cnst"
    # pcd_merged(os.path.join(os.path.split(wip_folder)[0],'cnstpcim'))

    ##
    # root = "/Users/sy/CnstPCIM/wip/cnstpcim"
    # output = "/Users/sy/CnstPCIM/wip/cnstpcim_prep1"

    root = "../data/wip/cnstpcim"
    output = "../data/wip/cnstpcim_prep1"
    anno_prep(root, output, is_intensity = True,label = "cnst_label.json")



if __name__ == "__main__":
    main()
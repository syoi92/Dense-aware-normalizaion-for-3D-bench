import numpy as np
import pandas as pd
import uuid, json, os
from glob import glob


def class_dist(root, output, label):

    columns = []
    for cls in label: 
        columns.append(cls[1])
    cls_num = len(label)
    
    scans = glob(os.path.join(root,'*[!.json]'))
    scans.sort()

    df = pd.DataFrame(columns=columns)
    for scan in scans:
        annos = glob(os.path.join(scan, 'Annotation/*.txt'))
        scan_n = os.path.split(scan)[-1].lower()

        counts = np.zeros(cls_num)
        for anno in annos:
            fn = os.path.split(anno)[-1].split('.')[0].lower()
            fn = fn.split('-')[0]
            idx = columns.index(fn)            

            try:
                with open(anno, 'r', encoding='utf-8') as file:
                    counts[idx] = +sum(1 for _ in file)
            except FileNotFoundError:
                print("File not found.")

        print(scan_n, '- totol point #:',counts.sum())
        df.loc[scan_n] = counts
    df.to_csv(os.path.join(output, "class_dist.csv"))

    return df




def main():
    root = "/Users/sy/CnstPCIM/wip/cnstpcim/indoor"
    with open("./cnst_labelL.json", "r") as f:
        cnst_label = json.load(f)

    _ = class_dist(root, './', cnst_label)
    return None


if __name__ == "__main__":
    main()

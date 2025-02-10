import os
import json
import uuid
import random
from tqdm import tqdm


def main():
    data_path = '/Users/sy/CnstPCIM/wip/cnstpcim/indoor'
    folder_encrpytion = './data_confidential.json'


    mapping = generate_confidential(data_path)
    with open(folder_encrpytion, "w") as f:
        json.dump(mapping, f, indent=4)
    print(f"Confidential renaming completed. Mapping saved to {folder_encrpytion}.")



def generate_confidential(data_path, train_ratio=0.8):
    """
    Renames each folder in the given directory to a UUID, assigns train/test labels,
    and stores the mapping in a JSON file.
    """
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    random.shuffle(folders)  # Shuffle folders for random train-test split

    mapping = {}
    num_train = int(len(folders) * train_ratio)

    for i, folder in enumerate(folders):
        new_name = str(uuid.uuid4())  # Generate UUID
        mapping[folder] = {
            "confidential": new_name,
            "set": "train" if i < num_train else "test"
        }

    return mapping



def folder_name_switch(data_path, mapping_json, encrytion=True):
    """
    Restores folder names to their original names using a stored JSON mapping.

    :param data_path: Path to the dataset directory.
    :param mapping_json: Path to the JSON file containing the mapping.
    """
    with open(mapping_json, "r") as f:
        mapping = json.load(f)

    for data, info in tqdm(mapping.items()):
        uuid_n = info['confidential']
        uuid_path = os.path.join(data_path, uuid_n)
        ori_path = os.path.join(data_path, data)

        if encrytion:
            if os.path.exists(ori_path):
                os.rename(ori_path, uuid_path)
        else:
            if os.path.exists(uuid_path):
                os.rename(uuid_path, ori_path)

if __name__ == "__main__":
    main()




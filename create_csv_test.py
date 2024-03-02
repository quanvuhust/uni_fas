import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

DATASET_PATHS = ["datasets/phase1", "datasets/phase2"]
modes = ["dev", "test"]

parts = ["p1", "p2.1", "p2.2"]



for part in parts:
    test_labels = []
    test_paths = []
    for i in range(2):
        DATASET_PATH = DATASET_PATHS[i]
        part_path = os.path.join(DATASET_PATH, part)
        test_label_path = os.path.join(part_path, modes[i] + ".txt")
        
        file = open(test_label_path, 'rt')
        Lines = file.readlines()
        for line in Lines:
            file_name = line.strip()
            path = os.path.join(DATASET_PATH, file_name)
            test_paths.append(path)
            test_labels.append(0)
        file.close()

    test_labels = np.array(test_labels)
    test_paths = np.array(test_paths)
    test_labels = np.expand_dims(test_labels, 1)
    test_paths = np.expand_dims(test_paths, 1)
    train_df = pd.DataFrame(np.concatenate((test_paths, test_labels), axis=1), columns=["filename","label"])
    train_df.to_csv('code/data/test_{}.csv'.format(part), index=False)
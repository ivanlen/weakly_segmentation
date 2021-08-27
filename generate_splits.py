import argparse
import os
import numpy as np
import json

seed = 18
np.random.seed(seed)


def main(params):
    images_list = os.listdir(params.dataset_images_path)
    valid_ids = [i.replace('top_mosaic_09cm_area', '').replace('.tif', '') for i in images_list]
    np.random.shuffle(valid_ids)
    train1 = valid_ids[:params.n_labels]
    train2 = valid_ids[params.n_labels:params.n_weak + params.n_labels]
    val = valid_ids[params.n_labels + params.n_weak:]
    # TODO: by removing this we can use it in any other configuration
    assert len(train1) == 3, len(train1)
    assert len(train2) == 23, len(train2)
    assert len(val) == 7, len(val)

    os.makedirs(params.proc_data_path, exist_ok=True)
    with open(f'{params.proc_data_path}/train_ids_pixels.json', 'w') as f:
        print('pixels', {'ids': train1})
        json.dump({'ids': train1}, f)
    with open(f'{params.proc_data_path}/train_ids_weak.json', 'w') as f:
        print('weak', {'ids': train2})
        json.dump({'ids': train2}, f)
    with open(f'{params.proc_data_path}/val_ids.json', 'w') as f:
        print('val', {'ids': val})
        json.dump({'ids': val}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_images_path', required=True)
    parser.add_argument('--proc_data_path', required=True)
    parser.add_argument('--n_labels', required=False, default=3)
    parser.add_argument('--n_weak', required=False, default=23)

    params = parser.parse_args()
    main(params)

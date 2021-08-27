import argparse
import json
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from code import image_cropping
from code import labelling


def main(params):

    # Generate splits ---------
    images_list = os.listdir(params.images_path)
    valid_ids = [i.replace('top_mosaic_09cm_area', '').replace('.tif', '') for i in images_list]
    np.random.shuffle(valid_ids)
    train1_ids = valid_ids[:params.n_labels]
    train2_ids = valid_ids[params.n_labels:params.n_weak + params.n_labels]
    val_ids = valid_ids[params.n_labels + params.n_weak:]
    # TODO: by removing this we can use it in any other configuration
    assert len(train1_ids) == 3, len(train1_ids)
    assert len(train2_ids) == 23, len(train2_ids)
    assert len(val_ids) == 7, len(val_ids)

    os.makedirs(params.proc_data_path, exist_ok=True)
    with open(f'{params.proc_data_path}/train_ids_pixels.json', 'w') as f:
        print('pixels', {'ids': train1_ids})
        json.dump({'ids': train1_ids}, f)
    with open(f'{params.proc_data_path}/train_ids_weak.json', 'w') as f:
        print('weak', {'ids': train2_ids})
        json.dump({'ids': train2_ids}, f)
    with open(f'{params.proc_data_path}/val_ids.json', 'w') as f:
        print('val', {'ids': val_ids})
        json.dump({'ids': val_ids}, f)

    folder_proc_images = f'{params.proc_data_path}/images'
    folder_proc_px_labels = f'{params.proc_data_path}/labels'
    images_list = os.listdir(params.images_path)
    valid_ids = [i.replace('top_mosaic_09cm_area', '').replace('.tif', '') for i in images_list]

    os.makedirs(folder_proc_images, exist_ok=True)
    os.makedirs(folder_proc_px_labels, exist_ok=True)

    # process images, generate tiles and labels --------
    data = []
    for sample_id in tqdm(valid_ids):
        la_path = f'{params.labels_path}/top_mosaic_09cm_area{sample_id}.tif'
        im_path = f'{params.images_path}/top_mosaic_09cm_area{sample_id}.tif'
        _labels = plt.imread(la_path)
        _image = plt.imread(im_path)

        if sample_id in train1_ids:
            split = 't1'
        elif sample_id in train2_ids:
            split = 't2'
        else:
            split = 'val'

        boxes = image_cropping.generate_cropping_boxes(_image.shape[1], _image.shape[0], params.tile_size, thresh=0.3)
        for i, box in enumerate(boxes):
            im_cropped = image_cropping.crop_np_image_using_box(_image, box)
            lab_cropped = image_cropping.crop_np_image_using_box(_labels, box)
            oh_classes, oh_percents = labelling.create_px_annotations(lab_cropped)
            tile_id = f'{sample_id}_{i}'

            tile_im_path = f'{folder_proc_images}/{tile_id}.png'
            tile_lab_path = f'{folder_proc_px_labels}/{tile_id}.png'

            tile_data = {
                'image_id': sample_id,
                'split': split,
                'tile_id': tile_id,
                'raw_im_path': im_path,
                'raw_label_path': la_path,
                'tile_im_path': tile_im_path,
                'tile_lab_path': tile_lab_path,
                'raw_box_coords': box,
                'oh_classes': oh_classes,
                'oh_color_percents': oh_percents}
            data.append(tile_data)
            pil_im_cropped = Image.fromarray(im_cropped)
            pil_lab_cropped = Image.fromarray(lab_cropped)

            pil_im_cropped.save(tile_im_path)
            pil_lab_cropped.save(tile_lab_path)

    data_df = pd.DataFrame(data)
    proc_data_json_file = f'{params.proc_data_path}/proc_data.json'
    data_df.to_json(proc_data_json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', required=True)
    parser.add_argument('--labels_path', required=True)
    parser.add_argument('--proc_data_path', required=True)
    parser.add_argument('--tile_size', required=False, default=200, type=int)
    parser.add_argument('--n_labels', required=False, default=3, type=int)
    parser.add_argument('--n_weak', required=False, default=23, type=int)

    params = parser.parse_args()
    main(params)

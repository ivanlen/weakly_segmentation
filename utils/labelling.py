import numpy as np

label_color_map = {
    'surface': [255, 255, 255],  # white
    'building': [0, 0, 255],  # blue
    'low_veg': [0, 255, 255], #
    'tree': [0, 255, 0],
    'car': [255, 255, 0],
    'background': [255, 0, 0]}

color_label_map = {
    '[255, 255, 255]': 'surface',
    '[0, 0, 255]': 'building',
    '[0, 255, 255]': 'low_veg',
    '[0, 255, 0]': 'tree',
    '[255, 255, 0]': 'car',
    '[255, 0, 0]': 'background'}

label_classes_map = {
    'surface': 0,
    'bluiding': 1,
    'low_veg': 2,
    'tree': 3,
    'car': 4}


def create_px_annotations(px_labels):
    linear_pixes = np.vstack(px_labels)
    unique_colors = np.unique(linear_pixes, axis=0)
    classes = [color_label_map[str(list(c))] for c in unique_colors]
    classes_ids = [label_classes_map[c] for c in classes if c not in 'background']
    one_hot = [1 if i in classes_ids else 0 for i in range(5)]

    total_pixes = px_labels.shape[0] * px_labels.shape[1]
    percent_colors = [np.sum(np.all(linear_pixes == uc, axis=1)) / total_pixes for uc in unique_colors]
    percet_colors_by_class = [0] * 5
    for c, percent in zip(classes_ids, percent_colors):
        percet_colors_by_class[c] = percent

    return one_hot, percet_colors_by_class

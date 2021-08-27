import itertools


def generate_1d_limits(wind, limit, thresh):
    x_left = []
    x_right = []
    if limit >= wind:
        x_lim_reached = False
        i = 0
        while not x_lim_reached:
            x_l = i * wind
            x_r = (i + 1) * wind

            if x_r <= limit:
                x_right.append(x_r)
                x_left.append(x_l)
            else:
                x_lim_reached = True
                # some extra padding
                if (x_r - limit) / wind < thresh:
                    x_r = limit
                    x_l = limit - wind
                    x_right.append(x_r)
                    x_left.append(x_l)
            i += 1
    return x_left, x_right


def generate_cropping_boxes_from_limits(x_left, x_rigth, x_bottom, x_top):
    croping_boxes = []
    x_lims = [(x_l, x_r) for x_l, x_r in zip(x_left, x_rigth)]
    y_lims = [(x_l, x_r) for x_l, x_r in zip(x_bottom, x_top)]
    bounding_boxes = list(itertools.product(x_lims, y_lims))
    for i in range(len(bounding_boxes)):
        ((x1, x2), (y1, y2)) = bounding_boxes[i]
        croping_boxes.append((x1, y1, x2, y2))
    return croping_boxes


def generate_cropping_boxes(image_width, image_height, cropping_window, thresh):
    # image_width, image_height = image.size
    x_left, x_rigth = generate_1d_limits(cropping_window, image_width, thresh)
    x_bottom, x_top = generate_1d_limits(cropping_window, image_height, thresh)
    cropping_boxes = generate_cropping_boxes_from_limits(x_left, x_rigth, x_bottom, x_top)

    return cropping_boxes


def crop_np_image_using_box(im, box):
    return im[box[1]:box[3], box[0]:box[2]]

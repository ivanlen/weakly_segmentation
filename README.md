# Weakly supervised semantic segmentation

In this project we implement a weak supervised training to generate a semantic segmentation model of the ISPRS Vaihingen dataset.


### 1. Download data
run `/download_data.sh` to download the data. It will create the folder `./data` in which the data will be downloaded and unzipped.
```bash
./download_data.sh
```

### 2. Process data
In this step first we split the data into `n_labels` images that are going to be use with the gt and `n_weak` images that are going to be used as weak labels.

For each image we generate crops of `tile_size x tile_size`.
- For the `n_labels` images we use the cropped gt segmentation map.
- For the `n_weak` images create a one hot classification label `[0, 1, ..., 0]`.
    
```bash
python generate_proc_dataset.py --images_path=./data/images/top --labels_path=./data/labels --proc_data_path=./proc_data/ --tile_size=200 --n_labels=3 --n_weak=23
```

Arguments:
 --tile_size=200 --n_labels=3 --n_weak=23
- `tile_size`: size of the tiles
- `images_path`: path of the images, `./data/images/top`
- `labels_path`: path of the images, `./data/labels`
- `proc_data_path`: path where the processed data is going to be saved, `./proc_data/`
- `n_labels`: number of images that are going to be use as strong lables (using the gt segmentation map), `3`
- `n_weak`: number of images that are going to be use as strong weak (using generated classification labels), `23`

Using this configuration the remaining 7 images are kept for the validation data.
This remaining images are also split into tiles using the same configuration as the 
Notes: 
- For the weak labels we are going to discard the `clutter/background` data.
- The labeling convention can be found in `./utils/labelling.py`


### 3. Train a segmentation model
In this project we implement a deeplabv3_resnet101 segmentation model. 

> **Model Description**
Deeplabv3-ResNet101 is constructed by a Deeplabv3 model with a ResNet-101 backbone. The pre-trained model has been trained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
We download a pretrained model on 
We freeze all the layers and we only train the classification network.

Checkout `./nnet/model.py` to for the model implementation.

```bash
python train.py --proc_data_path=./proc_data/proc_data.json --num_epochs=10 --weak_loss=average_pooling --train_data='t1' --batch_size=32
```
Arguments:    
- `proc_data_path`: path of the processed data, `./proc_data`
- `n_classes`: number of classes in the data, `6`
- `train_data`: data to use in the train. `t1` for strong labels, `t2` for weak, `t1t2` for both
- `num_epochs`: number of epochs to use in the training, `10`
- `init_lr`: initial learning rate, `0.05`. If there is an scheduler this value can change during training.
- `batch_size`: batch size XD, `32`
- `weak_loss`: weak loss that is going to be used in the training. If `None` weak labels are ignored. `average_pooling` 
- `weak_lambda`: multiplication factor used to compute the total loss, `total_loss = seg_loss + weak_lambda * weak_loss`, default=`0.1`
- `save_path`: path where the trained model, parameters and statistics are going to be saved. `./trained_models`

The trained model, history, metrics and run parameters are stored in `./trained_models`

The in traiing we augment the data by using horizontal flip transformations and [90, 180, 270] rotations.

###### Weak Loss
Here we implement a very simple weak loss: 
- For each image we compute the probability of the classes as the softmax of the logits.
- We then average the probabilities of each class to compute the mean probabilities `probs=[pc0, pc1, ..., pc4]`
- Finally we compute the Binary Cross Entropy between the class labels and the probabilities`BCE(probs, gt)`.

### Notebooks

There are some notebooks that complement the previous files.
- dataset visualization
    - explore_data.ipynb: Visualise training data and masks. A fast overview of the dataset.
    - explore_proc_data.ipynb: Visualise the tiles and masks after processing the raw data.
- results
    - evaluate_and_explore_trained_model.ipynb: visualise model segmentation maps and gt.
    - show_model_results.ipynb: loads all the results and generates a summary

### TODOs:

###### Implement more sophisticated weak losses:
- global weighted rank-pooling (GWRP) implemented in [Seed, Expand and Constrain: Three Principles for Weakly-Supervised Image Segmentation](https://arxiv.org/abs/1603.06098)

###### Make use of validation and test data:
Here we do not use train, validation and test data. We only use two splits (train and val).
If we want tune hyperparameters, use callbacks, early stopping, or any other approach, we need to use a validation set and keep the current validation/test set of 7 images unseen.
In this new scenario we use only the test set to test compute metrics and test the model. 


###### Implement classes weights
Analyse the strong data images and compute the percentage of each class in the images. We can then use this data to compute weights and balanced unrepresented classes.   

###### Change tile sizes
Here we use tiles of `200x200`. Since the model that we use need tiles of `224x224` at least we implement a resize function in the preprocessing.
If we use bigger bigger tile sizes we need refactor the code to ignore the rescale factor to 224, if not the bigger tiles are not going to have any impact.

###### Data augmentation
- Data augmentation is always 'turned on'. If we want to disable it we need to refactor the code to be able to disable it.
- The augmentation process is done on the cropped tiles generated the preprocessing step. A better approach would be to work with the raw images and crop random tiles for the image so we do not only flip or rotate a fixed set of tiles. Doing this we increase the varibility of the tiles and we have an "infinite set" of tiles.
 

### References
- Dataset: [Vaihingen data](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/)
- [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
- https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
- [Seed, Expand and Constrain: Three Principles for Weakly-Supervised Image Segmentation](https://arxiv.org/abs/1603.06098)
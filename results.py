# set paths to train and test image datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import *
from PIL import Image

from scipy.ndimage import label, generate_binary_structure

TRAIN_PATH = './severstal-steel-defect-detection/train_images/'
TEST_PATH = './severstal-steel-defect-detection/test_images/'

# load dataframe with train labels
train_df = pd.read_csv('./severstal-steel-defect-detection/train.csv')

# load my predictions
results_df = pd.read_csv('./submission_training.csv')

print(train_df.head(3))
print(results_df.head(3))


def preprocess_df(df):
    '''
    Function for train dataframe preprocessing.
    Creates additional columns 'Image' with image filename and 'Label' with label number.
    '''
    # split column
    split_df = df["ImageId_ClassId"].str.split("_", n = 1, expand = True)
    new_df = df.copy()

    # add new columns to train_df
    new_df['Image'] = split_df[0]
    new_df['Label'] = split_df[1]

    # check the result
    return new_df



train_df = preprocess_df(train_df)
results_df = preprocess_df(results_df)

print(train_df.columns.values.tolist())
print(train_df.head(1).values.tolist())


def get_row_by_img_clsid(df, fname):
    return df.loc[df["ImageId_ClassId"] == fname]


def get_mask_by_img_clsid(df, fname):
    try:
        row = get_row_by_img_clsid(df, fname)
        return row["EncodedPixels"].values[0]
    except IndexError:
        return None


def rle2maskResize(rle):
    # CONVERT RLE TO MASK
    if (pd.isnull(rle)) | (rle == '') | (rle == '-1'):
        return np.zeros((256, 1600), dtype=np.uint8)

    height = 256
    width = 1600
    mask = np.zeros(width * height, dtype=np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2] - 1
    lengths = array[1::2]
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1

    return mask.reshape((height, width), order='F')

def compute_iou(y_true, y_pred):
    a = y_true.tolist()
    return jaccard_score(y_true.tolist(), y_pred.tolist(), average='micro')


# def compute_iou(target, prediction):
#     '''
#     Function to compute IOU metric
#     See:
#     https://www.jeremyjordan.me/evaluating-image-segmentation-models/
#     '''
#     target = np.asarray(target, dtype=float)
#     prediction = np.asarray(prediction, dtype=float)
#
#     smooth = 1e-5  # smoothing for empty masks
#
#     intersection = np.logical_and(target, prediction)
#     union = np.logical_or(target, prediction)
#     iou_score = np.sum(intersection) / (np.sum(union) + smooth)
#
#     return iou_score



def plot_mask(image_filename):
    '''
    Function to plot an image and true/predicted segmentation masks.
    INPUT:
        image_filename - filename of the image (with full path)
    '''
    img_id = image_filename.split('/')[-1]
    image = Image.open(image_filename)
    train = train_df.fillna('-1')
    pred = results_df.fillna('-1')

    rle_masks = train[(train['Image'] == img_id)]['EncodedPixels'].values
    pred_masks = pred[(pred['Image'] == img_id)]['EncodedPixels'].values
    # ImageId_ClassId
    fig, axs = plt.subplots(4, 2, figsize=(20, 7))

    iou = 0
    for defect in range(1, pred_masks.size+1):
        rle_mask = rle_masks[defect - 1]
        pred_mask = pred_masks[defect - 1]
        np_mask = np.zeros((256, 1600), dtype=np.uint8)
        np_mask_pred = 0

        if rle_mask != '-1':
            np_mask = rle2maskResize(rle_mask)
            axs[defect - 1, 0].imshow(image)
            axs[defect - 1, 0].imshow(np_mask, alpha=0.5, cmap="Reds")
            axs[defect - 1, 0].axis('off')
            axs[defect - 1, 0].set_title('Mask with defect #{}'.format(defect))
        else:
            axs[defect - 1, 0].imshow(image)
            axs[defect - 1, 0].axis('off')
            axs[defect - 1, 0].set_title('No defects type #{}'.format(defect))

        if pred_mask != '-1':
            np_mask_pred = rle2maskResize(pred_mask)
            axs[defect - 1, 1].imshow(image)
            axs[defect - 1, 1].imshow(np_mask_pred, alpha=0.5, cmap="Reds")
            axs[defect - 1, 1].axis('off')
            axs[defect - 1, 1].set_title('Prediction for mask with defect #{}'.format(defect))
        else:
            axs[defect - 1, 1].imshow(image)
            axs[defect - 1, 1].axis('off')
            axs[defect - 1, 1].set_title('No prediction for defects type #{}'.format(defect))

        # calculate average IOU for all defects
        iou += compute_iou(np_mask, np_mask_pred)

    plt.suptitle('IOU for image: {:.2f}'.format(iou), fontsize=16)

    plt.show()


# plot_mask(TRAIN_PATH + '0002cc93b.jpg')


def plot_mask_by_id(idx):
    '''
    Plots true mask and predicted mask by id in train_df
    '''
    image_name = train_df.iloc[idx]['Image']
    image_filename = TRAIN_PATH + image_name
    image = Image.open(image_filename)

    rle_mask = train_df.iloc[idx]['EncodedPixels']
    img_name = train_df.iloc[idx]["ImageId_ClassId"]
    pred_mask = get_mask_by_img_clsid(results_df, img_name)

    defect = train_df.iloc[idx]['Label']

    true = rle2maskResize(rle_mask)
    pred = rle2maskResize(pred_mask)

    fig, axs = plt.subplots(1, 2, figsize=(20, 3.5))

    iou = compute_iou(true, pred)

    axs[0].imshow(image)
    axs[0].imshow(true, alpha=0.5, cmap="Reds")
    axs[0].axis('off')
    axs[0].set_title('Mask with defect #{}'.format(defect))

    axs[1].imshow(image)
    axs[1].imshow(pred, alpha=0.5, cmap="Reds")
    axs[1].axis('off')
    axs[1].set_title('Predicted mask for defect #{}'.format(defect))

    plt.suptitle('IOU for image: {:.2f}'.format(iou), fontsize=16)

    plt.show()


def get_mask(line_id):
    '''
    Function to visualize the image and the mask.
    INPUT:
        line_id - id of the line to visualize the masks
    RETURNS:
        np_mask - numpy segmentation map
    '''

    # convert rle to mask
    rle = train_df.loc[line_id]['EncodedPixels']

    np_mask = rle2maskResize(rle)
    np_mask = np.clip(np_mask, 0, 1)

    return np_mask


def add_mask_areas(train_df):
    '''
    Helper function to add mask area as a new column to the dataframe
    INPUT:
        train_df - dataset with training labels
    '''
    masks_df = train_df.copy()
    masks_df['Area'] = 0

    for i, row in masks_df.iterrows():
        masks_df['Area'].loc[i] = np.sum(get_mask(i))

    return masks_df


def add_mask_number(train_df):
    '''
    Helper function to add mask area as a new column to the dataframe
    INPUT:
        train_df - dataset with training labels
    '''
    masks_df = train_df.copy()
    masks_df['NumMasks'] = 0

    s = generate_binary_structure(2, 2)

    for i, row in masks_df.iterrows():
        mask = get_mask(i)

        if np.sum(mask) > 0:
            labeled_array, labels = label(mask, structure=s)
            masks_df['NumMasks'].loc[i] = labels
        else:
            masks_df['NumMasks'].loc[i] = 0

    return masks_df

# masks_df = add_mask_number(train_df)


def get_iou_by_id(idx):

    rle_mask = train_df.iloc[idx]['EncodedPixels']
    img_name = train_df.iloc[idx]["ImageId_ClassId"]
    pred_mask = get_mask_by_img_clsid(results_df, img_name)

    true = rle2maskResize(rle_mask)
    pred = rle2maskResize(pred_mask)

    iou = compute_iou(true, pred)
    return iou


train_img_ids = train_df.index.values


def has_mask(row):
    return isinstance(row['EncodedPixels'], str)


def calculate_avarage_iou(train_img_ids):
    total_iou = 0
    for idx in train_img_ids:
        if has_mask(train_df.iloc[idx]):
            total_iou += get_iou_by_id(idx)
    average_iou = total_iou/len(train_img_ids)
    return average_iou

# print("Average IOU:")
# print(calculate_avarage_iou(train_img_ids))


for idx in train_img_ids[30:40]:
    plot_mask_by_id(idx)

#
#
# many_masks_df = masks_df[masks_df['NumMasks'] > 10]
# line_ids = many_masks_df.index.values
# #
# #
# # rnd_idx = line_ids[np.random.randint(len(line_ids))]
# # plot_mask_by_id(rnd_idx)
# #
# # rnd_idx = line_ids[np.random.randint(len(line_ids))]
# # plot_mask_by_id(rnd_idx)
# #
# #
# #
# #
# total_iou = sum([get_iou_by_id(iid) for iid in line_ids])
# average_iou = total_iou/len(line_ids)
# print("Average IOU:")
# print(average_iou)

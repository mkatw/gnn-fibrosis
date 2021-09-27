from argparse import ArgumentParser
import numpy as np
import pandas as pd
from pathlib import Path
import math
from skimage import color, filters, morphology, img_as_ubyte
from scipy import ndimage as ndi
from openslide import OpenSlide
from skimage.io import imsave
import os
import cv2
import random
import matplotlib.pyplot as plt
from joblib import load
from sklearn.cluster import KMeans

from PIL import Image
from sklearn.preprocessing import normalize

from PIL import Image
from torchvision.transforms.functional import to_tensor
import torch
from torch import nn
from wsi_reader import WSIReader, TiffReader


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

        self.resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        self.resnet.fc = nn.Identity()  # a fake layer

        # we just substituted the original fc layer with the identity placeholder layer so we can extract the embedding

    def forward(self, x):
        return self.resnet(x)  # latent space representation for clustering


def normalize_input(x):
    c = x.shape[0]
    mean = x.view(c,-1).mean(dim=-1)[:,None,None].expand_as(x)
    std = x.view(c,-1).std(dim=-1)[:,None,None].expand_as(x)
    return (x-mean)/(std+1e-9)  # 1e-9 is for numerical stability


def load_image_and_map(map_path):
    # load the original slide and the collagen segmentation map
    # the original slide is used to identify tissue vs background tiles

    map_path = Path(os.path.abspath(map_path))
    image_path = (str(Path(map_path.parents[1])) + '/original_scans/' + map_path.stem + '.svs')  # find the original slide
    print(image_path)
    collagen_map = TiffReader(str(map_path))
    image = TiffReader(str(image_path))

    return image, collagen_map


def get_output_dimensions(image, collagen_map, slide_format, level, tile_size):

    if slide_format == 'svs' and level >= 1:
        svs_level = level - 1  # svs files don't have level 1 (downsample 2)
        image_level = svs_level
    else:
        image_level = level

    map_level = level
    width_ds, height_ds = image.get_dimensions(image_level)
    width_map, height_map = collagen_map.get_dimensions(map_level)
    if (width_ds != width_map) or (height_ds != height_map):
        print("WARNING: image and map are of different dimensions")
    tiles_horizontal = int(math.floor(width_map / tile_size))
    tiles_vertical = int(math.floor(height_map / tile_size))

    print('Using map level ' + str(map_level) + ' with dimensions: ' + str(width_map) + ' ' + str(height_map))
    #print("Map downsample: " + str(collagen_map.level_downsamples[map_level]))
    print('Using image level ' + str(image_level) + ' with dimensions: ' + str(width_ds) + ' ' + str(height_ds))
    #print("Image downsample: " + str(image.level_downsamples[image_level]))
    print(str(tiles_horizontal) + ' x ' + str(tiles_vertical) + ' output tiles')
    print('Full image dimensions: ' + str(image.get_dimensions(0)))

    return tiles_horizontal, tiles_vertical, image_level, map_level


def find_tissue_mask_threshold(image, min_threshold=0.88):

    thumbnail_level = 2  # level 2 is often the max level for svs biopsy files

    try:
        image_slide, _ = image.read_region((0, 0), thumbnail_level, image.get_dimensions(thumbnail_level))
    except:
        image_slide, _ = image.read_region((0, 0), 0, image.get_dimensions(0))  # in case slide is very small

    np_image = (image_slide * 255).astype(np.uint8)
    grey_image = color.rgb2gray(np_image)
    mask_threshold = filters.threshold_otsu(grey_image)

    if mask_threshold < min_threshold:
        mask_threshold = min_threshold  # safety threshold for weakly stained slides

    return mask_threshold


def get_tile(slide, level, slide_level, tile_size, t_h, t_v):

    x = t_h * 2 ** level * tile_size  # level 0 equivalent coordinates
    y = t_v * 2 ** level * tile_size

    tile_image, _ = slide.read_region((x, y), slide_level, (tile_size, tile_size))

    return tile_image


def is_tissue(tile, mask_threshold, cutoff):

    grey_image = color.rgb2gray(tile)  # svs files are 0-255, converted to BW are 0-1
    tissue_mask = (grey_image < mask_threshold) & (grey_image > 0)  # the 0 is for the padded black border
    tissue_mask_filled = ndi.binary_fill_holes(tissue_mask)
    tissue_ratio = (sum(sum(tissue_mask_filled)) / tissue_mask.size)

    if tissue_ratio > cutoff:
        return True

    else:
        return False


def extract_features(tile, model):

    model.cuda()
    feature_list = []

    model.eval()
    with torch.no_grad():

        tile = color.gray2rgb(tile)
        img_data = to_tensor(tile).cuda()
        img_data = normalize_input(img_data)

        features = model(img_data.unsqueeze(0)).squeeze(0).cpu().numpy()
        feature_list.append(features)

    feature_list = normalize(feature_list, axis=1)  # L2 normalisation across samples

    return feature_list


def get_predictions(slide, map, level, tiles_horizontal, tile_size, tiles_vertical, image_level, mask_threshold):

    predictions = np.full((tiles_horizontal, tiles_vertical), -1)
    features_map = np.zeros((512, tiles_horizontal, tiles_vertical))
    kmeans = load('res_net_kmeans4.joblib')
    model = ResNet()

    for t_h in range(tiles_horizontal):
        for t_v in range(tiles_vertical):

            tile = get_tile(slide, level, image_level, tile_size, t_h, t_v)
            tile = (tile * 255).astype(np.uint8)
            map_tile = get_tile(map, level, image_level, tile_size, t_h, t_v)

            if is_tissue(tile, mask_threshold, 0.3):
                features = extract_features(map_tile, model)

                features_map[:, t_h, t_v] = features
                predictions[t_h, t_v] = kmeans.predict(features)

            else:
                predictions[t_h, t_v] = -1

    # translating prediction classes to collagen classes:
    collagen_classes = {-1: -1, 0: 0, 1: 1, 2: 2, 3: 3}  # use this to rename the clusters if needed: they are assigned randomly
    collagen_predictions = np.vectorize(collagen_classes.get)(predictions)

    return features_map, collagen_predictions


def get_casewise_percentages(collagen_predictions):

    hist, bin_edges = np.histogram(collagen_predictions.flatten(), bins=[0, 1, 2, 3, 4])
    # print(max(collagen_predictions.flatten()))
    # print(hist)
    # print(bin_edges)
    n_tiles = sum(hist)

    return hist/n_tiles


def plot_results(slide, collagen_predictions, dataset, case_name, output_dir):
    # plot a downsampled overlay of slide and predictions

    clusters = [0, 1, 2, 3]
    number_clusters = len(clusters)

    upscale_parameter = 5
    collagen_predictions = np.transpose(collagen_predictions)
    predictions_upsampled = collagen_predictions.repeat(upscale_parameter, axis=0).repeat(upscale_parameter, axis=1)
    target_dims = predictions_upsampled.shape

    for level in range(slide.get_level_count()):
        dims = slide.get_dimensions(level)
        if dims[0] <= target_dims[0]:
            dims = slide.get_dimensions(max(level - 1, 0))
            break

    downsample = slide.levels_downsample[max(level - 1, 0)]
    thumbnail, _ = slide.get_downsampled_slide(downsample)
    thumbnail = (thumbnail * 255).astype(np.uint8)
    thumbnail = cv2.resize(thumbnail, target_dims, interpolation=cv2.INTER_CUBIC)

    histogram_dir = str(output_dir) + '/histograms/'
    overlay_dir = str(output_dir) + '/overlays/'

    try:
        os.makedirs(histogram_dir)
        os.makedirs(overlay_dir)
    except OSError:
        pass

    predictions_upsampled = np.ma.array(predictions_upsampled, mask=(predictions_upsampled == -1))  # making backgroud transparent

    fig = plt.figure()
    fig.set_size_inches(9, 8)
    plt.imshow(predictions_upsampled, cmap='RdYlGn_r')
    plt.colorbar(ticks=[0, 1, 2, 3])
    plt.imshow(thumbnail, alpha=0.8)
    fig.suptitle(case_name, fontsize=16)
    plt.axis('off')
    file_name = overlay_dir + dataset + '_' + case_name + '_' + str(number_clusters) + '_prediction_overlay.png'
    plt.savefig(file_name, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=None,
                transparent=False, bbox_inches=None, pad_inches=0, metadata=None)

    fig = plt.figure()
    fig.set_size_inches(10, 8)
    plt.hist(collagen_predictions.flatten(), range=(0, 3))
    plt.xticks(clusters)
    fig.suptitle(case_name, fontsize=16)
    file_name = histogram_dir + dataset + '_' + case_name + '_' + str(number_clusters) + '_histogram.png'
    plt.savefig(file_name, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=None,
                transparent=False, bbox_inches=None, pad_inches=0, metadata=None)


def process_slide(slide, map, level, tile_size):

    mask_threshold = find_tissue_mask_threshold(slide)
    tiles_horizontal, tiles_vertical, image_level, map_level = get_output_dimensions(slide, map, slide_format='svs', level, tile_size)
    features, collagen_predictions = get_predictions(slide, map, level, tiles_horizontal, tile_size, tiles_vertical, image_level, mask_threshold)

    return features, collagen_predictions


def main(args):

    map_path = args.map_path
    dataset = args.dataset
    case_name = map_path.stem
    level = 0
    tile_size = 256
    output_dir = Path('./plots')

    slide, map = load_image_and_map(map_path)
    features_map, collagen_predictions = process_slide(slide, map, level, tile_size)

    file_name = './prediction_maps/' + case_name + '_prediction.npy'
    np.save(file_name, collagen_predictions)

    features_map_file_name = './prediction_maps/' + case_name + '_features_map.npy'
    np.save(features_map_file_name, features_map)

    plot_results(slide, collagen_predictions, dataset, case_name, output_dir)

    #percentages = get_casewise_percentages(collagen_predictions)

    #data = {case_name: percentages}
    #df = pd.DataFrame.from_dict(data, orient='index')
    #file_name = dataset + '_res_net_prediction_percentages.csv'
    #df.to_csv(file_name, mode='a', header=False)


parser = ArgumentParser(description='Tile classification')
parser.add_argument('map_path', type=Path)
parser.add_argument('dataset', type=str, default='')
args = parser.parse_args()

main(args)

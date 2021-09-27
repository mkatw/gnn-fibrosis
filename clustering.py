import numpy as np
import os, shutil, glob, os.path, re
import random
import math
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from pylab import get_cmap
from argparse import ArgumentParser
import pandas as pd

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import normalize
from joblib import dump, load

from PIL import Image
from torchvision.transforms.functional import to_tensor
import torch
from torch import nn
from tensorflow.keras.models import Model, load_model
import tensorflow as tf


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


def extract_features(im_dir, model_name, fraction_used=0.1):

    if model_name == 'resnet':
        model = ResNet()
        model.cuda()
        model.eval()
    else:
        print('Unknown model', model_name)
        return

    files = glob.glob(os.path.join(im_dir, '*.png'))

    random.shuffle(files)  # shuffling tiles to avoid bias to particular cases
    subset = int(math.floor(len(files) * fraction_used))  # take a subset of all tiles
    files = files[:subset]

    files.sort()
    feature_list = []

    with torch.no_grad():
        for index, image_path in enumerate(files):
            print("    Status: %s / %s" %(index, len(files)), end="\r")

            img = Image.open(image_path).convert('RGB')
            img_data = to_tensor(img).cuda()
            img_data = normalize_input(img_data)

            features = model(img_data.unsqueeze(0)).squeeze(0).cpu().numpy()
            feature_list.append(features)

    feature_list = normalize(feature_list, axis=1)  # L2 normalisation across samples

    return feature_list, files


def make_clusters(feature_list, number_clusters, algorithm):

    if algorithm == 'kmeans':
        clusters = KMeans(n_clusters=number_clusters, random_state=0).fit(np.array(feature_list))

    if algorithm == 'minibatch_kmeans':
        clusters = MiniBatchKMeans(n_clusters=number_clusters, random_state=0).fit(np.array(feature_list))

    if algorithm == 'spherical_kmeans':
        clusters = SphericalKMeans(n_clusters=number_clusters, random_state=0).fit(np.array(feature_list))

    return clusters


def analyse_clusters(feature_list, clusters, dataset_name, tiles_lvl_dims, model):

    number_clusters = max(clusters.labels_) + 1  # label indexing starts at 0

    reduced_data = PCA(n_components=3).fit_transform(np.array(feature_list))

    # Cluster silhouette analysis

    silhouette_avg = silhouette_score(np.array(feature_list), clusters.labels_)
    print("For n_clusters =", number_clusters,
          "the average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(np.array(feature_list), clusters.labels_)

    # Plotting

    fig = plt.figure()
    fig.set_size_inches(18, 7)

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    cm = get_cmap('gist_rainbow')

    # scatterplot

    for cluster in range(number_clusters):
        points = reduced_data[clusters.labels_ == cluster, :]
        print(cluster, points.shape)
        colour = mcolors.to_hex(cm(1. * cluster / number_clusters))
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=colour)

    ax1.set_title('K-means clustering on the ' + dataset_name + ' dataset (ResNet18 encoding).\n' + tiles_lvl_dims)

    # silhouette plot

    ax2 = fig.add_subplot(1, 2, 2)

    y_lower = 10
    for cluster in range(number_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[clusters.labels_ == cluster]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        colour = mcolors.to_hex(cm(1. * cluster / number_clusters))
        ax2.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=colour, edgecolor=colour, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax2.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax2.set_title("Cluster silhouette plot")
    ax2.set_xlabel("The silhouette coefficient values")
    ax2.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax2.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax2.set_yticks([])  # Clear the yaxis labels / ticks
    ax2.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    file_name = './clustering_plots/' + model + '/' + dataset_name + '_' + tiles_lvl_dims + '_' + str(number_clusters) + '_clusters.png'
    plt.savefig(file_name, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)


def plot_centroid_images(im_dir, feature_list, files, clusters, model):

    number_clusters = max(clusters.labels_) + 1
    dataset_name = (str(Path(im_dir).parents[0].stem)).split('_')[0]
    tiles_lvl_dims = str(Path(im_dir).stem)

    fig = plt.figure()
    fig.set_size_inches(18, 7)
    fig, axes = plt.subplots(nrows=number_clusters, ncols=10)
    fig.suptitle('Images closest to cluster centre', fontsize=16)

    for cluster in range(number_clusters):
        distance_to_centroid = clusters.transform(np.array(feature_list))[:, cluster]
        indices_10_closest = np.argsort(distance_to_centroid)[::][:10]

        for j in range(10):
            image = plt.imread(files[indices_10_closest[j]])
            axes[cluster, j].imshow(image)
            axes[cluster, j].set_title(str(cluster))
            axes[cluster, j].set_axis_off()

    file_name = './clustering_plots/' + model + '/' + dataset_name + '_' + tiles_lvl_dims + '_' + str(number_clusters) + '_centre_images.png'
    plt.savefig(file_name, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)


def get_k_nearest_of_random(feature_list, files, k, dataset_name, tiles_lvl_dims, model):
    # clustering sanity check
    # plot k nearest neighbours of 5 random tiles to check if they look similar

    neigh = NearestNeighbors(n_neighbors=k, radius=0.4)
    neigh.fit(feature_list)
    num_random_images = 5
    fig = plt.figure()
    fig.set_size_inches(6, 18)
    fig, axes = plt.subplots(nrows=num_random_images, ncols=k)
    fig.suptitle('Closest neighbours of random tiles', fontsize=16)

    for i in range(num_random_images):
        random_point = random.choice(feature_list).reshape(1, -1)
        neighbours = neigh.kneighbors(X=random_point, n_neighbors=k, return_distance=True)[1][0]

        for j, neighbour in enumerate(neighbours):
            im = plt.imread(files[neighbour])
            axes[i, j].imshow(im)
            axes[i, j].set_axis_off()

    file_name = './clustering_plots/' + model + '/' + dataset_name + '_' + tiles_lvl_dims + '_k_nearest_of_random.png'
    plt.savefig(file_name, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=None,
                transparent=False, bbox_inches=None, pad_inches=None, metadata=None)


def plot_random_from_clusters(im_dir, files, clusters, model):
    # plot random tiles from each cluster

    number_clusters = max(clusters.labels_) + 1
    dataset_name = 'name'

    tiles_lvl_dims = str(Path(im_dir).stem)
    number_images = 10

    fig = plt.figure()
    fig.set_size_inches(18, 6)
    fig, axes = plt.subplots(nrows=number_clusters, ncols=number_images)
    fig.suptitle('Random samples from each cluster', fontsize=16)

    for cluster in range(number_clusters):
        print(cluster)
        for i in range(number_images):
            cluster_indices = np.where(clusters.labels_ == cluster)[0]
            random_file = files[random.choice(cluster_indices)]
            random_image = plt.imread(random_file)
            print(random_file)
            axes[cluster, i].imshow(random_image)
            axes[cluster, i].set_title(str(cluster))
            axes[cluster, i].set_axis_off()

    file_name = './clustering_plots/' + model + '/' + dataset_name + '_' + tiles_lvl_dims + '_' + str(number_clusters) + '_clusters_' + str(number_images) + '_random.png'
    plt.savefig(file_name, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=None,
                transparent=False, bbox_inches=None, pad_inches=None, metadata=None)


def is_tissue(image, mask_threshold, cutoff):

    grey_image = color.rgb2gray(color.rgba2rgb(image))  # svs files are 0-255, converted to BW are 0-1
    tissue_mask = (grey_image < 1 * mask_threshold) & (grey_image > 0.01)  # the 0.01 is for the padded black border
    tissue_mask_filled = ndi.binary_fill_holes(tissue_mask)
    tissue_ratio = (sum(sum(tissue_mask_filled)) / tissue_mask.size)

    if tissue_ratio > cutoff:
        return True

    else:
        return False


def main(args):

    im_dir = '/dir/with/tiles/for/clustering'

    dataset_name = 'name'
    tiles_lvl_dims = str(Path(im_dir).stem)

    if args.use_precomputed_features:

        feature_list = np.load('featurelist.npy')
        files = np.load('files.npy')

    else:

        feature_list, files = extract_features(im_dir, model_name=args.model)
        np.save('featurelist.npy', np.array(feature_list))
        np.save('files.npy', np.array(files))

    if args.no_clustering:
        return

    else:

        k = 10
        get_k_nearest_of_random(feature_list, files, k, dataset_name, tiles_lvl_dims, model)

        min_number_clusters = 3
        max_number_clusters = 9

        for number_clusters in range(min_number_clusters, max_number_clusters):
            clusters = make_clusters(feature_list, number_clusters, algorithm='minibatch_kmeans')
            dump(clusters, 'res_net_kmeans.joblib')
            #clusters = load('res_net_kmeans.joblib')
            analyse_clusters(feature_list, clusters, dataset_name, tiles_lvl_dims, model)
            plot_centroid_images(im_dir, feature_list, files, clusters, model)
            plot_random_from_clusters(im_dir, files, clusters, model)


parser = ArgumentParser(description='Perform clustering analysis')
parser.add_argument('--model', default='resnet', type=str)
parser.add_argument('--no_clustering', default=False, action='store_true')
parser.add_argument('--use_precomputed_features', default=False, action='store_true')
args = parser.parse_args()


main(args)

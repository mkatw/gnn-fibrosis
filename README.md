# gnn-fibrosis

This is the code from the "Early Detection of Liver Fibrosis Using Graph Convolutional Networks" paper originally presented at the MICCAI'21 conference.
The purpose is to predict fibrosis grade from a PSR-stained liver biopsy. The graph construction pipeline is designed to explicitly mimic liver tissue architecture.

![image](https://user-images.githubusercontent.com/50372773/134362166-e995826a-cd64-403c-872e-c93e355c6c17.png)

The pipeline is divided into two main parts:

1) (Tile clustering + tile classification) for tissue landmark identification
2) Biopsy graph construction for biopsy grade prediction


## Tile clustering


The clustering pipeline takes in a folder of tissue tiles saved as image files and performs k-means clustering of the tiles based on tile deep features. 
In the paper we used pre-segmented binary collagen tiles as input. 

<img width="807" alt="Screenshot 2021-09-22 at 15 31 51" src="https://user-images.githubusercontent.com/50372773/134363533-5801f82e-2a36-4bc1-9049-f1929f4af17e.png">

## Tile classification

The tile classification takes in a biopsy slide and performs collagen pattern classification on each tile separately, according to the learnt clusters. The classification output is saved in an array of tile-by-tile classification results, where each tile is represented by an integer in the range [-1, 3]. Background tiles are encoded with -1, and 0 to 3 encode fibrosis pattern subtypes.

The purpose of this part is to identify the liver tissue landmarks, i.e. dense collagen regions (tile subtype 3: red), which later become central nodes in the liver graph.

## Biopsy graph construction

This part performs a Voronoi tesselation over the tile array. Centroids of dense-collagen regions become the centre nodes for the tesselation. 

Each node in the graph represents one tile from the biopsy. The nodes contain tile features extracted with an ImageNet pre-trained deep network. The edge weights explicitly represent the distance between the tile and the centre node. 

The reason for choosing this particular predefined graph structure is that liver tissue has a highly regular architecture centred around portal tracts, which are made of dense collagen.

## Grade prediction

Here we are using a GNN (GCN, GIN or GAT) to predict fibrosis grade from the constructed tissue graph. 
![image](https://user-images.githubusercontent.com/50372773/134362315-3871725f-5f80-4719-8f4b-c2d1b01fb375.png)




<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

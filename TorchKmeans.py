import time
import torch
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor
import os
import pickle

def KMeans(x, K=10, Niter=10, verbose=True, device='cuda:0'):
    
    use_cuda = 'cuda' in device
    dtype = torch.float32 if use_cuda else torch.float64
    
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c

def load_clusterset(cls_inf_name) : 
    with open(cls_inf_name, 'rb') as f:
        (data, cl, c, numdata_each_cluster) = pickle.load(f)
        return data, cl, c, numdata_each_cluster

def save_clusterset(cls_inf_name, data, cl, c, numdata_each_cluster) : 
    with open(cls_inf_name, 'wb') as f :
        pickle.dump((data, cl, c, numdata_each_cluster), f)

def clusterset_with_gm_load_or_make(generative_model, train_loader, inception_model_info_path='../inception_model_info/', \
                                    K=100, device='cuda:0', shuffle=True, batch_size=2048) : 
    os.makedirs(inception_model_info_path, exist_ok=True)
    cls_inf_name = inception_model_info_path+generative_model.trainloaderinfo_to_hashedname(train_loader) + '_cls_info.pickle'
    if os.path.exists(cls_inf_name) : 
        print("use pickle file : %s", cls_inf_name)
        generative_model.load_or_make(train_loader)
        data, cl, c, numdata_each_cluster = load_clusterset(cls_inf_name)
    else : 
        generative_model.load_or_make(train_loader, force_make=True)
        real_feature_tensor_cuda = torch.tensor(generative_model.real_feature_np, device=device)
        cl, c = KMeans(real_feature_tensor_cuda, K, Niter=100)
        numdata_each_cluster = torch.bincount(cl)
        data = generative_model.real_images
        save_clusterset(cls_inf_name, data, cl, c, numdata_each_cluster)
        print("make and save pickle file : %s", cls_inf_name)
        
    from torch.utils.data import  TensorDataset, DataLoader

    dataset = TensorDataset(data, cl)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)    
        

    #dataloader(image, cluster_num)
    #num of image feature in each cluster
    return dataloader, numdata_each_cluster
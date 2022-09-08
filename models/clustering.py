import torch
import torchvision.transforms as tf
import numpy as np
import faiss

from fast_pytorch_kmeans import KMeans as torch_KMeans

def get_normal_descriptors(descriptors_list):
  """ Returns normalized descriptor array
    :param descriptor list: list of descriptors [1,1,N,D]
    :return: normalized descriptor array [N,D]
  """
  all_descriptors = np.ascontiguousarray(np.concatenate(descriptors_list, axis=2)[0, 0])
  normalized_all_descriptors = all_descriptors.astype(np.float32)
  faiss.normalize_L2(normalized_all_descriptors)  # in-place operation
  return normalized_all_descriptors


def get_K_means(descriptors_list, sample_interval, elbow, 
                n_cluster_range = list(range(1, 15))):
  """ Returns a fitted KMeans algorithm.
    :param descriptors_list: list of descriptors
    :param sample_interval: sample every ith descriptor before applying clustering.
    :param elbow: elbow coefficient to set number of clusters.
  """  
  # Sample normalized descriptors
  normalized_all_descriptors = get_normal_descriptors(descriptors_list)
  sampled_descriptors_list = [x[:, :, ::sample_interval, :] for x in descriptors_list]
  normalized_all_sampled_descriptors = get_normal_descriptors(sampled_descriptors_list)
  
  # Fit until elbow point
  sum_of_squared_dists = []
  for n_clusters in n_cluster_range:
    print("Attempting K=%d"%n_clusters)
    algorithm = faiss.Kmeans(d=normalized_all_sampled_descriptors.shape[1], k=n_clusters, niter=300, nredo=10)
    algorithm.train(normalized_all_sampled_descriptors.astype(np.float32))
    squared_distances, labels = algorithm.index.search(normalized_all_descriptors.astype(np.float32), 1)
    objective = squared_distances.sum()
    sum_of_squared_dists.append(objective / normalized_all_descriptors.shape[0])
    if (len(sum_of_squared_dists) > 1 and sum_of_squared_dists[-1] > elbow * sum_of_squared_dists[-2]):
      break
  #centroids = np.asarray( algorithm.centroids )
  return algorithm, labels 

from pytorch_metric_learning import distances

def get_K_means_remake(descriptors_list, sample_interval, elbow, 
                n_cluster_range = list(range(1, 15))):
  """ Returns a fitted KMeans algorithm.
    :param descriptors_list: list of descriptors
    :param sample_interval: sample every ith descriptor before applying clustering.
    :param elbow: elbow coefficient to set number of clusters.
  """  
  # Sample normalized descriptors
  normalized_all_descriptors = get_normal_descriptors(descriptors_list)
  sampled_descriptors_list = [x[:, :, ::sample_interval, :] for x in descriptors_list]
  normalized_all_sampled_descriptors = get_normal_descriptors(sampled_descriptors_list)
  
  # Fit until elbow point
  sum_of_squared_dists = []
  for n_clusters in n_cluster_range:
  
    if n_clusters > normalized_all_sampled_descriptors.shape[0]:
      if n_clusters == 1:
        algorithm = torch_KMeans(n_clusters=n_clusters, verbose=1, max_iter=300)
        normalized_all_sampled_descriptors = torch.Tensor(normalized_all_sampled_descriptors).cpu()
    
        algorithm.centroids = torch.zeros((1,normalized_all_descriptors.shape[-1]))
        labels = np.zeros((normalized_all_descriptors.shape[0], 1))
      break
    
    algorithm = torch_KMeans(n_clusters=n_clusters, verbose=1, 
                            max_iter=300, tol=1e-8)
    normalized_all_sampled_descriptors = torch.Tensor(normalized_all_sampled_descriptors).cpu()
    
    labels = algorithm.fit_predict(normalized_all_sampled_descriptors)
    normalized_all_descriptors = torch.Tensor(normalized_all_descriptors).cpu()
    d_fn = distances.LpDistance(p=2, power=1,normalize_embeddings=False)
    d = d_fn(normalized_all_descriptors, algorithm.centroids)
    #labels = d.argmax(dim=1, keepdim=True).cpu().detach().numpy().astype(np.uint16) # N,1
    labels = d.cpu().detach().numpy().argmax(axis=1)[:,np.newaxis] # N,1
    objective = d.max(dim=1).values.sum().cpu().detach().numpy()
    sum_of_squared_dists.append(objective / normalized_all_descriptors.shape[0])
    if (len(sum_of_squared_dists) > 1 and sum_of_squared_dists[-1] > elbow * sum_of_squared_dists[-2]):
      break 
  print("Got %s clusters"%n_clusters)
  algorithm.centroids = algorithm.centroids.cpu().detach().numpy()
  return algorithm, labels 

def get_K_means_v2(descriptors_list, sample_interval, elbow, 
                n_cluster_range = list(range(1, 15)), 
                kmeans_type = "torch"):
  """ Returns a fitted KMeans algorithm.
    :param descriptors_list: list of descriptors
    :param sample_interval: sample every ith descriptor before applying clustering.
    :param elbow: elbow coefficient to set number of clusters.
  """  
  # Sample normalized descriptors
  normalized_all_descriptors = get_normal_descriptors(descriptors_list)
  sampled_descriptors_list = [x[:, :, ::sample_interval, :] for x in descriptors_list]
  normalized_all_sampled_descriptors = get_normal_descriptors(sampled_descriptors_list)
  
  
  # Fit until elbow point
  sum_of_squared_dists = []
  for n_clusters in n_cluster_range:
    print("Attempting K=%d"%n_clusters)
    if kmeans_type == "torch":
      print ("Using KMeans - PyTorch, Cosine Similarity, No Elbow")
      print ("Output centroids are normalized")
      algorithm = torch_KMeans(n_clusters=n_clusters, mode='cosine', verbose=1, max_iter=300)
      normalized_all_descriptors = torch.Tensor(normalized_all_descriptors).cuda()
      normalized_all_sampled_descriptors = torch.Tensor(normalized_all_sampled_descriptors).cuda()
      labels = algorithm.fit_predict(normalized_all_sampled_descriptors)
      labels = algorithm.predict(normalized_all_descriptors)
      algorithm.centroids = algorithm.centroids.detach().cpu()
      centroids = np.asarray( algorithm.centroids )
      faiss.normalize_L2(centroids)
      break
    elif kmeans_type=="faiss":
      print ("Using KMeans - Faiss, Eucliedean Distance, with Elbow")
      print ("Output centroids are NOT normalized")
      algorithm = faiss.Kmeans(d=normalized_all_sampled_descriptors.shape[1], k=n_clusters, niter=300, nredo=10)
      algorithm.train(normalized_all_sampled_descriptors.astype(np.float32))
      squared_distances, labels = algorithm.index.search(normalized_all_descriptors.astype(np.float32), 1)
    else:
      NotImplementedError("Unrecognized type")

    objective = squared_distances.sum()
    sum_of_squared_dists.append(objective / normalized_all_descriptors.shape[0])
    centroids = np.asarray( algorithm.centroids )
    if (len(sum_of_squared_dists) > 1 and sum_of_squared_dists[-1] > elbow * sum_of_squared_dists[-2]):
      break
  return centroids, labels 
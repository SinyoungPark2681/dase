from src import Estimation_dsbm

import warnings
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import time
from sklearn.metrics.cluster import normalized_mutual_info_score

import numpy as np
import networkx as nx

from scipy.sparse.linalg import svds


def get_ratio(n, sparsity, relsize, p_matrix, k, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Calculate the total possible edges (T)
    T = n * (n-1)
    
    # Calculate the total expected edges (f)
    f_intra = sum(p_matrix[i,i] * relsize[i] * (relsize[i]-1) for i in range(k))
    f_inter1 = sum(p_matrix[i,j] * relsize[i] * relsize[j] for i in range(k) for j in range(i + 1, k))
    f_inter2 = sum(p_matrix[j,i] * relsize[i] * relsize[j] for i in range(k) for j in range(i + 1, k))
    f = f_intra + f_inter1 + f_inter2
    
    # Calculate scaling factors
    c = sparsity * T / f
        
    return c * p_matrix


def make_labels(n, cluster_sizes):
    """
    Generate cluster labels for a network with k clusters (0-indexed).

    Parameters:
    - n: Total number of nodes.
    - cluster_sizes: List of sizes for each cluster. The sum of this list should equal n.

    Returns:
    - clusters: List of lists where each sublist contains the node indices for that cluster.
    - labels: List of labels where each node index is assigned a label according to its cluster.
    """
    
    if sum(cluster_sizes) != n:
        raise ValueError("Sum of cluster sizes must equal the total number of nodes")

    total_nodes = list(range(n))
    clusters = []
    labels = []

    start_index = 0
    for cluster_id, size in enumerate(cluster_sizes, start=0):  # 👈 start=0 으로 변경
        end_index = start_index + size
        cluster_nodes = total_nodes[start_index:end_index]
        clusters.append(cluster_nodes)
        labels.extend([cluster_id] * size)  # 이제 0부터 시작
        start_index = end_index

    return clusters, labels


def calculate_relw_relb(n, relsize, p_matrix, k, sparsity, iteration):
    
    relw11 = []
    relw22 = []
    relb12 = []
    relb21 = []
    
    for s in sparsity:
        b11_list = []
        b22_list = []
        b12_list = []
        b21_list = []
        
        for i in range(iteration):
            
            b_matrix = get_ratio(n, s, relsize, p_matrix, k, seed=i)
            b11 = b_matrix[0, 0]
            b22 = b_matrix[1, 1]
            b12 = b_matrix[0, 1]
            b21 = b_matrix[1, 0]
            
            b11_list.append(b11)
            b22_list.append(b22)
            b12_list.append(b12)
            b21_list.append(b21)
            
        # Calculate mean values of b1 and b2 for this sparsity level
        relw11.append(np.mean(b11_list))
        relw22.append(np.mean(b22_list))
        relb12.append(np.mean(b12_list))
        relb21.append(np.mean(b21_list))
    
    return relw11, relw22, relb12, relb21


def process_sparsity_iterations_SBM(n, labels, iteration, sizes, P_matrix, sparsity, d, k, direction, method):
    """
    Process SBM network iterations over different sparsity levels, computing various metrics.

    Parameters:
    - X: Covariate matrix
    - labels: True labels for evaluation
    - sparsity_list: List of sparsity values to iterate over
    - iteration: Number of iterations per sparsity value
    - relsize: Relative size of each cluster
    - relw12: Relative weights within blocks
    - relb: Relative weights between blocks
    - beta: Parameters for ACSBM network generation
    - k: Number of clusters

    Returns:
    - Tuple containing lists of results for density, NMI scores, standard deviations, alphas, and computation times
    """

    relw_setting = np.diag(P_matrix)
    relw_setting = relw_setting.tolist()
    relb_setting = P_matrix.tolist()
    
    relw11, relw22, relb12, relb21 = calculate_relw_relb(n, sizes, P_matrix, k, sparsity, iteration)        
    
            
    # Initialize result lists
    dense_results = []
    
    NMI_sc = []
    NMI_ase1 = []
    NMI_ase2 = []
    NMI_ase3 = []

    std_sc = []
    std_ase1 = []
    std_ase2 = []
    std_ase3 = []
    
    time_sc = []
    time_ase1 = []
    time_ase2 = []
    time_ase3 = []

    # Progress bar setup
    progress_bar = tqdm(total=len(sparsity) * iteration, desc="Processing")

    # Iterate over each sparsity value
    for s in range(len(sparsity)):
        density_list = []
        
        score_sc = []
        score_ase1 = []
        score_ase2 = []
        score_ase3 = []

        time_sc_list = []
        time_ase1_list = []
        time_ase2_list = []
        time_ase3_list = []

        for i in range(iteration):
            
            # Generate SBM network
            
            P = np.array([[relw11[s], relb12[s]], [relb21[s], relw22[s]]])
            
            G = nx.stochastic_block_model(sizes, P, directed = direction, seed = i)
            Adj = nx.to_numpy_array(G)
            density_list.append(nx.density(G))
            
            # Spectral Clustering
            pipeline = Estimation_dsbm.ClusteringPipeline(Adj)
            pipeline.spectral_clustering(case='sym', k=k)
            results = pipeline.results
            score_sc.append(normalized_mutual_info_score(labels, results['spectral']['labels_pred']))
            time_sc_list.append(results['spectral']['time'])
            
            # ASE
            start = time.time()
            ase1 = pipeline.gen_ASE(case='ASE1', k=k, d=d, rs=i, direction = direction, method=method)
            end = time.time()
            time_ase1_list.append(end-start)
            score_ase1.append(normalized_mutual_info_score(ase1, labels))
            
            
            start = time.time()
            ase2 = pipeline.gen_ASE(k=k, d=d, rs=i, direction = direction, method=method)
            end = time.time()
            time_ase2_list.append(end-start)
            score_ase2.append(normalized_mutual_info_score(ase2, labels))
            
            start = time.time()
            ase3 = pipeline.gen_ASE(case='Atilde', k=k, d=d, rs=i, direction = direction, method=method)
            end = time.time()
            time_ase3_list.append(end-start)
            score_ase3.append(normalized_mutual_info_score(ase3, labels))



            progress_bar.update(1)

        # Aggregate results
        
        dense_results.append(np.mean(density_list))
        
        NMI_sc.append(np.mean(score_sc))
        NMI_ase1.append(np.mean(score_ase1))
        NMI_ase2.append(np.mean(score_ase2))
        NMI_ase3.append(np.mean(score_ase3))

        std_sc.append(np.std(score_sc))
        std_ase1.append(np.std(score_ase1))
        std_ase2.append(np.std(score_ase2))
        std_ase3.append(np.std(score_ase3))

        time_sc.append(np.mean(time_sc_list))
        time_ase1.append(np.mean(time_ase1_list))
        time_ase2.append(np.mean(time_ase2_list))
        time_ase3.append(np.mean(time_ase3_list))

    progress_bar.close()
    print("Processing completed.")

    return (
        dense_results,
        
        NMI_sc, 
        NMI_ase1, NMI_ase2, NMI_ase3,
        std_sc, 
        std_ase1, std_ase2, std_ase3,
        time_sc, 
        time_ase1, time_ase2, time_ase3
    )


def process_group_ratio_iterations_SBM(N, sparsity, iteration, size_ratio, P_matrix, d, k, direction, method):
    """
    Process SBM network iterations over different sparsity levels, computing various metrics.

    Parameters:
    - X: Covariate matrix
    - labels: True labels for evaluation
    - sparsity_list: List of sparsity values to iterate over
    - iteration: Number of iterations per sparsity value
    - relsize: Relative size of each cluster
    - relw12: Relative weights within blocks
    - relb: Relative weights between blocks
    - beta: Parameters for ACSBM network generation
    - k: Number of clusters

    Returns:
    - Tuple containing lists of results for density, NMI scores, standard deviations, alphas, and computation times
    """

    # Initialize result lists
    dense_results = []
    ratio_results = []
    
    NMI_sc = []
    NMI_ase1 = []
    NMI_ase2 = []
    NMI_ase3 = []

    std_sc = []
    std_ase1 = []
    std_ase2 = []
    std_ase3 = []
    
    time_sc = []
    time_ase1 = []
    time_ase2 = []
    time_ase3 = []
    

    # Progress bar setup
    progress_bar = tqdm(total=len(size_ratio) * iteration, desc="Processing")
                
                

    # Iterate over each sparsity value
    for sizes in size_ratio:
        
        # Ground truth label
        cluster_sizes = [int(N * s) for s in sizes]
        clusters, labels = make_labels(int(N), cluster_sizes)
        
        ratio_results.append(cluster_sizes[0]/N)
        
        relw_setting = np.diag(P_matrix)
        relw_setting = relw_setting.tolist()
        relb_setting = P_matrix.tolist()

        relw11, relw22, relb12, relb21 = calculate_relw_relb(N, cluster_sizes, P_matrix, k, sparsity, iteration)
        
        P = np.array([[relw11, relb12], [relb21, relw22]])

        density_list = []
        
        score_sc = []
        score_ase1 = []
        score_ase2 = []
        score_ase3 = []

        time_sc_list = []
        time_ase1_list = []
        time_ase2_list = []
        time_ase3_list = []

        for i in range(iteration):
            
            G = nx.stochastic_block_model(cluster_sizes, P, directed = direction, seed = i)
            Adj = nx.to_numpy_array(G)
            density_list.append(nx.density(G))
            
            
            # Spectral Clustering
            pipeline = Estimation_dsbm.ClusteringPipeline(Adj)
            pipeline.spectral_clustering(case='sym', k=k)
            results = pipeline.results
            score_sc.append(normalized_mutual_info_score(labels, results['spectral']['labels_pred']))
            time_sc_list.append(results['spectral']['time'])

            
            # ASE
            start = time.time()
            ase1 = pipeline.gen_ASE(case='ASE1', k=k, d=d, rs=i, direction = direction, method=method)
            end = time.time()
            time_ase1_list.append(end-start)
            score_ase1.append(normalized_mutual_info_score(ase1, labels))
            
            start = time.time()
            ase2 = pipeline.gen_ASE(case='DASE', k=k, d=d, rs=i, direction = direction, method=method)
            end = time.time()
            time_ase2_list.append(end-start)
            score_ase2.append(normalized_mutual_info_score(ase2, labels))
            
            start = time.time()
            ase3 = pipeline.gen_ASE(case='Atilde', k=k, d=d, rs=i, direction = direction, method=method)
            end = time.time()
            time_ase3_list.append(end-start)
            score_ase3.append(normalized_mutual_info_score(ase3, labels))
            

            progress_bar.update(1)

        # Aggregate results
        dense_results.append(np.mean(density_list))
        
        NMI_sc.append(np.mean(score_sc))
        NMI_ase1.append(np.mean(score_ase1))
        NMI_ase2.append(np.mean(score_ase2))
        NMI_ase3.append(np.mean(score_ase3))

        std_sc.append(np.std(score_sc))
        std_ase1.append(np.std(score_ase1))
        std_ase2.append(np.std(score_ase2))
        std_ase3.append(np.std(score_ase3))

        time_sc.append(np.mean(time_sc_list))
        time_ase1.append(np.mean(time_ase1_list))
        time_ase2.append(np.mean(time_ase2_list))
        time_ase3.append(np.mean(time_ase3_list))

    progress_bar.close()
    print("Processing completed.")

    return (
        dense_results, ratio_results,
        
        NMI_sc, NMI_ase1, NMI_ase2, NMI_ase3,
        std_sc, std_ase1, std_ase2, std_ase3,
        time_sc, time_ase1, time_ase2, time_ase3
    )

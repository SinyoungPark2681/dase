from Functions import GenDSBM_core, Estimation_dsbm, NMI
from Functions.functions import *

import warnings
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import time
from sklearn.metrics.cluster import normalized_mutual_info_score

import numpy as np
import networkx as nx

from scipy.sparse.linalg import svds


def process_nodes_iterations_SBM(node_list, sparsity, iteration, size_ratio, P_matrix, d, k, direction, method):
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
    progress_bar = tqdm(total=len(node_list) * iteration, desc="Processing")
                
                

    # Iterate over each sparsity value
    for n in node_list:
        
        # Ground truth label
        cluster_sizes = [int(n * r) for r in size_ratio]
        clusters, labels = make_labels(int(n), cluster_sizes)
        
        relw_setting = np.diag(P_matrix)
        relw_setting = relw_setting.tolist()
        relb_setting = P_matrix.tolist()

        relw11, relw22, relb12, relb21 = calculate_relw_relb(n, cluster_sizes, P_matrix, k, sparsity, iteration)
        
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
        
        NMI_sc, NMI_ase1, NMI_ase2, NMI_ase3,
        std_sc, std_ase1, std_ase2, std_ase3,
        time_sc, time_ase1, time_ase2, time_ase3
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
    
    
    NMI_dsbm = []
    NMI_ata = []
    NMI_aat = []
    NMI_mix = []
    NMI_degree = []

    std_dsbm = []
    std_ata = []
    std_aat = []
    std_mix = []
    std_degree = []
    
    time_dsbm = []
    time_ata = []
    time_aat = []
    time_mix = []
    time_degree = []
    
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

        cluster_sizes2 = [int(N/2), int(N/2)]
        
        relw_setting = np.diag(P_matrix)
        relw_setting = relw_setting.tolist()
        relb_setting = P_matrix.tolist()

        relw11, relw22, relb12, relb21 = calculate_relw_relb(N, cluster_sizes2, P_matrix, k, sparsity, iteration)
        
        P = np.array([[relw11, relb12], [relb21, relw22]])

        density_list = []
        
        score_dsbm = []
        score_ata = []
        score_aat = []
        score_mix = []
        score_degree = []

        time_dsbm_list = []
        time_ata_list = []
        time_aat_list = []
        time_mix_list = []
        time_degree_list = []
        
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
            ase2 = pipeline.gen_ASE(k=k, d=d, rs=i, direction = direction, method=method)
            end = time.time()
            time_ase2_list.append(end-start)
            score_ase2.append(normalized_mutual_info_score(ase2, labels))
            
            start = time.time()
            ase3 = pipeline.gen_ASE(case='Atilde', k=k, d=d, rs=i, direction = direction, method=method)
            end = time.time()
            time_ase3_list.append(end-start)
            score_ase3.append(normalized_mutual_info_score(ase3, labels))
            
            
            # DSBM
            start = time.time()
            dsbm = pipeline.gen_DSBM1(k=k, d=d, rs=i, direction = direction)
            end = time.time()
            time_dsbm_list.append(end-start)
            score_dsbm.append(normalized_mutual_info_score(dsbm, labels))
            
            
            start = time.time()
            ata = pipeline.gen_DSBM2(case='A1A', k=k, d=d, rs=i, direction = direction, method=method)
            end = time.time()
            time_ata_list.append(end-start)
            score_ata.append(normalized_mutual_info_score(ata, labels))
            
            start = time.time()
            aat = pipeline.gen_DSBM2(case='AA1', k=k, d=d, rs=i, direction = direction, method=method)
            end = time.time()
            time_aat_list.append(end-start)
            score_aat.append(normalized_mutual_info_score(aat, labels))
            
            start = time.time()
            mix = pipeline.gen_DSBM2(case='mix', k=k, d=d, rs=i, direction = direction, method=method)
            end = time.time()
            time_mix_list.append(end-start)
            score_mix.append(normalized_mutual_info_score(mix, labels))
            
            start = time.time()
            degree = pipeline.gen_D_Discounted(k=k, d=d, rs=i, direction = direction, method=method)
            end = time.time()
            time_degree_list.append(end-start)
            score_degree.append(normalized_mutual_info_score(degree, labels))
            

            progress_bar.update(1)

        # Aggregate results
        NMI_dsbm.append(np.mean(score_dsbm))
        NMI_ata.append(np.mean(score_ata))
        NMI_aat.append(np.mean(score_aat))
        NMI_mix.append(np.mean(score_mix))
        NMI_degree.append(np.mean(score_degree))

        std_dsbm.append(np.std(score_dsbm))
        std_ata.append(np.std(score_ata))
        std_aat.append(np.std(score_aat))
        std_mix.append(np.std(score_mix))
        std_degree.append(np.std(score_degree))

        time_dsbm.append(np.mean(time_dsbm_list))
        time_ata.append(np.mean(time_ata_list))
        time_aat.append(np.mean(time_aat_list))
        time_mix.append(np.mean(time_mix_list))
        time_degree.append(np.mean(time_degree_list))
        
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
        
        NMI_sc, 
        NMI_ase1, NMI_ase2, NMI_ase3,
        std_sc, 
        std_ase1, std_ase2, std_ase3,
        time_sc, 
        time_ase1, time_ase2, time_ase3,
        
        NMI_dsbm, 
        NMI_ata, NMI_aat, NMI_mix, NMI_degree,
        std_dsbm, 
        std_ata, std_aat, std_mix, std_degree,
        time_dsbm, 
        time_ata, time_aat, time_mix, time_degree
    )


def process_nodes_iterations_DSBM(node_list, sparsity, iteration, size_ratio, P_matrix, d, k, direction, method):
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
    
    NMI_sc2 = []
    NMI_ata = []
    NMI_aat = []
    NMI_mix = []
    NMI_degree = []

    std_sc2 = []
    std_ata = []
    std_aat = []
    std_mix = []
    std_degree = []
    
    time_sc2 = []
    time_ata = []
    time_aat = []
    time_mix = []
    time_degree = []
    
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
    progress_bar = tqdm(total=len(node_list) * iteration, desc="Processing")
                
                

    # Iterate over each sparsity value
    for n in node_list:
        
        # Ground truth label
        cluster_sizes = [int(n * r) for r in size_ratio]
        clusters, labels = make_labels(int(n), cluster_sizes)
        
        relw_setting = np.diag(P_matrix)
        relw_setting = relw_setting.tolist()
        relb_setting = P_matrix.tolist()

        relw11, relw22, relb12, relb21 = calculate_relw_relb(n, cluster_sizes, P_matrix, k, sparsity, iteration)
        
        P = np.array([[relw11, relb12], [relb21, relw22]])

        density_list = []
        
        score_sc2 = []
        score_ata = []
        score_aat = []
        score_mix = []
        score_degree = []

        time_sc2_list = []
        time_ata_list = []
        time_aat_list = []
        time_mix_list = []
        time_degree_list = []
        
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
            time_sc2_list.append(results['spectral']['time'])

            pipeline = Estimation_dsbm.ClusteringPipeline(Adj)
            pipeline.spectral_clustering(case='square', k=k)
            results = pipeline.results
            score_sc2.append(normalized_mutual_info_score(labels, results['spectral']['labels_pred']))
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
            
            
            # DSBM        
            start = time.time()
            ata = pipeline.gen_DSBM2(case='A1A', k=k, d=d, rs=i, direction = direction, method=method)
            end = time.time()
            time_ata_list.append(end-start)
            score_ata.append(normalized_mutual_info_score(ata, labels))
            
            start = time.time()
            aat = pipeline.gen_DSBM2(case='AA1', k=k, d=d, rs=i, direction = direction, method=method)
            end = time.time()
            time_aat_list.append(end-start)
            score_aat.append(normalized_mutual_info_score(aat, labels))
            
            start = time.time()
            mix = pipeline.gen_DSBM2(case='mix', k=k, d=d, rs=i, direction = direction, method=method)
            end = time.time()
            time_mix_list.append(end-start)
            score_mix.append(normalized_mutual_info_score(mix, labels))
            
            start = time.time()
            degree = pipeline.gen_D_Discounted(k=k, d=d, rs=i, direction = direction, method=method)
            end = time.time()
            time_degree_list.append(end-start)
            score_degree.append(normalized_mutual_info_score(degree, labels))
            

            progress_bar.update(1)

        # Aggregate results
        dense_results.append(np.mean(density_list))
        
        NMI_sc2.append(np.mean(score_sc2))
        NMI_ata.append(np.mean(score_ata))
        NMI_aat.append(np.mean(score_aat))
        NMI_mix.append(np.mean(score_mix))
        NMI_degree.append(np.mean(score_degree))

        std_sc2.append(np.std(score_sc2))
        std_ata.append(np.std(score_ata))
        std_aat.append(np.std(score_aat))
        std_mix.append(np.std(score_mix))
        std_degree.append(np.std(score_degree))

        time_sc2.append(np.mean(time_sc2_list))
        time_ata.append(np.mean(time_ata_list))
        time_aat.append(np.mean(time_aat_list))
        time_mix.append(np.mean(time_mix_list))
        time_degree.append(np.mean(time_degree_list))
        
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
        time_ase1, time_ase2, time_ase3,
        
        NMI_sc2, 
        NMI_ata, NMI_aat, NMI_mix, NMI_degree,
        std_sc2, 
        std_ata, std_aat, std_mix, std_degree,
        time_sc2, 
        time_ata, time_aat, time_mix, time_degree
    )

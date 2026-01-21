import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import time

from sklearn.mixture import GaussianMixture
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.sparse.linalg import svds


class ClusteringPipeline:
    def __init__(self, A, X=None, labels=None):
        """
        Initialize the clustering pipeline.
        
        Parameters:
        A : numpy array
            Adjacency matrix of the graph.
        X : numpy array, optional
            Covariate matrix.
        labels : numpy array, optional
            True labels for the nodes (for evaluating clustering performance).
        """
        self.A = A
        self.X = X
        self.labels = labels
        
        # Initialize results storage
        self.spectral_results = {}
        self.ACSBM_results = {}
        self.results = {}  # Initialize the results dictionary here
        
    def diag(self, A):
        """Return the degree matrix for the adjacency matrix A."""
        return np.diag(np.sum(A, axis=1))

    def spectral_clustering(self, case='sym', k=2, rs=0):
        """Perform spectral clustering based on the specified Laplacian."""
        start = time.time()
        D = self.diag(self.A)
        L = D - self.A
        
        if case == 'unnorm':
            eigvals, eigvecs = np.linalg.eig(L)
            U = eigvecs[:, np.argsort(eigvals)[:k]]
            
        elif case == 'sym':
            D_norm = np.linalg.inv(np.sqrt(D))
            L_sym = D_norm @ self.A @ D_norm
            eigvals, eigvecs = eigsh(L_sym.astype(np.float64), k, which='LA')
            U = D_norm @ eigvecs
            
        elif case == 'rw':
            D_inv = np.linalg.inv(D)
            L_rw = np.eye(len(self.A)) - D_inv @ self.A
            eigvals, eigvecs = np.linalg.eig(L_rw)
            U = eigvecs[:, np.argsort(eigvals)[:k]]
        
        labels_pred = KMeans(n_clusters=k, random_state=rs).fit_predict(U)
        elapsed_time = time.time() - start
        
        self.results['spectral'] = {
            'labels_pred': labels_pred,
            'time': elapsed_time,
            'NMI': normalized_mutual_info_score(self.labels, labels_pred) if self.labels is not None else None
        }


    def gen_ASE(self, case='ASE1', k=2, d=2, rs=0, direction=True, method='GMM'):

        # Number of nodes in the network
        n = self.A.shape[0]
        A = self.A
        A = A.astype(float)

        if case == 'ASE':
            Adj = A
        elif case == 'DASE':
            Adj = (A @ A)
        elif case == 'DASE_norm':
            Adj = (A @ A)/n

        
        # Compute the singular value decomposition (smallest to largest singular values)
        U, Sigma, Vt = svds(Adj, k=d)

        # Reverse order to get descending singular values
        U2 = U[:, ::-1]
        Sigma2 = Sigma[::-1]
        Vt2 = Vt[::-1, :]

        # Take the square root of singular values
        Sigma_sqrt = np.diag(np.sqrt(Sigma2))

        # Scale U and Vt by sqrt of singular values
        U1 = U2 @ Sigma_sqrt
        V1 = Vt2.T @ Sigma_sqrt

        # Concatenate for ASE embedding
        if direction == True:
            Z = np.hstack((U1, V1))
        else:
            Z = U1
            
        if method == 'GMM':
            
            gm = GaussianMixture(n_components=k, random_state=rs).fit(Z)
            pred_labels = gm.predict(Z)

            return pred_labels
        
        elif method == 'K-Means':
            
            pred_labels = KMeans(n_clusters=k, random_state=rs).fit_predict(Z)

            return pred_labels

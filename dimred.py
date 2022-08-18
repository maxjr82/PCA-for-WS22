# =======================================================
# Author: Max Pinheiro Jr <max.pinheiro-jr@univ-amu.fr>
# Date: July 28 2022
# =======================================================
import os, glob
import numpy as np
import pandas as pd

from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def r2_descriptor(xyz_matrix: np.ndarray) -> np.ndarray:
    """Build the pairwise atom-atom distance descriptor.

    Args:
       xyz_matrix: a numpy 3D tensor of shape (n_geometries, n_atoms, 3) containing the stacked 
                   Cartesian coordinates of the molecular structures.

    Return:
       r2_matrix: a numpy 2D array of shape (n_geometries, n_atoms * (n_atoms - 1)/2) with the
                  unique pairwise distances vector calculated for each molecular geometry.
    """
    n_samples, n_atoms, _ = xyz_matrix.shape
    dist_vec_size = int(n_atoms * (n_atoms - 1)/2)
    r2_matrix = np.empty((n_samples, dist_vec_size))
    for idx in range(n_samples):
        distance_matrix = np.zeros((n_atoms, n_atoms))
        geom = xyz_matrix[idx]
        for i, j in combinations(range(len(geom)), 2):
            R = np.linalg.norm(geom[i] - geom[j])
            distance_matrix[j, i] = R
        r2_vector = distance_matrix[np.tril_indices(len(distance_matrix), -1)]
        r2_matrix[idx, :] = r2_vector.reshape(1,-1)
    return r2_matrix

npz_files = glob.iglob("*.npz")

df_pca = []

for file in npz_files:
    mol = file.replace('ws22_','').replace('.npz','')
    print("======================================")
    print("Loading data for molecule {}".format(mol))
    print("======================================\n")
    data = dict(np.load(file))
    print("Calculating the inverse R2 descriptor...\n")
    r2 = r2_descriptor(data['R'])
    print("Number of samples = {}".format(r2.shape[0]))
    print("Number of features = {}".format(r2.shape[1]))
    print(" ")
    conformations = np.unique(data['CONF'])
    print("Conformations ---> {}\n".format(conformations))
    X = MinMaxScaler().fit_transform(r2)
    print("Performing dimensionality reduction with PCA...\n")
    pca_n2 = PCA(n_components=2)
    pca_n2.fit(X)
    X_transformed = pca_n2.transform(X)
    print("*************************")
    print("Explained variance ratio")
    print("*************************")
    print(" ")
    print(pca_n2.explained_variance_ratio_)
    print(" ")
    col_names = [mol + '_pc1', mol + '_pc2']
    df = pd.DataFrame(X_transformed, columns=col_names)
    df[mol + '_conf'] = data['CONF'].flatten()
    df_pca.append(df)

df_pca = pd.concat(df_pca, axis=1)
print("Saving all data to a compressed csv file\n")
compression_opts = dict(method='zip', archive_name='ws22_pca_n2.csv')
df_pca.to_csv('pca.zip', index=False, header=True, compression=compression_opts)

print("FINISHED!")

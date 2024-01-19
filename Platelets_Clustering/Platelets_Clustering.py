#!/usr/bin/env python
# coding: utf-8

# In[39]:


import argparse
import sys
import readfcs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import sem 
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from glob import glob

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', type=str, help='Input directory for folder with datasets')
    parser.add_argument('outdir', type=str, help='Output directory folder')
    parser.add_argument('--algorithm', type=str, help='Choose algoritm for clusterization', 
                        choices=['DBSCAN', 'k-means', 'Agglom'], default='DBSCAN')
    parser.add_argument('--figures', type=str, help='Should graphs with clustering be saved', 
                        choices=['yes', 'no'], default='yes')
    parser.add_argument('--concentration', nargs='+', type=float, 
                        help='List of concentration for each dataset', default=None)
    return parser


def pd_read_fcs(path: str) -> pd.DataFrame:
    return pd.DataFrame(readfcs.ReadFCS(path).data)

def df_add_log_columns(df: pd.DataFrame, default_columns = ['FSC-A', 'SSC-A', 'PerCP-Cy5-5-A', 'APC-A']) -> pd.DataFrame:
    
    df_work = df[default_columns]
    df_work = df_work.loc[(df_work['PerCP-Cy5-5-A']>0) & (df_work['APC-A']>0) & 
                          (df_work['FSC-A']>0) & (df_work['SSC-A']>0)]
    #logarithmic scale
    for column in default_columns:
        df_work['log'+column] = np.log10(df_work[column])
    return df_work

def noise_cutting(df: pd.DataFrame, columns, tangent = 1, min_samples=10) -> pd.DataFrame:

    def optimal_epsilon(df: pd.DataFrame, columns, tangent = 1) -> float:
        neigh = NearestNeighbors(n_neighbors = 2)
        nbrs = neigh.fit(df[columns])
        distances, indices = nbrs.kneighbors(df[columns])
        distances = np.sort(distances, axis = 0)[:, 1]
        y_norm = np.array(distances)/max(distances)*len(distances)
        dy_norm = np.gradient(y_norm, range(len(y_norm)))
        index_of_curvesity_max = abs(dy_norm - tangent).argmin()
        return distances[index_of_curvesity_max]

    dbscan = DBSCAN(eps=optimal_epsilon(df, columns = columns, tangent = tangent),
              min_samples=10)

    dbscan.fit(df[columns])
    labels = dbscan.labels_
    label_values, label_counts = np.unique(labels, return_counts= True)
    df = df[labels == label_values[np.argmax(label_counts)]]
    
    return df

def clustering(data_files: list, algorithm: str, output_path: str):
    STAT_file = pd.DataFrame(columns = ['Subpopulation','Mean', 'std', 'sem'])
    LABELS = {}
    
    for num_data, data_path in enumerate(data_files):
        print(data_path)
        df = pd_read_fcs(data_path)
        df_work = df_add_log_columns(df)
        df_work_2 = noise_cutting(df_work, columns = ['logFSC-A', 'logSSC-A'], min_samples=10)
        
        xy2 = df_work_2[['logPerCP-Cy5-5-A', 'logAPC-A']]
        x2 = df_work_2['logAPC-A']
        y2 = df_work_2['logPerCP-Cy5-5-A']

        times = []
        labels = {}
        
        if algorithm == 'DBSCAN':
            print('DBSCAN\n')
            dbscan = DBSCAN(eps=0.05, min_samples=30)
            dbscan.fit(xy2)
            labels = dbscan.labels_
        
        elif algorithm == 'k-means':
            print('K-means')
            k_means = KMeans(n_clusters=2, init= 'k-means++', n_init=5)
            k_means.fit(xy2)
            labels = k_means.labels_
            
        elif algorithm == 'Agglom':
            print('Agglom')
            ac = AgglomerativeClustering(n_clusters=2,
                                     metric='euclidean',
                                     linkage='ward')

            ac.fit(xy2)
            labels = ac.labels_   
            
        LABELS[data_path] = labels
        
        #compute the statistics for each subpopulations
        
        df_g1 = df_work_2[labels == 0]
        df_g2 = df_work_2[labels == 1]
        data_path = data_path.split('\\')[-1].split('.')[0]

        bigger_group = 1 if df_g1['PerCP-Cy5-5-A'].mean() > df_g2['PerCP-Cy5-5-A'].mean() else 2
        
        mean_1, mean_2 = round(float(df_g1['APC-A'].mean()), 3),  round(float(df_g2['APC-A'].mean()), 3)
        std_1, std_2 = round(float(df_g1['APC-A'].std()), 3),  round(float(df_g2['APC-A'].std()), 3)
        sem_1, sem_2 = round(sem(df_g1['APC-A'], axis=None), 3), round(sem(df_g2['APC-A'], axis=None), 3)

        new_row_1 = {'Subpopulation': 'Group 1',
                     'Mean': mean_1 if bigger_group == 1 else mean_2, 
                     'std': std_1 if bigger_group == 1 else std_2,
                     'sem': sem_1 if bigger_group == 1 else sem_2}
        new_row_2 = {'Subpopulation': 'Group 2',
                     'Mean': mean_1 if bigger_group == 2 else mean_2, 
                     'std': std_1 if bigger_group == 2 else std_2,
                     'sem': sem_1 if bigger_group == 2 else sem_2}
        
        STAT_file = pd.concat([STAT_file, pd.DataFrame([new_row_1],
                                            index=[data_path],columns=['Subpopulation','Mean', 'std', 'sem'])])
        STAT_file = pd.concat([STAT_file, pd.DataFrame([new_row_2],
                                            index=[data_path],columns=['Subpopulation','Mean', 'std', 'sem'])])

    output_path = output_path+'Results.csv'
    STAT_file.to_csv(output_path, index = True)    
            
    return LABELS, STAT_file

def visualization_clustering(data_files: list, LABELS: dict, output_path: str, answer: str):
    
    size = len(data_files)
    i=0
    fig, ax = plt.subplots(math.ceil(size/2), 2, sharex=False, sharey=False, figsize=(12, 6*math.ceil(size/2)))
    
    for num_data, data_path in enumerate(data_files):
        df = pd_read_fcs(data_path)
        df_work = df_add_log_columns(df)
        df_work_2 = noise_cutting(df_work, columns = ['logFSC-A', 'logSSC-A'], min_samples=10)

        x2 = df_work_2['logAPC-A']
        y2 = df_work_2['logPerCP-Cy5-5-A']
        
        ax[num_data//2, i].scatter(x=x2, y=y2, c=LABELS[data_path], s=0.5, cmap="Paired")
        ax[num_data//2, i].set_title(data_path.split('\\')[-1].split('.')[0])
        ax[num_data//2, i].set(ylabel='logPerCP-Cy5-5-A')
        ax[num_data//2, i].set(xlabel='logAPC-A')
        
        i = (i+1)%2   
    
    if answer=='yes':
        output_path = output_path+'Results.png'
        plt.savefig(output_path, dpi=300, facecolor='1')

    plt.show()
    
def visualization_stat(data_files: list, STAT_file: pd.DataFrame, concentration: list):
    
    if len(data_files) != len(concentration):
        return print('Incorrect list of concentration')
    else:
        s1 = STAT_file.loc[STAT_file['Subpopulation'] == 'Group 1']
        s2 = STAT_file.loc[STAT_file['Subpopulation'] == 'Group 2']
        plt.errorbar(x=concentration, y=s1['Mean'], yerr=s1['std'], fmt='o--b', 
                     ecolor='black', elinewidth=1, markersize=5, label='Group 1')
        plt.errorbar(x=concentration, y=s2['Mean'], yerr=s2['std'], fmt='o--r', 
                     ecolor='black', elinewidth=1, markersize=5, label='Group 1')
        plt.xlabel('Concentration')
        plt.ylabel('Mean flurescence, a.u.')
        plt.legend()
        plt.show()
        
        
        
if __name__ == "__main__":
    
    parser = createParser()
    namespace = parser.parse_args()
        
    folder_path = namespace.indir
    output_path = namespace.outdir
    algorithm = namespace.algorithm 
    
    data_files = sorted(glob(folder_path + '*.fcs'))
    size = len(data_files)
    print(f'{size} datasets is read\n')

    LABELS, STAT = clustering(data_files, algorithm, output_path)
    visualization_clustering(data_files, LABELS, output_path, namespace.figures)
    if len(namespace.concentration) > 0:
        visualization_stat(data_files, STAT, namespace.concentration)


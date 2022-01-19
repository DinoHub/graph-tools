import torch
import pickle
import json
import pandas as pd
import numpy as np
from cuml.manifold.umap import UMAP as cumlUMAP
from cuml.cluster import HDBSCAN


class Clustering:

    def __init__(self,er_emb_path,doc_emb_path):
        self.er_emb_path = er_emb_path
        self.doc_emb_path = doc_emb_path

    def load_data(self):
        er_emb = torch.load(self.er_emb_path) 
        doc_emb = pickle.load(open(self.doc_emb_path,'rb')) 
        return er_emb, doc_emb

    def numpy_convert(self,doc_emb):
        id_list = []
        emb_list = []
        for doc_dict in doc_emb:
            id_list = id_list + list(doc_dict.keys())
            for doc in doc_dict.keys():
                emb_list.append(doc_dict[doc].detach().numpy())
            doc_arrays = np.array(emb_list)
        print(len(id_list))
        return id_list, doc_arrays

    def umap_hdfs(self,doc_emb):
        id_list,doc_arrays = self.numpy_convert(doc_emb)
        g_embedding = cumlUMAP(n_neighbors=30,min_dist=0.0,n_components=2,init="spectral").fit_transform(doc_arrays)
        model = HDBSCAN(min_samples=10,min_cluster_size=100)
        labels = model.fit_predict(g_embedding)
        cluster_df = pd.DataFrame()
        cluster_df['id'] = id_list
        cluster_df['emb'] = doc_arrays.tolist()
        cluster_df['labels'] = labels
        return cluster_df
    
    def output_cluster_dict(self,doc_emb):
        cluster_df = self.umap_hdfs(doc_emb)
        cluster_labels = cluster_df['labels'].unique()
        full_dict = {}
        for label in cluster_labels:
            temp_ids = cluster_df[cluster_df['labels']==label]['id'].tolist()
            if label==-1:
                temp_centroids = []
            else:
                temp_centroids = np.mean(cluster_df[cluster_df['labels']==label]['emb'].tolist(),axis=0).tolist()
            temp_dict = {"id_list":temp_ids, "centroid":temp_centroids}
            full_dict[str(label)] = temp_dict
        return full_dict
    

if __name__ == '__main__':
    cluster_object = Clustering("data/transe.ckpt","data/temporal_list_by_idx.pkl")
    er_emb, doc_emb = cluster_object.load_data()
    output_dict = cluster_object.output_cluster_dict(doc_emb)
    with open('data/cluster_data.json', 'w') as fp:
        json.dump(output_dict, fp)

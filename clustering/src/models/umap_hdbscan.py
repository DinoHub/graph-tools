import pandas as pd
import numpy as np
from cuml.manifold.umap import UMAP as cumlUMAP
from cuml.cluster import HDBSCAN
from typing import Any, List, Dict, Tuple

class Clustering:
    def __init__(self, config: Any):
        self.config=config
        self.reduction_model = cumlUMAP(n_neighbors=self.config.n_neighbours, min_dist=self.config.min_distance, n_components=self.config.n_components, init="spectral")
        self.cluster_model = HDBSCAN(
            min_samples=self.config.min_samples, 
            min_cluster_size=self.config.min_cluster_size, 
            max_cluster_size=self.config.max_cluster_size, 
            cluster_selection_epsilon=self.config.cluster_selection_epsilon,
            metric=self.config.metric,
            alpha=self.config.alpha,
            p=self.config.p,
            cluster_selection_method=self.config.cluster_selection_method,
            allow_single_cluster=self.config.allow_single_cluster,
            gen_min_span_tree=self.config.gen_min_span_tree            
        )

    def convert_to_numpy_arrays(self, doc_emb: List)->Tuple[List, np.array]:
        id_list = []
        emb_list = []
        # only takes last element of doc_emb
        for doc_dict in doc_emb:
            id_list = id_list + list(doc_dict.keys())
            for doc in doc_dict.keys():
                emb_list.append(doc_dict[doc].detach().numpy())
            doc_arrays = np.array(emb_list)
        print(f"id length: {len(id_list)}")
        print("doc array size: ", doc_arrays.shape)
        return id_list, doc_arrays

    def umap_reduce(self, doc_arrays: np.array):
        return self.reduction_model.fit_transform(doc_arrays)        

    def cluster(self, embedding: np.array):
        return self.cluster_model.fit_predict(embedding)

    def reduce_and_cluster(self, doc_emb: List)->pd.DataFrame:
        print("umap reduction...")
        id_list, doc_arrays = self.convert_to_numpy_arrays(doc_emb)
        g_embedding = self.umap_reduce(doc_arrays)
        print("clustering...")
        labels = self.cluster(g_embedding)
        cluster_df = pd.DataFrame()
        cluster_df['id'] = id_list
        cluster_df['emb'] = doc_arrays.tolist()
        cluster_df['labels'] = labels
        return cluster_df

    def group_ids_by_labels(self, cluster_df: pd.DataFrame)->Dict:
        cluster_labels = cluster_df['labels'].unique()
        full_dict = {}
        for label in cluster_labels:
            cluster_ids = cluster_df[cluster_df['labels']==label]['id'].tolist()
            if label==-1:
                centroids = []
            else:
                centroids = np.mean(cluster_df[cluster_df['labels']==label]['emb'].tolist(),axis=0).tolist()
            cluster_dict = {"id_list":cluster_ids, "centroid":centroids}
            full_dict[str(label)] = cluster_dict
        return full_dict

    def generate_clusters(self, doc_emb):
        cluster_df = self.reduce_and_cluster(doc_emb)
        return self.group_ids_by_labels(cluster_df)
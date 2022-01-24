import json, os
from clearml import Task, StorageManager, Dataset

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--n_components', help='number of components')
# parser.add_argument('--n_neighbour', help='number of neighbour documents in cluster')
# parser.add_argument('--min_distance', help='minimum distance')
# parser.add_argument('--min_samples', help='minimum cluster samples')
# parser.add_argument('--min_cluster_size', help='minimum cluster size')
# args = parser.parse_args()

remote_path = "s3://experiment-logging"
Task.add_requirements("hydra-core")
task = Task.init(project_name='topic-cluster', task_name='graph-clustering-HDBScan',
                 output_uri=os.path.join(remote_path, "storage"))
task.set_base_docker("rapidsai/rapidsai-dev:21.10-cuda11.0-devel-ubuntu18.04-py3.8")
task.execute_remotely(queue_name="compute", exit_process=True)

import torch
import pickle
import pandas as pd
import numpy as np
from cuml.manifold.umap import UMAP as cumlUMAP
from cuml.cluster import HDBSCAN

class Clustering:

    def __init__(self,er_emb_path:str, doc_emb_path:str, config):
        self.config=config
        self.er_emb_path = er_emb_path
        self.doc_emb_path = doc_emb_path

    def load_data(self):
        er_emb = torch.load(StorageManager.get_local_copy(self.er_emb_path))
        #doc_emb = pickle.load(open(StorageManager.get_local_copy(self.doc_emb_path),'rb')) 
        doc_emb = torch.load(StorageManager.get_local_copy(self.doc_emb_path)) 
        return er_emb, doc_emb

    # def numpy_convert(self,doc_emb):
    #     id_list = []
    #     emb_list = []
    #     for doc_dict in doc_emb:
    #         id_list = id_list + list(doc_dict.keys())
    #         for doc in doc_dict.keys():
    #             emb_list.append(doc_dict[doc].detach().numpy())
    #         doc_arrays = np.array(emb_list)
    #     print(len(id_list))
    #     return id_list, doc_arrays

    def numpy_convert(self,doc_emb):
        id_list = [key for key, _ in doc_emb.items()]
        doc_arrays = np.array([value.detach().numpy() for _, value in doc_emb.items()])
        return id_list, doc_arrays

    def umap_hdfs(self,doc_emb):
        print("umap reduction...")
        id_list,doc_arrays = self.numpy_convert(doc_emb)
        print("doc array size: ", doc_arrays.shape)
        g_embedding = cumlUMAP(n_neighbors=self.config.n_neighbours, min_dist=self.config.min_distance, n_components=self.config.n_components, init="spectral").fit_transform(doc_arrays)
        model = HDBSCAN(
            min_samples=self.config.min_samples, 
            min_cluster_size=self.config.min_cluster_size, 
            max_cluster_size=self.config.max_cluster_size, 
            cluster_selection_epsilon=self.config.epsilon, 
            alpha=self.config.alpha, 
            metric=self.config.metric,
            p=self.config.p,
            cluster_selection_method=self.config.cluster_selection_method,
            cluster_selection_epsilon=self.config.cluster_selection_epsilon
            metric=self.config.euclidean
            alpha=self.config.alpha
            p=self.config.p
            cluster_selection_method=self.config.cluster_selection_method,
            allow_single_cluster=self.config.allow_single_cluster,
            gen_min_span_tree=self.config.gen_min_span_tree            
            )
        print("clustering...")
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


import hydra

@hydra.main(config_path="../configs", config_name="main")
def run_cluster(cfg) -> None:
    dataset_obj = Dataset.get(dataset_project="datasets/gdelt", dataset_name="gdelt_openke_format_w_extras", only_published=True)
    dataset_path = dataset_obj.get_local_copy()

    doc_emb_path = cfg.main.doc_emb_path
    entity_embedding_path = cfg.main.entity_embedding_path

    cluster_object = Clustering(er_emb_path=entity_embedding_path, doc_emb_path=doc_emb_path, config=cfg.main)
    er_emb, doc_emb = cluster_object.load_data()
    output_dict = cluster_object.output_cluster_dict(doc_emb)

    with open('./cluster_data.json', 'w') as fp:
        json.dump(output_dict, fp)
    task.upload_artifact("cluster_data.json", artifact_object='./cluster_data.json')


if __name__ == '__main__':
    run_cluster()

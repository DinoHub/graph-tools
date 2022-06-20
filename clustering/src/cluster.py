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
task = Task.init(project_name='topic-cluster', task_name='graph-clustering-HDBScan',
                 output_uri=os.path.join(remote_path, "storage"))
task.set_base_docker("rapidsai/rapidsai-dev:21.10-cuda11.0-devel-ubuntu18.04-py3.8")
task.execute_remotely(queue_name="compute", exit_process=True)

import torch
import pickle
from models.umap_hdbscan import Clustering
import hydra


def load_data(cfg):
    doc_emb_path = cfg.main.doc_emb_path
    entity_embedding_path = cfg.main.entity_embedding_path
    er_emb = torch.load(StorageManager.get_local_copy(entity_embedding_path))
    doc_emb = pickle.load(open(StorageManager.get_local_copy(doc_emb_path),'rb')) 
    return er_emb, doc_emb


@hydra.main(config_path="../configs", config_name="main")
def run_cluster(cfg) -> None:
    task.connect(cfg.main)
    # dataset_obj = Dataset.get(dataset_project="datasets/gdelt", dataset_name="gdelt_openke_format_w_extras", only_published=True)
    # dataset_path = dataset_obj.get_local_copy()
    _, doc_emb = load_data(cfg)

    cluster_controller = Clustering(config=cfg.main)
    output_dict = cluster_controller.generate_clusters(doc_emb)

    with open('./cluster_data.json', 'w') as fp:
        json.dump(output_dict, fp)
    # task.upload_artifact("cluster_data.json", artifact_object='./cluster_data.json')


if __name__ == '__main__':
    run_cluster()

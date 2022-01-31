#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 16:18:51 2022

@author: aaron
"""

from clearml import Task, StorageManager, Dataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split

ds = Dataset.get(dataset_project='datasets/gdelt', dataset_name='raw_gdelt_2020_w_document_cluster_ids')
ds_path = ds.get_mutable_local_copy("./data")

cluster2id = pd.read_csv(os.path.join(ds_path, "cluster2id.txt"), sep=",")
entity2id = pd.read_csv(os.path.join(ds_path, "ent2id.txt"), sep=",")
evt2id = pd.read_csv(os.path.join(ds_path, "evt2id.txt"), sep=",")
raw_data = pd.read_csv(os.path.join(ds_path, "train2id.txt"), sep="\t")

'''
179,951 triples
253 event relations
11384 entities
110 cluster themes
#train data - 143960 triples
#test data - 35991 triples
'''

def explain_data(split: str, data_split: pd.DataFrame, logger=None)->dict:
    # unique entity counts of source and target
    # unique relation counts
    # number of unique clusters
    # number of triples per month - 

    results= {}
    src_distribution = data_split["0"].value_counts()

    #logger.report_vector(title="{}".format(split), series="{} src_distribution".format(split), values=src_distribution.values.reshape(len(src_distribution.values), 1), iteration=0, labels=src_distribution.index.tolist(), xaxis="Entity IDs", yaxis="Count")

    tgt_distribution = data_split["1"].value_counts()
    rel_distribution = data_split["2"].value_counts()
    triples_per_month = data_split["5"].value_counts()
    cluster_distribution = data_split["6"].value_counts()
    unique_entities_distribution = pd.concat([data_split["0"], data_split["1"]], axis=0).value_counts()
    src_tgt_distribution = data_split.groupby(["0", "1"])["2"].count().to_dict()
    src_tgt_distribution = {"{}_{}".format(key[0], key[1]): value for key, value in src_tgt_distribution.items()}
    src_rel_distribution = data_split.groupby(["0", "2"])["1"].count().to_dict()
    src_rel_distribution = {"{}_{}".format(key[0], key[1]): value for key, value in src_rel_distribution.items()}
    url_distribution = data_split.groupby(["3"])["0"].count()

    results = {
        "src_distribution": src_distribution.to_dict(),
        "src_tgt_pair_distribution": src_tgt_distribution,
        "src_rel_pair_distribution": src_rel_distribution,
        "tgt_distribution": tgt_distribution.to_dict(),
        "unique_entities_distribution": unique_entities_distribution.to_dict(),
        "rel_distribution": rel_distribution.to_dict(),
        "triples_per_month": triples_per_month.to_dict(),
        "cluster_distribution": cluster_distribution.to_dict(),
        "url_triple_distribution": url_distribution.to_dict()
        }
    return results

def split_data(raw_data: pd.DataFrame)->(pd.DataFrame, pd.DataFrame):
    train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42, shuffle=True)
    train_data.to_csv("./data/train2id.txt", index=None)
    test_data.to_csv("./data/test2id.txt", index=None)
    return train_data, test_data




if __name__ == '__main__':
    train_data, test_data = split_data(raw_data)
    new_dataset = Dataset.create(
      dataset_name='gdelt_2020_clusters_train_split',
      dataset_project='datasets/gdelt', 
      parent_datasets=[ds.id]
    )
    dataset_task = Task.get_task(task_id=new_dataset.id)
    logger = new_dataset.get_logger()
    train_explanation = explain_data("train", train_data)
    test_explanation = explain_data("test", test_data)
    
    new_dataset.sync_folder(local_path="./data")
    dataset_task.upload_artifact("train_explanation", train_explanation)
    dataset_task.upload_artifact("test_explanation", test_explanation)
    new_dataset.upload()
    new_dataset.finalize()
    new_dataset.publish()





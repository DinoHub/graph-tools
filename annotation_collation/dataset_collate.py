from clearml import Task, Dataset, StorageManager
import pandas as pd
import os

def read_file(dataset_path:str, file_name:str):
    data = pd.read_csv('{}/{}'.format(dataset_path, file_name), sep="\t", header=None, dtype=str)[1:].reset_index(drop=True)
    return data

def add_annotations(main_df: pd.DataFrame, annotations: pd.DataFrame):
    main_df[5].astype("int32")
    new_df = main_df.merge(annotations,how="left", left_on=3, right_on="url_id").drop(columns=["url_id"])
    new_df.rename(columns={"cluster_id": 6}, inplace=True)
    return new_df    

def reformat_train_data(train_data:pd.DataFrame, date_threshold:int=202101)->pd.DataFrame:
    train_data[5] = train_data[4].str[:6]
    train_data[5] = train_data[5].astype("int32")
    train_data[3] = train_data[3].astype("int32")
    train_data = train_data[train_data[5]<date_threshold]
    return train_data

dataset_obj = Dataset.get(dataset_project="datasets/gdelt", dataset_name="gdelt_openke_format_w_extras", only_published=True)
dataset_path = dataset_obj.get_mutable_local_copy("./data")

entity_dict = read_file(dataset_path, "entity2id.txt")
relation_dict = read_file(dataset_path, "relation2id.txt")
train_data = read_file(dataset_path, "train2id.txt")

train_data = reformat_train_data(train_data)
annotations = pd.read_csv("labels.csv", dtype="int32")
new_df = add_annotations(train_data, annotations)
new_df.to_csv("{}/train2id.txt".format(dataset_path), sep="\t", index=None)
read_file(".", "anno2id.txt").to_csv("{}/cluster2id.txt".format(dataset_path), sep="\t", index=None)

new_dataset = Dataset.create(dataset_project='datasets/gdelt', dataset_name='raw_gdelt_2021_w_document_cluster_ids', parent_datasets=[dataset_obj.id])
#new_dataset.add_files(["{}/cluster2id".format(dataset_path)])
new_dataset.sync_folder(local_path="./data")
new_dataset.upload()
new_dataset.finalize()
new_dataset.publish()



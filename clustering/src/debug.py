from clearml import Dataset, StorageManager
import hydra
import torch
import ipdb

def load_data(doc_emb_path, er_emb_path):
    er_emb = torch.load(StorageManager.get_local_copy(er_emb_path))
    #doc_emb = pickle.load(open(StorageManager.get_local_copy(self.doc_emb_path),'rb')) 
    doc_emb = torch.load(StorageManager.get_local_copy(doc_emb_path)) 
    return er_emb, doc_emb

def numpy_convert(doc_emb):
    id_list = [key for key, _ in doc_emb.items()]
    doc_arrays = np.array([value.detach().numpy() for _, value in doc_emb.items()])
    return id_list, doc_arrays


import hydra

@hydra.main(config_path="../configs", config_name="main")
def run_cluster(cfg) -> None:
    doc_emb_path = cfg.main.doc_emb_path
    entity_embedding_path = cfg.main.entity_embedding_path
    er_emb, doc_emb = load_data(doc_emb_path, entity_embedding_path)
    ipdb.set_trace()

if __name__ == '__main__':
    run_cluster()

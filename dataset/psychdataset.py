from torch_geometric.data import InMemoryDataset, Data, download_url
import torch
import os
import pandas as pd

class PsychDataset(InMemoryDataset):
    def __init__(self, root="./data/", holdout_size=0.6, seed=None, use_graph: bool = False, use_feats: bool = True, use_snps: bool = False, use_age: bool = False, *args, **kwargs):
        self.use_feats = use_feats
        self.use_snps = use_snps
        self.use_graph = use_graph
        self.holdout_size = holdout_size
        self.seed = seed
        super().__init__(root=root, *args, **kwargs)
        self.data = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ["y_labels.txt", "DER-01_PEC_Gene_expression_matrix_normalized.txt"]
    
    @property
    def download_file_names(self):
        return ["DER-01_PEC_Gene_expression_matrix_normalized.txt"]

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.

        if not os.path.exists(os.path.join(self.raw_dir, self.raw_file_names[0])):
            self.download_label_dialog()

        base_path = "http://resource.psychencode.org/Datasets/Derived/"
        [download_url(os.path.join(base_path, url), self.raw_dir) for url in self.download_file_names]

    def download_label_dialog(self):
        import synapseclient 
        import getpass
        import shutil

        syn = synapseclient.Synapse() 
        print("Label Data has to be downloaded from primary repository after signup: https://www.synapse.org/#!Synapse:syn18915911")
        print("If you want to, you can enter your credentials here interactively to download or press CTRL+C, download the data yourself, store it in {} and resume.".format(os.path.join(self.raw_dir, self.raw_file_names[0])))
        synapse_username = input("Enter Synapse username:")
        password = getpass.getpass()
        syn.login(synapse_username, password) 
        
        # Obtain a pointer and download the data 
        syn18915911 = syn.get(entity='syn18915911'  ) 
        
        # Get the path to the local copy of the data file 
        filepath = syn18915911.path 
        shutil.copy(filepath, os.path.join(self.raw_dir, self.raw_file_names[0]))
        os.remove(filepath)
        print("Downloaded labels")

    def process(self):
        from sklearn.preprocessing import OneHotEncoder
        label_df = pd.read_csv("/mnt/storage/psychHGT/raw/y_labels.txt", header=0, usecols=["diagnosis"], index_col=["Synapse: individualID"]])
        label_df["diagnosis"] = label_df["diagnosis"].astype(str)
        label_df = label_df[label_df["diagnosis"] != "nan"]
        label_df = label_df[label_df["diagnosis"] != 'BP (not BP)']
        self.labelencoder = OneHotEncoder()
        label_df["onehot"] = self.labelencoder.fit_transform(df["diagnosis"].values.reshape(-1,1)).toarray()
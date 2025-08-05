import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import pandas as pd
import os
import requests as r

torch.cuda.empty_cache()
Target_Uniprot = "O60885"
E3_Uniprot = "Q96SW2"
Warhead_Smiles = "CC1=C(C)C2=C(S1)N1C(C)=NN=C1[C@H](CC(=O)OC(C)(C)C)N=C2C1=CC=C(Cl)C=C1"
E3_Smiles = "O=C1CCC(N2C(=O)C3=CC=CC=C3C2=O)C(=O)N1"
cell_type = "UNKNOWN"
target_seq = ""
E3_seq = ""
cell_type = cell_type.upper()
cell_type = cell_type.replace(';', '-')
cell_type = cell_type.replace('/', ',')
print("cell:",cell_type)
with open("cell_type_columns.txt", "r") as f:
    trained_cell_type_columns = [line.strip() for line in f]

try:
    baseUrl = "http://www.uniprot.org/uniprot/"
    currentUrl = baseUrl + Target_Uniprot + ".fasta"
    print(f"Requesting target sequence from: {currentUrl}")
    response = r.get(currentUrl)
    print(response.status_code) 
    cData = ''.join(response.text)
    i = cData.index('\n') + 1
    target_seq = cData[i:].strip()
    target_seq = target_seq.replace('\n', '') 
    print("target_seq:", target_seq)
except Exception as e:
    print(f"Error fetching target sequence: {e}")

try:
    # access E3 Uniprot seq
    baseUrl = "http://www.uniprot.org/uniprot/"
    currentUrl = baseUrl + E3_Uniprot + ".fasta"
    print(f"Requesting E3 sequence from: {currentUrl}")
    response = r.get(currentUrl)
    print(response.status_code) 
    cData = ''.join(response.text)
    i = cData.index('\n') + 1
    E3_seq = cData[i:].strip()
    E3_seq = E3_seq.replace('\n', '') 
    print("E3_seq:", E3_seq)
except Exception as e:
    print(f"Error fetching E3 sequence: {e}")

# init BERT
protbert_tokenizer = BertTokenizer.from_pretrained("prot_bert/")
protbert_model = BertModel.from_pretrained("prot_bert/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
protbert_model = protbert_model.to(device)

# BERT embeddings
def get_bert_embeddings(text_data, tokenizer, model):
    inputs = tokenizer(text_data, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output

# Morgan
def get_morgan_fp(smiles, nbits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
        array = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array
    else:
        return np.zeros((nbits,), dtype=np.int8)

class ImprovedClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=1):
        super(ImprovedClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.4)
        
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        return x


# define model path
model_paths = [
    "model1.pth",
    "model2.pth",
]

# load all models
models = []
for path in model_paths:
    model = ImprovedClassifier(input_dim=5335, num_classes=1).to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"Model {path} load successfully.")
        models.append(model)
    else:
        print(f"Model file {path} not found.")

# Prediction model
def predict_with_multiple_models(models, target_seq, e3_seq, warhead_smiles, linker_smiles, cell_type):
    print("linker_smiles:",linker_smiles)
    # BERT embedding
    target_embeddings = get_bert_embeddings(target_seq, protbert_tokenizer, protbert_model).to(device)
    e3_embeddings = get_bert_embeddings(e3_seq, protbert_tokenizer, protbert_model).to(device)
    
    # Morgan
    warhead_fingerprints = get_morgan_fp(warhead_smiles)
    linker_fingerprints = get_morgan_fp(linker_smiles)
    e3_fingerprints = get_morgan_fp(E3_Smiles)
    
    warhead_fp_features = torch.tensor(warhead_fingerprints, dtype=torch.float32).unsqueeze(0).to(device)
    linker_fp_features = torch.tensor(linker_fingerprints, dtype=torch.float32).unsqueeze(0).to(device)
    e3_fp_features = torch.tensor(e3_fingerprints, dtype=torch.float32).unsqueeze(0).to(device)

    cell_one_hot = pd.DataFrame({'cell_type': [cell_type]})
    cell_feature = pd.get_dummies(cell_one_hot['cell_type'], prefix='ct')
    cell_feature = cell_feature.reindex(columns=trained_cell_type_columns, fill_value=0)

    columns_with_ones = cell_feature.columns[cell_feature.eq(1).any()].tolist()
    print("cell type：", columns_with_ones)
    cell_feature = cell_feature.astype(float)
    cell_feature_tensor = torch.tensor(cell_feature.values, dtype=torch.float32).to(device)

    features = torch.cat([target_embeddings, e3_embeddings, warhead_fp_features, linker_fp_features, e3_fp_features, cell_feature_tensor], dim=1)

    scores = []
    for model in models:
        with torch.no_grad():
            output = model(features)
            prob = torch.sigmoid(output).cpu().numpy().flatten()[0]
            print("score: ",prob)
            scores.append(prob)

    return scores, np.mean(scores)

linker_df = pd.read_csv("data/linker.csv")
linker_smiles_list = linker_df["Smiles"].tolist()

model_names = ["model1", "model2"]

results = []

for linker_smiles in linker_smiles_list:
    scores, avg_score = predict_with_multiple_models(models, target_seq, E3_seq, Warhead_Smiles, linker_smiles, cell_type)
    result_row = {"Linker_Smiles": linker_smiles}
    for name, score in zip(model_names, scores):
        result_row[name] = score
    result_row["Average"] = avg_score
    results.append(result_row)

top_results = sorted(results, key=lambda x: x["Average"], reverse=True)[:30]

print("Top30 of Linker：")
for item in top_results:
    print(f'Linker_Smiles = "{item["Linker_Smiles"]}", Average: {item["Average"]:.4f}')

df = pd.DataFrame(top_results)
df.to_excel("docking_results.xlsx", index=False)
print("Top30 are save in docking_results.xlsx")


# results = []
# for linker_smiles in linker_smiles_list:
#     score = predict_with_multiple_models(models, target_seq, E3_seq, Warhead_Smiles, linker_smiles, cell_type)
#     results.append((linker_smiles, score))

# sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:150]

# print("Top 30 Linker：")
# for linker_smiles, avg_score in sorted_results:
#     print(f"Linker_Smiles = \"{linker_smiles}\", Avg_score: {avg_score:.4f}")

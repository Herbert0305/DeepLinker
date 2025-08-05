import torch
import pandas as pd
import os
from utils.fetch_sequence import fetch_uniprot_sequence
from utils.feature_extraction import get_bert_embeddings, get_morgan_fp, encode_cell_type
from utils.classifier import ImprovedClassifier

Target_Uniprot = "O60885"
E3_Uniprot = "Q96SW2"
Warhead_Smiles = "CC1=C(C)C2=C(S1)N1C(C)=NN=C1[C@H](CC(=O)OC(C)(C)C)N=C2C1=CC=C(Cl)C=C1"
E3_Smiles = "O=C1CCC(N2C(=O)C3=CC=CC=C3C2=O)C(=O)N1"
cell_type = "UNKNOWN"
model_path = "models/model1.pth"
cell_type_columns_path = "data/cell_type_columns.txt"
linker_csv_path = "data/linker.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

target_seq = fetch_uniprot_sequence(Target_Uniprot)
print("target_seq:", target_seq)
e3_seq = fetch_uniprot_sequence(E3_Uniprot)
print("e3_seq:", e3_seq)

target_emb = get_bert_embeddings(target_seq).to(device)
e3_emb = get_bert_embeddings(e3_seq).to(device)
print("e3_emb: ",e3_emb.shape)
fp_bits = 1024

Warhead_fingerprints = get_morgan_fp(Warhead_Smiles, fp_bits)
E3_fingerprints = get_morgan_fp(E3_Smiles, fp_bits)
warhead_fp = torch.tensor(Warhead_fingerprints, dtype=torch.float32).to(device)
e3_fp = torch.tensor(E3_fingerprints, dtype=torch.float32).to(device)
print("e3_fp: ",e3_fp.shape)

warhead_fp = warhead_fp.unsqueeze(0)
e3_fp = e3_fp.unsqueeze(0)

cell_tensor, columns_with_ones = encode_cell_type(cell_type, cell_type_columns_path)
cell_tensor = cell_tensor.to(device)
print("cell_tensor: ",cell_tensor.shape)
print(columns_with_ones)

model = ImprovedClassifier(input_dim=5335).to(device)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model {model_path} Loaded successfully.")
else:
    raise FileNotFoundError(f"Model file not found.: {model_path}")
model.eval()

linker_df = pd.read_csv(linker_csv_path)
linker_smiles_list = linker_df["Smiles"].tolist()


def predict_score(linker_smiles):
    linker_fp = torch.tensor(get_morgan_fp(linker_smiles), dtype=torch.float32).unsqueeze(0).to(device)
    input_tensor = torch.cat([target_emb, e3_emb, warhead_fp, linker_fp, e3_fp, cell_tensor], dim=1)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).cpu().numpy().flatten()[0]
    return prob

results = []
for smiles in linker_smiles_list:
    score = predict_score(smiles)
    print(f"Linker: {smiles}, Score: {score:.4f}")
    results.append((smiles, score))

top_linkers = sorted(results, key=lambda x: x[1], reverse=True)[:10]

print("\nTop Linkers:")
for linker, score in top_linkers:
    print(f'Linker_Smiles = "{linker}", {score:.4f}')

import json
import torch
import numpy as np
import os

from utils.fetch_sequence import fetch_uniprot_sequence
from utils.feature_extraction import get_bert_embeddings, get_morgan_fp, encode_cell_type
from utils.classifier import ImprovedClassifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("prediction_input.jason", "r") as f:
    data = json.load(f)

target_seq = fetch_uniprot_sequence(data["Target_Uniprot"])
print("target_seq:", target_seq)
e3_seq = fetch_uniprot_sequence(data["E3_Uniprot"])
print("e3_seq:", e3_seq)

target_emb = get_bert_embeddings(target_seq)
e3_emb = get_bert_embeddings(e3_seq)
print("e3_emb: ",e3_emb.shape)

fp_bits = 1024

Warhead_fingerprints = get_morgan_fp(data["Warhead_Smiles"], fp_bits)
Linker_fingerprints = get_morgan_fp(data["Linker_Smiles"], fp_bits)
E3_fingerprints = get_morgan_fp(data["E3_Smiles"], fp_bits)

warhead_fp = torch.tensor(Warhead_fingerprints, dtype=torch.float32).to(device)
linker_fp = torch.tensor(Linker_fingerprints, dtype=torch.float32).to(device)
e3_fp = torch.tensor(E3_fingerprints, dtype=torch.float32).to(device)
print("e3_fp: ",e3_fp.shape)

warhead_fp = warhead_fp.unsqueeze(0)
linker_fp = linker_fp.unsqueeze(0)
e3_fp = e3_fp.unsqueeze(0)


cell_tensor, columns_with_ones = encode_cell_type(data["cell_type"], "data/cell_type_columns.txt")
cell_tensor = cell_tensor.to(device)
print("cell_tensor: ",cell_tensor.shape)


features = torch.cat([target_emb, e3_emb, warhead_fp, linker_fp, e3_fp, cell_tensor], dim=1)
print(f"features: {features.shape}")

model = ImprovedClassifier(input_dim=features.shape[1]).to(device)
model_path = "models/model1.pth"

if os.path.exists(model_path):
    #model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print("Model loaded successfully, proceeding with prediction.")
    model.eval()
    with torch.no_grad():
        output = model(features)
        prob = torch.sigmoid(output).cpu().numpy().flatten()
        prediction = (prob >= 0.5).astype(int)
        print("Score: {:.4f}".format(prob[0]))
        print("Predition Score:", prediction[0])

else:
    print("Model file does not exist :", model_path)
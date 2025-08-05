import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from transformers import BertTokenizer, BertModel
from rdkit.Chem import rdFingerprintGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("prot_bert/")
bert_model = BertModel.from_pretrained("prot_bert/").to(device)

def get_bert_embeddings(seq):
    inputs = tokenizer(seq, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.pooler_output

def get_morgan_fp(smiles, nbits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits)
        fp = generator.GetFingerprint(mol)
        array = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, array)
        return array
    else:
        return np.zeros((nbits,), dtype=np.int8)

def encode_cell_type(cell_type, columns_file):
    cell_type = cell_type.upper()
    cell_type = cell_type.replace(';', '-')
    cell_type = cell_type.replace('/', ',')
    print("cell:",cell_type)
    with open(columns_file, "r") as f:
        trained_columns = [line.strip() for line in f]
    df = pd.DataFrame({'cell_type': [cell_type]})
    df_encoded = pd.get_dummies(df['cell_type'], prefix='ct').reindex(columns=trained_columns, fill_value=0)
    df_encoded = df_encoded.astype(float)
    columns_with_ones = df_encoded.columns[df_encoded.eq(1).any()].tolist()
    return torch.tensor(df_encoded.values, dtype=torch.float32), columns_with_ones

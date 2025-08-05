import gradio as gr
import torch
import json
import numpy as np
import os
from utils.fetch_sequence import fetch_uniprot_sequence
from utils.feature_extraction import get_bert_embeddings, get_morgan_fp, encode_cell_type
from utils.classifier import ImprovedClassifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def predict(Target_Uniprot, E3_Uniprot, Warhead_Smiles, Linker_Smiles, E3_Smiles, cell_type):
    if not cell_type:
        cell_type = "UNKNOWN"
    if not (Target_Uniprot and E3_Uniprot and Warhead_Smiles and Linker_Smiles and E3_Smiles):
        return "Error: All fields must be filled in!", ""

    target_seq = fetch_uniprot_sequence(Target_Uniprot)
    print("target_seq:", target_seq)
    e3_seq = fetch_uniprot_sequence(E3_Uniprot)
    print("e3_seq:", e3_seq)

    target_emb = get_bert_embeddings(target_seq)
    e3_emb = get_bert_embeddings(e3_seq)
    print("target_emb:", target_emb)

    fp_bits = 1024
    Warhead_fingerprints = get_morgan_fp(Warhead_Smiles, fp_bits)  
    Linker_fingerprints = get_morgan_fp(Linker_Smiles, fp_bits)  
    E3_fingerprints = get_morgan_fp(E3_Smiles, fp_bits)  

    warhead_fp = torch.tensor(Warhead_fingerprints, dtype=torch.float32).to(device)
    linker_fp = torch.tensor(Linker_fingerprints, dtype=torch.float32).to(device)
    e3_fp = torch.tensor(E3_fingerprints, dtype=torch.float32).to(device)

    warhead_fp = warhead_fp.unsqueeze(0)
    linker_fp = linker_fp.unsqueeze(0)
    e3_fp = e3_fp.unsqueeze(0)
    print("e3_fp: ",e3_fp.shape)

    cell_tensor, columns_with_ones = encode_cell_type(cell_type, "data/cell_type_columns.txt")
    cell_tensor = cell_tensor.to(device)
    print("cell_tensor: ",cell_tensor.shape)
    print("cell_type: ",columns_with_ones)

    features = torch.cat([target_emb, e3_emb, warhead_fp, linker_fp, e3_fp, cell_tensor], dim=1)
    print(f"features: {features.shape}")

    model = ImprovedClassifier(input_dim=features.shape[1]).to(device)
    model_path = "models/model1.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        with torch.no_grad():
            output = model(features)
            prob = torch.sigmoid(output).cpu().numpy().flatten()
            prediction = (prob >= 0.5).astype(int)
            print(f"Score: {prob[0]:.4f}", f"Precition Result: {prediction[0]}")
            return f"Score: {prob[0]:.4f}", f"Precition Result: {prediction[0]}"
    else:
        return "The model file does not exist", ""
# 通过 UniProt 获取序列
def get_sequence(uniprot_id):
    try:
        seq = fetch_uniprot_sequence(uniprot_id.strip())
        return seq if seq else "No matching sequence found."
    except Exception as e:
        return f"Error：{str(e)}"
# 清空输入
def clear_inputs(*args):
    return ["" for _ in args] + ["", ""]
# 检查所有字段是否填写
def check_inputs(*args):
    return all(args)
# Gradio UI 界面
with gr.Blocks(css=".gr-button { background: #FF5722; color: white; font-weight: bold; border-radius: 8px; padding: 10px 16px; }") as iface:
    gr.Image("logoko-2.png", label="DeepLinker Logo", show_label=False, interactive=False)
    gr.Markdown("""
    # 🧬 DeepLinker: Protein-PROTAC Interaction Prediction Platform
    DeepLinker is a platform for predicting potential interactions between proteins and PROTAC molecules, using a combination of protein sequence embeddings and chemical molecular fingerprints for modeling.
    ## 📝 Instructions for Use:
    1. Enter the UniProt ID to retrieve the protein sequence.
    2. Provide the SMILES expressions for the ligand, linker, and E3.
    3. Input the cell type (default is UNKNOWN).
    4. Click the "Generate Prediction Results" button.
    """)
    with gr.Row():
        with gr.Column():
            Target_ID = gr.Textbox(label="🎯 Target UniProt ID")
            fetch_target = gr.Button("Retrieve Target Protein Sequence")
            Target_seq = gr.Textbox(label="Target Protein Sequence", interactive=False)
            fetch_target.click(fn=get_sequence, inputs=Target_ID, outputs=Target_seq)
        with gr.Column():
            E3_ID = gr.Textbox(label="🔗 E3 UniProt ID")
            fetch_e3 = gr.Button("Retrieve E3 Protein Sequence")
            E3_seq = gr.Textbox(label="E3 Protein Sequence", interactive=False)
            fetch_e3.click(fn=get_sequence, inputs=E3_ID, outputs=E3_seq)
    SMILES1 = gr.Textbox(label="POI ligand SMILES")
    SMILES2 = gr.Textbox(label="Linker SMILES")
    SMILES3 = gr.Textbox(label="E3 ligand SMILES")
    Cell = gr.Textbox(label="Cell Type (default: UNKNOWN)", value="UNKNOWN")
    gr.Markdown("🔍 [SwissADME](http://www.swissadme.ch/) Can help you convert the structure to SMILES.")
    output1 = gr.Textbox(label="📈 Predicted Score")
    output2 = gr.Textbox(label="📊 Predicted Result")
    generate_button = gr.Button("Generate Prediction Results")
    clear_button = gr.Button("Clear Input")
    generate_button.click(
        fn=lambda a, b, c, d, e, f: predict(a, b, c, d, e, f) if check_inputs(a, b, c, d, e, f) else ("Error: All fields are required!", ""),
        inputs=[Target_ID, E3_ID, SMILES1, SMILES2, SMILES3, Cell],
        outputs=[output1, output2]
    )
    clear_button.click(
        fn=clear_inputs,
        inputs=[Target_ID, E3_ID, SMILES1, SMILES2, SMILES3, Cell],
        outputs=[Target_ID, E3_ID, SMILES1, SMILES2, SMILES3, Cell, output1, output2]
    )
    gr.Markdown("""
    ## ℹ️ Result Interpretation：
    - Predicted Score: Indicates the probability that the molecule meets the model's criteria.
    - Prediction Result: Classified based on the following rule:
      Within the concentration range of 0 ~ 100 nM, if the degradation level is > 30%, it is classified as Active; otherwise, it is Inactive.
    """)

iface.launch()

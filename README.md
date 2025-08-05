# DeepLinker: A Deep Learning Method for Predicting the Protein Degradation of PROTACs
**DeepLinker**  is a high-precision AI model designed for predicting the degradation efficacy of proteolysis-targeting chimeras (PROTACs). It integrates molecular structure, protein sequences, and cellular context to accurately estimate degradation activity across diverse chemical spaces.

🚀 How to Use
Once the environment is configured, you can run the following files depending on your use case:
1. main_platform.py – Launch the Gradio Web Platform ✅ Recommended
This launches a simple and user-friendly Gradio interface for interacting with the DeepLinker platform.

python main_platform.py

2. single_prediction.py – Single Sample Activity Prediction
Use this script to make an activity prediction for a single sample.

python single_prediction.py

3. Top_linker_prediction.py – Batch Prediction with Top-k Selection
Use this script to test multiple samples and select the top-k candidates based on predicted activity scores.

python Top_linker_prediction.py

The complete ProBERT model can be downloaded from Hugging Face.



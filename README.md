# Exploring Antifungal Agents to Combat Escalating Drug Resistance Using Language and Diffusion Models

This repository contains the code and resources for training and generating antifungal agents using language and diffusion models. The project aims to address the growing issue of drug resistance in fungi by exploring novel compounds.

## Repository Structure

- `Data/`: Directory to store training datasets, trained models, and the compound library.
- `Train.py`: Script for training the model.
- `Generate.py`: Script for generating new compounds.

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-url.git
   cd your-repo-directory
   ```

2. Set up the conda environment:
   ```bash
   conda create -n MolDiffusion 
   conda activate MolDiffusion
   conda install python=3.10
   pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
   pip install fair-esm faiss-cpu tqdm einops pytorch-lightning
   ```

### Data Preparation

1. Download the training dataset from [here](https://drive.google.com/file/d/1Jq1JIXAiQeOsuJ5oCB_iRbwvS0aTq7CD/view?usp=drive_link) and place it in the `Data/` directory.
2. Download the pre-trained model from [here](https://drive.google.com/file/d/1XfvFwwsiXYP9Vr65q9vZNIqUxN3tUUWs/view?usp=drive_link) and place it in the `Data/` directory.
3. Download the compound library from [here](https://drive.google.com/file/d/1-6Ao6FxGZwqfVTOXADRJ673WjInaNVl3/view?usp=drive_link), unzip it, and place the `vs_data` folder inside the `Data/` directory.

### Training

To train the model, run the following command:
```bash
torchrun --nproc_per_node=GPU_NUM Train.py > ./model/train.log
```
Replace `GPU_NUM` with the number of GPUs you wish to use.

### Generation

To generate compounds, use the following command:
```bash
python Generate.py --proseq AAAAAAAAA --savepath protein_name
```
Replace `AAAAAAAAA` with the desired protein sequence and `protein_name` with the name you want to save the generated protein as.





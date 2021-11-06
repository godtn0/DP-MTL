# DP-MTL - Official PyTorch Implementation

## Dependencies
- pip install -r requirements.txt

## Usage
- unzip "./data/enem_data/mappers.zip".
- unzip "./data/enem_data/enem_dep.zip".
- Modify the config file in "./configs/" to fit on your environment.
### Snapshot Models
python snap_trainer.py
- If you move enem_data to other dir, modify "data.root" in "./configs/irt_enem.yaml".
### Sequential Models
python seq_trainer.py
- If you move enem_data to other dir, modify "data.root" in "./configs/seq_irt_enem.yaml".

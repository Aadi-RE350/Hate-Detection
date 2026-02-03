# Hate-Detection (MetaHate)

Detecting hate and abusive speech using a hybrid approach that combines text-based AI models (BERT + CNN) with user-based metadata.

## 📁 Project Structure
* **bert_cnn_frozen.pth**: Pre-trained model weights for the BERT-CNN architecture.
* **metahate_final_model_saved/**: Directory containing the serialized final model.
* **Training_Model.py**: Main execution script for model training.
* **Bert_Training.py**: Specialized script for fine-tuning the BERT transformer.
* **Data-PS.py / Data_analysis.ipynb**: Scripts for data preprocessing and exploratory analysis.

## 🚀 Getting Started
1. Ensure you have the required CSV datasets in the `/csv` or `/Data` folders.
2. Run `Data-PS.py` for preprocessing.
3. Execute `Training_Model.py` to begin the training pipeline.
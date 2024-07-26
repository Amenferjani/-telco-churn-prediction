# Telco Churn Prediction

This project aims to predict customer churn in a telecom company using a neural network implemented in PyTorch.

## Project Structure

- `.gitignore`: Specifies files and directories to be ignored by git.
- `Telco-Customer-Churn.csv`: The dataset containing customer information and churn status.
- `churnModel.pt`: The trained model saved as a PyTorch state dictionary.
- `model.py`: Defines the neural network architecture.
- `train.py`: Script for training the neural network.

## Requirements

To run the project, you need the following libraries:

- PyTorch
- Pandas
- Scikit-learn

You can install the required libraries using:

```bash
pip install torch pandas scikit-learn
git clone https://github.com/yourusername/telco-churn-prediction.git
cd telco-churn-prediction

python train.py

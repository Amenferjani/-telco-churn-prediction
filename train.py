import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import copy
from model import Model

data = pd.read_csv(
    "c:/Users/AMEN/Desktop/pytorch/telco-churn-prediction/Telco-Customer-Churn.csv")

data.drop(['customerID'], axis=1, inplace=True)

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

data['TotalCharges'].fillna(0, inplace=True)

binary_cols = ['gender', 'Partner', 'Dependents',
                'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    data[col] = data[col].apply(
        lambda x: 1 if x == 'Yes' or x == 'Male' else 0)

multi_class_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
data = pd.get_dummies(data, columns=multi_class_cols)

dataFrame = data.drop(['Churn'], axis=1)
target = data['Churn']

xTrain, xTest, yTrain, yTest = train_test_split(
    dataFrame, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

xTrain = torch.FloatTensor(xTrain)
xTest = torch.FloatTensor(xTest)
yTrain = torch.LongTensor(yTrain.values)
yTest = torch.LongTensor(yTest.values)

inputData = dataFrame.shape[1]
model = Model(inputData=inputData)
torch.manual_seed(50)
print (inputData)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)


epochs = 5000
patience = 2
best_loss = float('inf')
epochs_without_improvement = 0
best_model_weights = copy.deepcopy(model.state_dict())


for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    yPred = model(xTrain)
    loss = criterion(yPred.squeeze(), yTrain.float())
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        model.eval()
        with torch.no_grad():
            yEval = model(xTest)
            val_loss = criterion(yEval.squeeze(), yTest.float())
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(
            f"Epoch {epoch}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")

model.load_state_dict(best_model_weights)
# torch.save(model.state_dict(),
#             "c:/Users/AMEN/Desktop/pytorch/telco-churn-prediction/churnModel.pt")


model.eval()
with torch.no_grad():
    yEval = model(xTest)
    test_loss = criterion(yEval.squeeze(), yTest.float())
    print("Evaluation Loss:", test_loss.item())

yEval_prob = torch.sigmoid(yEval).squeeze()
yEval_class = (yEval_prob >= 0.5).float()

mse = mean_squared_error(yTest, yEval_class)
mae = mean_absolute_error(yTest, yEval_class)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

print("\n ****************** \n")
total_error = 0
with torch.no_grad():
    for i in range(len(xTest)):
        yVal = model(xTest[i])
        yVal_prob = torch.sigmoid(yVal).item()
        yVal_class = 1.0 if yVal_prob >= 0.5 else 0.0
        error = abs(yVal_class - yTest[i].item())
        total_error += error
        print(
                f"Predicted: {yVal_class}, Actual: {yTest[i].item()}, Error: {error}")

average_error = total_error / len(xTest)
print("Average Error:", average_error)

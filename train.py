from __future__ import division
from __future__ import print_function

import os
import random
import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.beta import Beta
from torch.utils.data import DataLoader, TensorDataset, Subset
from accelerate import Accelerator

from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler


from model import RuleEncoder, DataEncoder, Net, DataonlyNet
from model import *
from utils_dp import *



from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go

import torch
import torch.multiprocessing as mp
from torch import nn, optim


df= pd.read_parquet("data_pricing_germany.parquet")

features=['quantity_supply','product_name','turnover', 'material_total_nb_cylinder','customer_id','material_id']
target= 'cylinder_price'

label_encoder = LabelEncoder()
df['product_name'] = label_encoder.fit_transform(df['product_name'])

df['material_id']=label_encoder.fit_transform(df['material_id'])


df_filtered = df[df['cylinder_price'] <= 400]


X = df_filtered[features]
y = df_filtered[target]



sample_size = 0.1

X_sample = X.sample(frac=sample_size, random_state=42)
y_sample = y.loc[X_sample.index]
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X_sample, y_sample, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.1, random_state=42)


scaler_X = StandardScaler()
scaler_train = scaler_X.fit(X_train)  # Fit sur X_train

scaler_y = StandardScaler()
scaler_y.fit(y_train.values.reshape(-1, 1))  # Fit sur y_train

X_train = scaler_train.transform(X_train)
X_valid = scaler_train.transform(X_valid)
X_test = scaler_train.transform(X_test)

y_train = scaler_y.transform(y_train.values.reshape(-1, 1)).flatten()
y_valid = scaler_y.transform(y_valid.values.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()



def tensorize(data, device=torch.device("cpu")):
    return torch.tensor(data, dtype=torch.float32, device=device)


X_train= tensorize(X_train)
X_valid = tensorize(X_valid)
X_test= tensorize(X_test)



# Conversion des cibles en tenseurs
y_train = torch.tensor(y_train, dtype=torch.float32)
y_valid = torch.tensor(y_valid, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
#y_test = torch.tensor(y_test.values, dtype=torch.float32)




# Création des DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

# Échantillonnage d'un petit sous-ensemble des données d'entraînement
#sample_size = 30000  # Définir la taille de l'échantillon souhaitée
indices = np.random.choice(len(train_loader.dataset), size=sample_size, replace=False)
sampled_train_dataset = Subset(train_loader.dataset, indices)

# Mise à jour du DataLoader pour utiliser le sous-ensemble échantillonné
#train_loader = DataLoader(sampled_train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=64, shuffle=False )
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

# Hyperparameters
SKIP = False 
input_dim = X.shape[1]
print(input_dim)
output_dim = 1
output_dim_encoder = input_dim
input_dim_encoder = 6
hidden_dim_encoder = 64
hidden_dim_db = 64
n_layers = 2
seed = 42
batch_size = 32
hidden_dim = 64
lr = 0.001
epochs = 100
early_stopping_thld = 10
valid_freq = 5
rule_threshold = -0.2
constraint = 0
scale = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rule_encoder = RuleEncoder(input_dim, output_dim_encoder)
data_encoder = DataEncoder(input_dim, output_dim_encoder)
model = SimpleNet(input_dim=X_train.shape[1], hidden_dim=64, output_dim=1).to('cpu')

#model = Net(input_dim, output_dim, rule_encoder, data_encoder, n_layers=n_layers).to(device)
#model = DataonlyNet(input_dim, output_dim, data_encoder, hidden_dim=4, n_layers=2, skip=False, input_type='state')
total_params = sum(p.numel() for p in model.parameters())
print("total parameters: {}".format(total_params))

loss_rule_func = lambda x,y: torch.mean(F.relu(x-y))

# Optimiseur et fonction de perte
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
accelerator = Accelerator()

model, optimizer, training_dataloader = accelerator.prepare(model, optimizer, train_loader) 
saved_filename = 'dataonly'

saved_filename =  os.path.join('saved_models', saved_filename)
print('saved_filename: {}\n'.format(saved_filename))


# Initial loss calculation
x_init, y_init = next(iter(train_loader))
x_init, y_init = x_init.to(device), y_init.to(device)
quantity_init = x_init[:, 0]
output_init = model(x_init).squeeze(1)

loss_task_0 = criterion(output_init, y_init)
loss_rule_0 = Custom_loss_rule(quantity_init, output_init, threshold=0)

rho = loss_rule_0.item() / loss_task_0.item()


beta_param= [1]
best_val_loss = float('inf')
train_losses = []
valid_losses = []
alpha_distribution = Beta(float(beta_param[0]), float(beta_param[0]))
for epoch in tqdm(range(epochs)):
    model.train()
    epoch_train_loss = []
    for x_train, y_train_batch in train_loader:
        optimizer.zero_grad()
        alpha = alpha_distribution.sample().item()

        x_train, y_train_batch = x_train.to(device), y_train_batch.to(device)
        quantity = x_train[:, 0]
        output = model(x_train).squeeze(1)
        
        loss_task = criterion(output, y_train_batch)
        loss_rule = Custom_loss_rule(quantity, output, threshold=0)

        loss = alpha * loss_rule + rho * (1-alpha) * loss_task
        epoch_train_loss.append(loss.item())
        accelerator.backward(loss)
        optimizer.step()
    
    train_losses.append(np.mean(epoch_train_loss))

    if (epoch + 1) % valid_freq == 0:
        model.eval()
        val_loss_task = 0
        val_loss_rule = 0
        val_ratio = 0
        total_samples = 0
        
        with torch.no_grad():
            for x_val, y_val in valid_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                output = model(x_val).squeeze(1)
                quantity = x_val[:, 0]
                
                val_loss_task += criterion(output, y_val).item() * x_val.size(0)
                val_loss_rule += Custom_loss_rule(quantity, output, threshold=0).item() * x_val.size(0)
                val_ratio += verification(quantity, output)* x_val.size(0)
                total_samples += x_val.size(0)
        
        avg_val_loss_task = val_loss_task / total_samples
        avg_val_loss_rule = val_loss_rule / total_samples
        avg_val_ratio = val_ratio / total_samples
        valid_losses.append(avg_val_loss_task)
        
        if avg_val_loss_task < best_val_loss:
            counter_early_stopping = 0
            best_val_loss = avg_val_loss_task
            print(f'[Valid] Epoch: {epoch} Loss(Task): {avg_val_loss_task:.6f} Loss(Rule): {avg_val_loss_rule:.6f} Ratio(Rule): {avg_val_ratio:.3f} (alpha: 0.2)\t best model is updated %%%')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss
            }, saved_filename)
        else:
            counter_early_stopping += 1
            print(f'[Valid] Epoch: {epoch} Loss(Task): {avg_val_loss_task:.6f} Loss(Rule): {avg_val_loss_rule:.6f} Ratio(Rule): {avg_val_ratio:.3f} (alpha: 0.2) ({counter_early_stopping}/{early_stopping_thld})')
            if counter_early_stopping >= early_stopping_thld:
                print("Early stopping triggered")
                break



# Charger le meilleur modèle
checkpoint = torch.load('saved_models/dataonly')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Best model loss: {:.6f} at epoch: {}".format(checkpoint['loss'], checkpoint['epoch']))


# Évaluation finale sur l'ensemble de test
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for data_x, y in test_loader:
        data_x, y = data_x.to(device), y.to(device)
        output = model(data_x)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(output.cpu().numpy())
########################################################################

########################################################################################
# Convertir en tableaux numpy
y_true = np.array(y_true)
y_pred = np.array(y_pred)
# Supposons que vous ayez déjà inversé les prédictions dans l'échelle d'origine
y_pred_original_scale = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_true_original_scale = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()

# Vérifiez et assurez-vous que y_true et y_pred_original_scale ont la même longueur
assert len(y_true_original_scale) == len(y_pred_original_scale)

# Calcul des scores
mse = mean_squared_error(y_true_original_scale, y_pred_original_scale)
mae = mean_absolute_error(y_true_original_scale, y_pred_original_scale)
r2 = r2_score(y_true_original_scale, y_pred_original_scale)

print(f'R-squared (R²): {r2:.2f}')
print(f'mean absolute error (Mae): {mae:.2f}')


def plot_results(train_losses, valid_losses, y_true, y_pred):
    epochs_train = range(1, len(train_losses) + 1)
    epochs_valid = range(valid_freq, epochs + 1, valid_freq)  # Calcul des epochs pour les pertes de validation
    
    plt.figure(figsize=(15, 5))

    # Plot training and validation losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs_train, train_losses, label='Training Loss')
    if valid_losses:
        plt.plot(epochs_valid, valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Plot predicted vs. actual values
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')
    plt.plot()
    
    plt.tight_layout()
    plt.show()


plot_results(train_losses, valid_losses, y_true_original_scale, y_pred_original_scale)

# Best model
def evaluate_model(model, test_loader, criterion, scaler_y, alphas):
    avg_losses = []
    r2_scores = []
    rmses = []
    ratios = []

    for alpha in alphas:
        model.eval()
        y_true = []
        y_pred = []
        test_loss_task = 0
        test_ratio = 0
        total_samples = 0
        
        with torch.no_grad():
            for test_x, test_y in test_loader:
                test_x, test_y = test_x.to(device), test_y.to(device)
                quantity = test_x[:, 0]
                output = model(test_x,alpha=alpha).squeeze(1)
                
                y_true.extend(test_y.cpu().numpy())
                y_pred.extend(output.cpu().numpy())

                test_loss_task += criterion(output, test_y).item() * test_x.size(0)
                test_ratio += verification(quantity, output)* test_x.size(0)
                total_samples += test_x.size(0)
        
        avg_test_loss_task = test_loss_task / total_samples
        avg_test_ratio = test_ratio / total_samples
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        y_pred_original_scale = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_true_original_scale = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()

        rmse = mean_squared_error(y_true_original_scale, y_pred_original_scale, squared=False)
        mae = mean_absolute_error(y_true_original_scale, y_pred_original_scale)
        r2 = r2_score(y_true_original_scale, y_pred_original_scale)
        
        assert len(y_true_original_scale) == len(y_pred_original_scale)
        
        avg_losses.append(avg_test_loss_task)
        r2_scores.append(r2)
        rmses.append(rmse)
        ratios.append(avg_test_ratio)
        
        print('Test set: Average loss: {:.8f} (alpha: {})'.format(avg_test_loss_task, alpha))
        print(f'R-squared (R²): {r2:.2f}')
        print(f'Root Absolute Error (RMSE): {rmse:.2f}')
        print("Ratio of verified predictions: {:.6f} (alpha: {})".format(avg_test_ratio, alpha))

    return avg_losses, r2_scores, rmses, ratios



# Paramètres et initialisation
alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]
avg_losses, r2_scores, maes, ratios = evaluate_model(model, test_loader, criterion, scaler_y, alphas)

# Plotting the metrics
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(alphas, avg_losses, marker='o')
plt.title('Average Loss vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Average Loss')

plt.subplot(2, 2, 2)
plt.plot(alphas, r2_scores, marker='o')
plt.title('R-squared (R²) vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('R-squared (R²)')

plt.subplot(2, 2, 3)
plt.plot(alphas, maes, marker='o')
plt.title('Mean Absolute Error (MAE) vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Mean Absolute Error (MAE)')

plt.subplot(2, 2, 4)
plt.plot(alphas, ratios, marker='o')
plt.title('Ratio of Verified Predictions vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Ratio of Verified Predictions')

plt.tight_layout()
plt.show()

# Créer un DataFrame avec les données
data = {
    'Alpha': alphas,
    'Average Loss': avg_losses,
    'R-squared (R²)': r2_scores,
    'Mean Absolute Error (MAE)': maes,
    'Ratio of Verified Predictions': ratios
}
df = pd.DataFrame(data)

# Enregistrer le DataFrame dans un fichier CSV
df.to_csv('model_metrics.csv', index=False)
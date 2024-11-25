import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt  # Importar para gráficos
import torch
import torch.nn as nn
import random  # Importar random para selección aleatoria
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import norm
import time
import pandas as pd

print("Inicio del script")
# Configuración para pruebas locales
data_local = True  # Cambiar a False para usar todo el dataset

# Verificar el número de procesadores que PyTorch está utilizando
print(f"Número de procesadores disponibles: {torch.get_num_threads()}")

# Configurar el número de procesadores a utilizar (por ejemplo, 8)
torch.set_num_threads(12)
print(f"Número de procesadores configurados: {torch.get_num_threads()}")

print("Procesando datos...")
# Función para procesar imágenes desde el archivo NDJSON
def image_ndjson2numpy(data, img_size=51, size_line=10, only_recognized=True):
    flagRecog = lambda x: x if only_recognized else lambda x: True
    imgSize = (img_size, img_size) if isinstance(img_size, int) else img_size
    images = []
    labels = []
    for xx in data:
        if flagRecog(xx['recognized']):
            try:
                yy = np.array(xx['drawing'], dtype=np.object_)[:,:2].T
                FF = [np.concatenate(yy[:,i], dtype=float).astype(int).reshape(2,-1).T[:,None] for i in range(len(xx['drawing']))]
                minmaxFF = np.concatenate([np.c_[i.min(0), i.max(0)] for i in FF])
                minFF = minmaxFF[:,:2].min(0, keepdims=True)
                maxFF = minmaxFF[:,2:].max(0, keepdims=True)
                FF = [i - minFF[None] for i in FF]
                img = np.zeros(((maxFF - minFF).tolist()[0][1], (maxFF - minFF).tolist()[0][0]), np.uint8)
                images.append(cv2.resize(cv2.polylines(img, FF, False, 1, size_line), imgSize))
                labels.append(xx['class'])
            except:
                continue
    return np.array(images), labels

# Cargar datos desde el archivo combinado NDJSON
output_file = './combined_quickdraw.ndjson'
with open(output_file, 'r') as f:
    data = [json.loads(line) for line in f]

if data_local:
    data = random.sample(data, 1000)  # Seleccionar 1000 muestras al azar
    print(f"Usando {len(data)} muestras seleccionadas al azar para pruebas locales.")


# Procesar imágenes y etiquetas
images, labels = image_ndjson2numpy(data, img_size=51)

# Crear un mapeo de clases a índices
classes = list(set(labels))
class_to_label = {cls: idx for idx, cls in enumerate(classes)}
labels = np.array([class_to_label[label] for label in labels])

# Dividir los datos en entrenamiento y validación
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Dataset personalizado para PyTorch
class QuickDrawDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images*255 
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Transformaciones
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = QuickDrawDataset(train_images, train_labels, transform=transform)
val_dataset = QuickDrawDataset(val_images, val_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=5000, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=5000, shuffle=False)

# Modelo MLP con inicialización Xavier
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

        # Inicialización Xavier
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Aplanar
        x = self.sigmoid(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# Entrenamiento del modelo
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=1000, early_stopping_patience=15, lr=0, run=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_val_loss = float('inf')
    early_stopping_counter = 0

    # Listas para almacenar métricas
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Promedio de pérdida de entrenamiento
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validación
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Gráficos del entrenamiento
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    # Gráfico de Pérdidas
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Gráfico de Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    
    # Guardar los gráficos como archivo PNG
    plt.savefig(f'/home/fespinoza/proyecto_ic/mlp_xel_lr/training_results_mlp_{lr}_{run}.png', dpi=300)  # Guardar en alta resolución
    #plt.savefig(f'/Users/felipeespinoza/Documents/00 Personal Drive/04 DIASMA/Inteligencia Computacional/ENTREGA1/cnn_training_project/graph_ej3/cnn_results_lr{lr}_run{run}.png', dpi=300)  # Guardar con nombre dinámico
    plt.show()
    plt.close()  # Cerrar la figura actual

    return train_losses, val_losses, val_accuracies  # Devuelve las métricas

print("Distribución en el conjunto de entrenamiento:", np.bincount(train_labels))
print("Distribución en el conjunto de validación:", np.bincount(val_labels))


# Configuración de hiperparámetros
input_size = 51 * 51
hidden_size = 500
num_classes = len(classes)
# Tasa de aprendizaje variable
learning_rate = [1e-2, 1e-1, 1, 10]
epochs = 1000
results = []
lr = []

# Función para generar las curvas ROC
def plot_roc_curve(model, val_loader, classes, lr=0, run=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = label_binarize(all_labels, classes=range(len(classes)))

    # Curvas ROC por clase
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_preds[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Clase {classes[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="No Skill")
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.title("Curvas ROC")
    plt.legend(loc="best")
    # Guardar los gráficos como archivo PNG
    plt.savefig(f'/home/fespinoza/proyecto_ic/mlp_xel_results/roc_curve_cross_e_{lr}_{run}.png', dpi=300) # Guardar en alta resolución
    plt.show()
    plt.close()  # Cerrar la figura actual

# Función para generar la matriz de confusión
def plot_confusion_matrix(model, val_loader, classes, lr=0, run=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="viridis")
    plt.title("Matriz de Confusión")

    # Guardar los gráficos como archivo PNG
    plt.savefig(f'/home/fespinoza/proyecto_ic/mlp_xel_lr/consusion_matrix_celoss_{lr}_{run}.png', dpi=300) # Guardar en alta resolución
    plt.show()
    plt.close()  # Cerrar la figura actual

# Función para graficar la curva DET
def plot_det_curve(model, val_loader, classes, lr=0, run=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = label_binarize(all_labels, classes=range(len(classes)))

    plt.figure()
    for i in range(len(classes)):
        fpr, fnr, _ = roc_curve(all_labels[:, i], all_preds[:, i])
        fnr = 1 - fpr
        plt.plot(norm.ppf(fpr), norm.ppf(fnr), label=f"Clase {classes[i]}")

    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("FNR (False Negative Rate)")
    plt.title("Curvas DET")
    plt.legend(loc="best")
    plt.grid(True)

    # Guardar los gráficos como archivo PNG
    plt.savefig(f'/home/fespinoza/proyecto_ic/mlp_xel_lr/det_curve_cross_e_{lr}_{run}.png', dpi=300) # Guardar en alta resolución
    plt.show()
    plt.close()  # Cerrar la figura actual

# Ciclo de tasas de aprendizaje
for lr in learning_rate:
    print(f"\nEvaluando tasa de aprendizaje: {lr}")
    accuracies = []
    training_times = []
    for run in range(5):
        print(f"Repetición {run + 1} para LR = {lr}")

        # Entrenamiento con Entropía Cruzada y Xavier Initialization
        model = MLP(input_size, hidden_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)

        # Medir tiempo de entrenamiento
        start_time = time.time()
        train_losses, val_losses, val_accuracies = train_model(model,train_loader,
         val_loader, criterion, optimizer, epochs=epochs, lr=lr, run=run +1)
        training_time = time.time() - start_time

        # Almacenar resultados
        accuracies.append(max(val_accuracies))  # Máxima precisión en validación
        training_times.append(training_time)
        
        # Llamar a la función
        plot_confusion_matrix(model, val_loader, classes, lr, run)

        # Llamar a la función
        plot_roc_curve(model, val_loader, classes, lr, run)
        
        # Llamar a la función
        plot_det_curve(model, val_loader, classes, lr, run)

        # Calcular promedio y desviación estándar
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        mean_time = np.mean(training_times)

        results.append({
            'learning_rate': lr,
            'run': run,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'mean_time': mean_time,
        })

        results_df = pd.DataFrame(results)
        print(f"Learning Rate: {lr} RUN: {run}\n{results_df}")

        # Gráficos
        plt.figure(figsize=(12, 6))

        # Gráfico de precisión
        plt.subplot(1, 2, 1)
        plt.errorbar(results_df['learning_rate'], results_df['mean_accuracy'], yerr=results_df['std_accuracy'], fmt='o-')
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Validation Accuracy (%)')
        plt.title('Impacto de Learning Rate en Accuracy')

        # Gráfico de tiempo de entrenamiento
        plt.subplot(1, 2, 2)
        plt.plot(results_df['learning_rate'], results_df['mean_time'], 'o-')
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Training Time (s)')
        plt.title('Impacto de Learning Rate en Tiempo de Entrenamiento')

        # Guardar los gráficos como archivo PNG
        #plt.savefig(f'/Users/felipeespinoza/Documents/00 Personal Drive/04 DIASMA/Inteligencia Computacional/ENTREGA1/cnn_training_project/graph_ej3/lr_impact.png', dpi=300)  # Guardar con nombre dinámico
        plt.savefig('/home/fespinoza/proyecto_ic/mlp_xel_lr/mlp_results_lr{lr}_run{run}.png', dpi=300) # Guardar en alta resolución

        plt.tight_layout()
        plt.show()



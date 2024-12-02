import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
from scipy.stats import norm
import random
import torch.optim as optim
from torch import tensor
import torch.nn.functional as F
from sklearn.metrics import classification_report
import sys
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay


# Verificar el número de procesadores que PyTorch está utilizando
print(f"Número de procesadores disponibles: {torch.get_num_threads()}")

# Configurar el número de procesadores a utilizar
torch.set_num_threads(8)
print(f"Número de procesadores configurados: {torch.get_num_threads()}")

data_numpy = np.load('data_all.npy', allow_pickle=True)
print(f"Tamaño en memoria: {data_numpy.nbytes / 1e9:.2f} GB")


# Configuración para pruebas locales
data_local = False  # Cambiar a False para usar todo el dataset

if data_local == True:
    # Barajar las filas de la matriz
    #np.random.seed(42)  # Semilla para reproducibilidad
    indices = np.random.permutation(len(data_numpy))  # Crear índices aleatorios
    data_numpy_shuffled = data_numpy[indices]  # Barajar la matriz completa
    
    # Tomar un subconjunto (por ejemplo, 10,000 muestras)
    subset_size = 1000
    data_numpy = data_numpy_shuffled[:subset_size]
    batch = 10
else:
    batch = 5000

# Separar imágenes y etiquetas después del muestreo
data_sample_images = np.array([item[0] for item in data_numpy])
data_sample_labels = np.array([item[1] for item in data_numpy])


# Crear un mapeo de clases a índices
classes = list(set(data_sample_labels))
class_to_label = {cls: idx for idx, cls in enumerate(classes)}
data_sample_labels = np.array([class_to_label[label] for label in data_sample_labels])
data_sample_labels = np.eye(len(class_to_label), dtype=np.float32)[data_sample_labels]



## Dividir los datos con muestreo estratificado
X_train, X_test, y_train, y_test = train_test_split(
    data_sample_images, data_sample_labels, test_size=0.2, random_state=42, 
    stratify=data_sample_labels)

# Transformaciones
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.to(torch.float32)),
    transforms.Normalize((0.5,), (0.5,))
])


# Dataset personalizado para PyTorch
class QuickDrawDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
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


# Modelo MLP
class MLP(nn.Module):
    def __init__(self, input_size=2601, hidden_size=500, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Inicialización Xavier
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Aplanar
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# Entrenamiento del modelo
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=1000, 
                early_stopping_patience=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_val_loss = float('inf')
    early_stopping_counter = 0

    # Listas para almacenar métricas
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Listas para almacenar
    y_pred = []
    y_true = []
    outputs_list = []
    

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
                y_pred.extend(predicted.cpu().numpy())  # Guardar predicciones
                
                # Guardar todas las predicciones (logits) y etiquetas reales
                _, true_labels = torch.max(labels, 1)  # Convertir etiquetas one-hot a índices
                outputs_list.extend(outputs.cpu().numpy()) # Guardar logits
                y_true.extend(true_labels.cpu().numpy())    # Guardar valores reales como índices



                total += labels.size(0)
                #correct += (predicted == labels).sum().item()
                correct += (predicted == labels.argmax(1)).sum().item()

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
        
    return train_losses, val_losses, val_accuracies, accuracy, outputs_list, y_pred, y_true

def train_graphs(train_losses, val_losses, val_accuracies):
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
    plt.savefig('/training_results_np_mse.png', dpi=300)  # Guardar en alta resolución
    #plt.show()
    plt.close()


# Transformando los datos a Tensores
train_dataset = QuickDrawDataset(X_train, y_train, transform=transform)
val_dataset = QuickDrawDataset(X_test, y_test, transform=transform)

# Preparar los datos de entrenamiento
train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

# Configuración de hiperparámetros
input_size = 51 * 51
hidden_size = 500
learning_rate = 0.1
epochs = 1000
num_classes = len(classes)


# Entrenamiento con MSE y Xavier Initialization
model = MLP(input_size, hidden_size, num_classes)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train_loss, val_loss, val_accur, accur, outputs, y_pred, y_true = train_model(
    model, train_loader, val_loader, criterion, optimizer, epochs=epochs)

train_graphs(train_loss, val_loss, val_accur)


# Generar el reporte
report = classification_report(y_true, y_pred, target_names=classes)

# Escribir el reporte en un archivo de texto
output_path = "/mse_classification_report.txt"  # Nombre del archivo

with open(output_path, "w") as f:
    f.write(report)

print(report)
print(f"Reporte exportado exitosamente a {output_path}")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap="viridis")
plt.title("Matriz de Confusión")
plt.show()
plt.close()  # Cerrar la figura actual

# Binarizar etiquetas
y_true_binarized = label_binarize(y_true, classes=range(len(classes)))
outputs_prob = F.softmax(torch.tensor(outputs), dim=1).numpy()

# Calcular curvas ROC para cada clase
fpr = {}
tpr = {}
roc_auc = {}

for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], outputs_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Graficar
import matplotlib.pyplot as plt

for i in range(len(classes)):
    plt.plot(fpr[i], tpr[i], label=f'Clase {classes[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Línea diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curvas ROC')
plt.legend(loc='best')
#plt.show()
plt.savefig('/roc_curve_np_mse.png', dpi=300) # Guardar en alta resolución
plt.close()  # Cerrar la figura actual

# Verificar las dimensiones de las probabilidades y etiquetas
assert outputs_prob.shape[0] == y_true_binarized.shape[0], "El número de muestras no coincide."
assert outputs_prob.shape[1] == y_true_binarized.shape[1], "El número de clases no coincide."

# Crear un subplot para graficar las curvas DET
plt.figure(figsize=(10, 7))

for i in range(len(classes)):  # Iterar sobre cada clase
    # Graficar la curva DET para la clase i
    display = DetCurveDisplay.from_predictions(
        y_true_binarized[:, i],         # Etiquetas binarizadas para la clase i
        outputs_prob[:, i],             # Probabilidades predichas para la clase i
        name=f"Clase {classes[i]}"      # Nombre de la clase
    )
    display.plot(ax=plt.gca())          # Agregar al mismo gráfico

# Configurar el gráfico
plt.title("Curvas DET por clase")
plt.grid(linestyle="--")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Falsos Negativos (FNR)")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('/dev_curve_np_mse.png', dpi=300) # Guardar en alta resolución
plt.close()  # Cerrar la figura actual

# Mostrar el gráfico
#plt.show()


# Comentario Final
print("Ha terminado el proceso de mlp con Mean Square Error")
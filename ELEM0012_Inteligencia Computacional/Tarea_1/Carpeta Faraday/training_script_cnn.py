import os
import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys

print("Iniciando carga de datos...")
sys.stdout.flush()

# Configuración para pruebas locales
data_local = False  # Cambiar a False para usar todo el dataset

# Verificar procesadores disponibles y configurarlos
print(f"Número de procesadores disponibles: {torch.get_num_threads()}")
torch.set_num_threads(8)
print(f"Número de procesadores configurados: {torch.get_num_threads()}")

print("Iniciando carga de datos...")
sys.stdout.flush()

# Cargar datos desde archivo NDJSON
# data = ndjson.load(open('/Users/felipeespinoza/Documents/00 Personal Drive/04 DIASMA/Inteligencia Computacional/ENTREGA1/cnn_training_project/combined_quickdraw.ndjson'))
output_file = './combined_quickdraw.ndjson'
with open(output_file, 'r') as f:
    data = [json.loads(line) for line in f]

if data_local:
    data = random.sample(data, 1000)  # Seleccionar 1000 muestras aleatorias para pruebas locales
    print(f"Usando {len(data)} muestras seleccionadas al azar para pruebas locales.")

def image_ndjson2numpy(data, img_size=51, size_line=10, only_recognized=True, debug=False):
    import cv2  # Importar OpenCV para manejo de imágenes
    
    # Validación del flag de reconocimiento
    flagRecog = lambda x: x if only_recognized else True
    
    # Configuración del tamaño de las imágenes
    imgSize = (img_size, img_size) if isinstance(img_size, int) else img_size
    
    images = []  # Lista para almacenar imágenes procesadas
    labels = []  # Lista para almacenar etiquetas asociadas

    for idx, xx in enumerate(data):
        try:
            # Validar que la entrada tenga un dibujo válido
            if not isinstance(xx.get('drawing'), list) or len(xx['drawing']) == 0:
                if debug:
                    print(f"Entrada {idx} inválida: falta 'drawing' o no está reconocida.")
                continue
            
            # Validar reconocimiento si es requerido
            if only_recognized and not xx.get('recognized', False):
                if debug:
                    print(f"Entrada {idx} no reconocida y se omitirá.")
                continue
            
            # Procesar el dibujo en coordenadas
            yy = np.array(xx['drawing'], dtype=object)[:, :2].T
            FF = [
                np.concatenate(yy[:, i], dtype=float).astype(int).reshape(2, -1).T[:, None]
                for i in range(len(xx['drawing']))
            ]
            
            # Calcular el rango del dibujo
            minmaxFF = np.concatenate([np.c_[i.min(0), i.max(0)] for i in FF])
            minFF = minmaxFF[:, :2].min(0, keepdims=True)
            maxFF = minmaxFF[:, 2:].max(0, keepdims=True)
            FF = [i - minFF[None] for i in FF]

            # Crear una imagen en blanco y negro
            img = np.zeros(((maxFF - minFF).tolist()[0][1], (maxFF - minFF).tolist()[0][0]), np.uint8)
            img = cv2.polylines(img, FF, False, 1, size_line)
            
            # Redimensionar al tamaño deseado
            resized_img = cv2.resize(img, imgSize)
            images.append(resized_img)
            labels.append(xx['class'])
        
        except Exception as e:
            # Manejar errores específicos en el procesamiento de cada entrada
            if debug:
                print(f"Error procesando imagen en entrada {idx}: {e}")
    
    # Validar que se hayan generado imágenes
    if len(images) == 0:
        raise ValueError("No se generaron imágenes válidas. Verifica el dataset y los filtros aplicados.")
    
    return np.array(images), labels

# Procesar datos
images, labels = image_ndjson2numpy(data, img_size=51)
print(f"Tamaño de las imágenes: {images[0].shape}")
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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = QuickDrawDataset(train_images, train_labels, transform=transform)
val_dataset = QuickDrawDataset(val_images, val_labels, transform=transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Definir la red CNN
class CNN(nn.Module):
    def __init__(self, filter_size, stride, padding, fc1_size, fc2_size, num_classes):
        super(CNN, self).__init__()

        # Cierre y revision
        print(f"Proceso Iniciado: CNN de {filter_size} kernels, stride de {stride}, padding de {padding} y tamaño de FC {fc1_size}")
        # Capas convolucionales
        self.conv1 = nn.Conv2d(1, 32, kernel_size=filter_size, stride=stride, padding=padding) # (32x51x51)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduce a (32x25x25)
    
        # Capas fully connected
        self.fc1 = nn.Linear(32 * 25 * 25, fc1_size) # Ajuste para 32 canales y dimensiones 25x25
        self.dropout1 = nn.Dropout(0.5)  # Dropout después de la primera
        self.fc2 = nn.Linear(fc1_size, num_classes) # # Última capa para clasificación
        
        # Activaciones
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x))) # Convolución + Pooling
        x = x.view(x.size(0), -1) # Aplanar las dimensiones
        x = self.relu(self.fc1(x)) # Capa fully connected 1
        x = self.dropout1(x)       # Dropout
        x = self.fc2(x) # # Sin Softmax
        return x

# Entrenar la red
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, early_stopping_patience=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando el dispositivo: {device}")

    # Mover el modelo al dispositivo
    model = model.to(device)
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    # Listas para almacenar métricas
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(torch.float32), labels
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()


        # Promedio de pérdida de entrenamiento
        train_loss /= len(train_loader)
        train_losses.append(train_loss / len(train_loader))

        # Evaluar en el conjunto de validación
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(torch.float32), labels
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
        

        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {100 * correct / total:.2f}%")

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
    plt.savefig('/home/fespinoza/proyecto_ic/cnn_results/cnn_results.png', dpi=300) # Guardar en alta resolución
    #plt.show()

    # Descomentar para Ejercicio 3
    #return train_losses, val_losses, val_accuracies  # Devuelve las métricas (Ejercicio 3)

# Configuración de la red
filter_size = 3
epochs = 1000
stride = 1
padding = 1
fc1_size = 64
num_classes = len(classes)

model = CNN(filter_size, stride, padding, fc1_size, fc2_size=0, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # Tasa de aprendizaje 0.001


# Entrenar el modelo
train_model(model, train_loader, val_loader, criterion, optimizer, epochs)

# Función para generar la matriz de confusión
def plot_confusion_matrix(model, val_loader, classes):
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
            
            # Si las etiquetas son one-hot, usa argmax
            if len(labels.shape) > 1:
                all_labels.extend(labels.argmax(dim=1).cpu().numpy())
            else:
                all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="viridis")
    plt.title("Matriz de Confusión")

    # Guardar los gráficos como archivo PNG
    plt.savefig('/home/fespinoza/proyecto_ic/cnn_results/consusion_matrix_cnn.png', dpi=300) # Guardar en alta resolución
    #plt.show()
    #plt.close()  # Cerrar la figura actual

# Llamar a la función
plot_confusion_matrix(model, val_loader, classes)


# Función para generar las curvas ROC
def plot_roc_curve(model, val_loader, classes):
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
    plt.savefig('/home/fespinoza/proyecto_ic/cnn_results/roc_curve_cnn.png', dpi=300) # Guardar en alta resolución
    #plt.show()
    #plt.close()  # Cerrar la figura actual

# Llamar a la función
plot_roc_curve(model, val_loader, classes)

# Función para graficar la curva DET
def plot_det_curve(model, val_loader, classes):
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
    plt.savefig('/home/fespinoza/proyecto_ic/cnn_results/det_curve_cnn.png', dpi=300) # Guardar en alta resolución

    #plt.show()
    #plt.close()  # Cerrar la figura actual

# Llamar a la función
plot_det_curve(model, val_loader, classes)

# Comentario Final
print("Ha terminado el Proceso de CNN")
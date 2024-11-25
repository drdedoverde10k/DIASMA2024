#!/bin/bash

# Nombre del trabajo
#SBATCH --job-name="CNN000"

# Nombre del archivo de salida (logs)
#SBATCH --output=slurm-%j.out

# Partición y configuración de nodos
#SBATCH --partition=intel
#SBATCH -n 12
#SBATCH --ntasks-per-node=12
#SBATCH --mail-user=fespinozav@utem.cl
#SBATCH --mail-type=ALL
#SBATCH --mem=16GB

# Tiempo máximo de ejecución (formato D-HH:MM:SS)
#SBATCH --time=90:00:00  # 90 horas

# Registro del inicio
TimeStart="$(date +"%c")"
echo "Starting of the scipt $TimeStart" > cnn_start_log.txt

# Activar el entorno de Conda
source /home/fespinoza/miniconda3/etc/profile.d/conda.sh
conda activate cnn_training || { echo "Error al activar conda"; exit 1; }

# Forzar salida inmediata
export PYTHONUNBUFFERED=1

# Ejecutar el script Python
python cnn_script.py || { echo "Error en el script Python"; exit 1; }

# Registro del fin
TimeEnd="$(date +"%c")"
echo "Ending script at $TimeEnd" >> cnn_end_log.txt

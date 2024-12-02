#!/bin/bash

# Nombre del trabajo
#SBATCH --job-name="MSE000"

# Nombre del archivo de salida (logs)
#SBATCH --output=slurm-%j.out

# Partición y configuración de nodos
#SBATCH --partition=intel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=fespinozav@utem.cl
#SBATCH --mail-type=ALL
#SBATCH --mem=16GB

# Tiempo máximo de ejecución (formato D-HH:MM:SS)
#SBATCH --time=90:00:00  # 90 horas

# Registro del inicio
TimeStart="$(date +"%c")"
echo "Starting of the scipt $TimeStart" > mse_start_log.txt

# Activar el entorno de Conda
source /home/fespinoza/miniconda3/etc/profile.d/conda.sh

conda activate master_nn_training || { echo "Error al activar conda"; exit 1; }
echo "Entorno Conda activado: $(conda info --envs | grep '*' | awk '{print $1}')"

# Ejecutar el script Python
free -h > memory_mse_start.txt
python mlp_mse.py || { echo "Error en el script Python"; exit 1; }
free -h > memomy_mse_end.txt

# Registro del fin
TimeEnd="$(date +"%c")"
echo "Ending script at $TimeEnd" >> mse_end_log.txt
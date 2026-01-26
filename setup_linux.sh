#!/bin/bash
set -e

echo "🔧 Configurando entorno para Darhisper en Linux..."

# 1. Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "📦 Creando virtual environment (venv)..."
    python3 -m venv venv
else
    echo "✅ Virtual environment ya existe."
fi

# 2. Activar entorno
source venv/bin/activate

# 3. Actualizar pip
echo "⬆️  Actualizando pip..."
pip install --upgrade pip

# 4. Instalar PyTorch con soporte CUDA (Recomendado para RTX 4060)
# Intentamos instalar primero torch compatible con CUDA para asegurar que NeMo lo use
echo "🔥 Instalando PyTorch con soporte CUDA..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Instalar dependencias del proyecto
echo "📥 Instalando resto de dependencias desde requirements.txt..."
# Cython es necesario para compilar algunas partes de NeMo
pip install Cython
pip install -r requirements.txt

echo "==================================================="
echo "✅ Instalación completada con éxito."
echo "🚀 Para iniciar la app, ejecuta: ./start.sh"
echo "==================================================="

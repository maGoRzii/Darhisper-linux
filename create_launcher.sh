#!/bin/bash
# Obtener directorio actual
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ICON_PATH="$DIR/icon.png"
EXEC_PATH="$DIR/start.sh"
DESKTOP_FILE="$HOME/.local/share/applications/darhisper.desktop"

# Crear archivo .desktop
cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Name=Darhisper
Comment=Asistente de transcripción de voz con IA
Exec=$EXEC_PATH
Icon=$ICON_PATH
Terminal=false
Type=Application
Categories=Utility;Audio;
StartupNotify=false
EOF

# Permisos
chmod +x "$DESKTOP_FILE"

# Notificar
echo "✅ Lanzador creado exitosamente!"
echo "📍 Ubicación: $DESKTOP_FILE"
echo "🔍 Ahora puedes buscar 'Darhisper' en tu menú de inicio."

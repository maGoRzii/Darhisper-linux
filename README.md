# Ghost Eagle 🦅🎙️

Aplicación de barra de menú para macOS que transcribe voz a texto localmente y a ultra-velocidad usando `mlx-whisper` (optimizado para Apple Silicon).

## Características

*   🚀 **Ultra-rápido**: Transcripción local en tiempo real usando MLX.
*   ⌨️ **Atajos Personalizables**: Elige o graba tu propia combinación de teclas.
*   📋 **Pegado Automático**: Escribe automáticamente el texto transcrito donde tengas el cursor.
*   🔊 **Feedback Sonoro**: Sonidos de inicio y fin de grabación.
*   🔒 **Privacidad Total**: Todo se procesa en tu Mac, nada sale a internet.

## Requisitos

*   **macOS** (Optimizado para Apple Silicon M1/M2/M3).
*   **Python 3.10+** instalado.
*   **FFmpeg** (Opcional, pero recomendado para manejo de audio). Puedes instalarlo con `brew install ffmpeg`.

## Instalación

1.  **Clonar el repositorio** (o descargar el código):
    ```bash
    git clone <tu-repo-url>
    cd ghost-eagle
    ```

2.  **Crear un entorno virtual** (Recomendado):
    ```bash
    python3 -m venv venv
    ```

3.  **Activar el entorno e instalar dependencias**:
    ```bash
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Configuración de Permisos (¡Importante! ⚠️)

Para que la aplicación pueda detectar atajos globales y pegar texto, necesitas dar permisos en **Ajustes del Sistema > Privacidad y Seguridad**:

1.  **Accesibilidad**: Permite detectar cuando pulsas el atajo de teclado.
    *   Añade tu Terminal (ej. iTerm, Terminal.app) o editor (VS Code) a la lista.
2.  **Monitorización de entrada** (Input Monitoring): Necesario para escuchar atajos globales.
3.  **Micrófono**: Te pedirá permiso la primera vez que intentes grabar.

> Si la aplicación arranca pero no graba o no pega, revisa estos permisos. A veces es necesario eliminar la entrada (-) y volverla a añadir (+).

## Uso

1.  **Iniciar la aplicación**:
    Simplemente ejecuta el script de inicio:
    ```bash
    ./start.sh
    ```
    Verás un icono de micrófono 🎙️ en la barra de menú superior.

2.  **Transcribir**:
    *   Mantén presionado el atajo (Por defecto **F5** o **Option Derecho**).
    *   Escucharás un *beep* agudo. Habla.
    *   Suelta la tecla. Escucharás un *beep* grave.
    *   El texto aparecerá mágicamente donde tengas el cursor. ✨

3.  **Configuración**:
    *   Haz clic en el icono 🎙️ para cambiar el modelo de Whisper (Tiny, Base, Small).
    *   Ve a **Shortcut > Record New Shortcut...** para grabar tu propia combinación de teclas.

## Solución de Problemas

*   **Error de Permisos (1002)**: Significa que la app no puede pegar el texto. Asegúrate de dar permisos de "Accesibilidad" a la terminal que estés usando.
*   **No se escucha nada**: Verifica que el volumen de tu Mac no esté en silencio para escuchar los beeps de feedback.

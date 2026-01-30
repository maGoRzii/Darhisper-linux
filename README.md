# Darhisper 🦅🎙️

> **Tu asistente de voz definitivo para transcripción instantánea en Linux.**
> *Optimizado para NVIDIA RTX (NeMo Parakeet).*

![Linux](https://img.shields.io/badge/Linux-NVIDIA_RTX-green?logo=linux&logoColor=white) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white) ![NVIDIA NeMo](https://img.shields.io/badge/Powered_by-NVIDIA_NeMo-76B900?logo=nvidia&logoColor=white)

**Darhisper** es una herramienta de productividad residente en tu bandeja del sistema (System Tray) que permite dictar texto en cualquier aplicación. Utiliza la potencia de tu GPU NVIDIA para transcribir voz a texto localmente con velocidad extrema.

---

## ✨ Características Principales

*   **⚡️ Velocidad Ultrarrapida (Local)**: Utiliza el motor **NVIDIA NeMo** con el modelo **Parakeet-TDT (0.6B)**, ofreciendo transcripción en tiempo real y privacidad total.
*   **☁️ Potencia en la Nube (Opcional)**: Integración con **Gemini 3 Flash Preview** para "Smart Prompts" (corrección de estilo, resúmenes, emails, etc.).
*   **🎨 Diseño Elegante**: Feedback visual moderno (Overlay flotante) y sonoro (Beeps de confirmación).
*   **⌨️ Push-to-Talk**: Mantén presionado `Control Derecho` (configurable) y habla. El texto se escribe mágicamente al soltar.
*   **📁 Transcripción de Archivos**: Sube audios (mp3, wav, m4a, ogg, flac) y recibe el texto en la interfaz.
*   **🪟 Ventana Redimensionable**: La interfaz se puede cambiar de tamaño desde las esquinas sin romper el layout.
*   **🐧 Linux Nativo**: Integración perfecta con escritorios Linux (Gnome, Cinnamon, KDE).

---

## 🖥️ Requisitos del Sistema

*   **Sistema Operativo**: Linux (Probado en Linux Mint / Ubuntu 22.04+).
*   **GPU**: NVIDIA RTX (RTX 3060/4060 o superior recomendada) con drivers propietarios instalados.
*   **Audio**: Servidor de audio PulseAudio o PipeWire funcionando.
*   **Dependencias**: Python 3.10+, `xclip` o `xsel` (opcional, para portapapeles).

---

## 🚀 Instalación

1.  **Clonar el repositorio**:
    ```bash
    git clone https://github.com/maGoRzii/Darhisper.git
    cd Darhisper
    ```

2.  **Instalar dependencias del sistema**:
    ```bash
    sudo apt install python3-pyaudio portaudio19-dev python3-tk
    ```

3.  **Configurar entorno automáticamente**:
    Ejecuta el script de instalación. Esto creará un entorno virtual aislado y descargará PyTorch con soporte CUDA.
    ```bash
    chmod +x setup_linux.sh start.sh
    ./setup_linux.sh
    ```

4.  **Iniciar la aplicación**:
    ```bash
    ./start.sh
    ```

5.  **Crear acceso directo (Opcional)**:
    Para abrir la app desde el menú de aplicaciones sin terminal:
    ```bash
    ./create_launcher.sh
    ```

---

## 📖 Guía de Uso

### Flujo de Trabajo
1.  Haz clic donde quieras escribir (Terminal, Slack, Obsidian, Navegador...).
2.  **Mantén presionado** la tecla `Control Derecho`.
3.  Escucharás un **Beep agudo** 🎵 y verás una **onda de voz** flotante. Habla con naturalidad.
4.  Suelta la tecla. Tras un **Beep grave** 🎵, el texto aparecerá escrito automáticamente.

### Configuración (Bandeja del Sistema)

Haz clic en el icono 🎙️ de la barra de tareas (Tray Icon) para:

#### 🧠 Modos Inteligentes (Smart Prompts)
*(Requiere configurar API Key de Gemini)*
*   **Transcripción Literal**: Escribe tal cual lo que dices (Modo Offline predeterminado).
*   **Lista de Tareas**: Formatea lo dictado como viñetas de una lista.
*   **Email Profesional**: Reescribe lo dictado con tono formal y estructura de correo.
*   **Modo Excel**: Formatea números y datos para hojas de cálculo.

### Transcripción de Archivos
1.  Abre la interfaz desde el icono de bandeja y selecciona **"Elegir Archivo..."**.
2.  Elige el audio y pulsa **"COMENZAR TRANSCRIPCIÓN"**.
3.  El progreso se muestra en la barra y el resultado aparece en el área de texto.

**Modelo de archivo (API)**: Solo se usa **Gemini 3 Flash Preview**. No hay otros modelos configurables.

#### 🔐 Configurar API Keys
*   Ve a la opción `Configurar API Key` para introducir tu clave de Google Gemini si deseas usar los modos inteligentes.
*   **Nota**: La transcripción básica (Literal) es 100% local y **NO requiere clave ni internet**.

---

## ❓ Solución de Problemas

| Problema | Solución |
| :--- | :--- |
| **Error `externally-managed-environment`** | Usa siempre `./start.sh` para ejecutar la app. No uses `python main.py` directamente fuera del entorno. |
| **La primera vez tarda mucho** | La primera ejecución descarga el modelo NVIDIA Parakeet (~1.1GB). Ten paciencia, las siguientes serán instantáneas. |
| **Crash al iniciar** | Verifica que tienes los drivers de NVIDIA cargados correctamente ejecutando `nvidia-smi` en la terminal. |

---

## 📄 Licencia
Este proyecto es de código abierto. ¡Contribuciones bienvenidas!

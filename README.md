# Darhisper 🦅🎙️

> **Tu asistente de voz definitivo para macOS. Transcripción instantánea, local y privada.**

![macOS](https://img.shields.io/badge/macOS-Apple_Silicon-white?logo=apple&logoColor=black) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white) ![MLX](https://img.shields.io/badge/Powered_by-Apple_MLX-yellow)

**Darhisper** es una herramienta de productividad residente en la barra de menú diseñada exclusivamente para **macOS (Apple Silicon)**. Permite dictar texto en cualquier aplicación con una velocidad y precisión sorprendentes, utilizando la potencia del motor neuronal de tu Mac o la flexibilidad de la nube.

---

## ✨ Características Principales

*   **⚡️ Velocidad Ultrarrapida (Local)**: Utiliza `mlx-whisper` optimizado específicamente para chips Apple Silicon (M1/M2/M3), ofreciendo transcripciones casi instantáneas sin enviar datos a internet.
*   **☁️ Potencia en la Nube (Opcional)**: Integración nativa con **Google Gemini 3.0 Flash** para cuando necesitas una "inteligencia" superior en la transcripción, capaz de entender contextos complejos, instrucciones y puntuación perfecta.
*   **🎨 Diseño Elegante**: Feedback visual moderno con una interfaz de ondas de voz animadas que flotan sobre tu pantalla mientras dictas.
*   **⌨️ Escribe Donde Sea**: Funciona globalmente. Simplemente coloca el cursor, mantén presionado tu atajo y habla. El texto se escribe mágicamente en la aplicación activa.
*   **⚙️ Totalmente Configurable**:
    *   Cambia de modelos de IA al vuelo.
    *   Graba tus propios atajos de teclado personalizados.
    *   Gestiona tus claves de API de forma segura.

---

## 🖥️ Requisitos del Sistema

Para garantizar el máximo rendimiento, Darhisper tiene requisitos específicos:

*   **Hardware**: Mac con chip **Apple Silicon** (M1, M1 Pro/Max/Ultra, M2, M3, etc.).
    *   *Nota: No es compatible con Macs basados en Intel debido a la dependencia de MLX.*
*   **Sistema Operativo**: macOS 12.0 (Monterey) o superior.
*   **Permisos**: Requiere acceso a **Micrófono** y **Accesibilidad** (para la inserción de texto).

---

## 🚀 Instalación y Uso

### Opción A: Para Usuarios (Aplicación Compilada)

1.  **Descarga**: Obtén la última versión de `Darhisper.app` (desde la carpeta `dist` si lo has compilado tú mismo).
2.  **Instala**: Arrastra la app a tu carpeta de **Aplicaciones**.
3.  **Primer Lanzamiento**:
    *   Al abrir la app, verás un icono 🎙️ en la barra de menú.
    *   **Importante**: Si macOS indica que la app "está dañada" o "no se puede abrir", ejecuta este comando en la Terminal para firmarla localmente:
        ```bash
        xattr -cr /Applications/Darhisper.app
        ```
4.  **Concede Permisos**: La primera vez que intentes usarla, macOS te pedirá permisos. Acepta:
    *   🎤 Micrófono.
    *   ⌨️ Accesibilidad/Eventos del sistema (para pegar el texto).

### Opción B: Para Desarrolladores (Código Fuente)

Si prefieres ejecutarlo desde el código o contribuir:

1.  **Clonar el repositorio**:
    ```bash
    git clone https://github.com/maGoRzii/Darhisper.git
    cd Darhisper
    ```

2.  **Configurar entorno**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
    *Es posible que necesites instalar `portaudio` para el audio:* `brew install portaudio`

3.  **Ejecutar**:
    ```bash
    ./start.sh
    ```

---

## 📖 Guía de Uso

### Flujo de Trabajo Básico
1.  Haz clic donde quieras escribir (Slack, Notion, VS Code, etc.).
2.  **Mantén presionado** el atajo de teclado (Por defecto: `F5` o `Opción Derecha`).
3.  Espera el **Beep** y habla cuando veas la **onda de voz** en pantalla.
4.  Suelta la tecla al terminar. El texto aparecerá automáticamente.

### Configuración Avanzada

Haz clic en el icono 🎙️ de la barra de menú para acceder a las opciones:

#### 🧠 Selección de Modelos (Model)
*   **Modelos Locales (MLX)**:
    *   *Tiny/Base/Small*: Extremadamente rápidos, bajo consumo de batería.
    *   *Large-v3-Turbo*: Mayor precisión, ideal para dictados largos y complejos.
*   **Modelos Cloud (API)**:
    *   *Gemini Flash*: Requiere API Key. Ofrece una "comprensión" superior, capaz de seguir instrucciones como "pon esto en una lista" o corregir gramática al vuelo.

#### 🎭 Selección de Modos (Smart Prompts)
*(Disponible solo con modelos Gemini)*

Personaliza cómo la IA procesa tu voz seleccionando un modo en el menú "Mode":
*   **Transcripción Literal**: Escribe exactamente lo que dices, letra por letra.
*   **Lista de Tareas (To-Do)**: Transforma tus divagaciones en una lista limpia y accionable de tareas.
*   **Email Profesional**: Convierte un dictado informal en un correo electrónico pulido, formal y listo para enviar.
*   **Modo Excel/Datos**: Formatea números y listas para que se peguen perfectamente en celdas de hojas de cálculo.

#### ⌨️ Atajos (Shortcut)
*   Elige entre presets comunes (`F5`, `Cmd+Opt+R`).
*   Selecciona **"Record New Shortcut..."** para grabar tu propia combinación de teclas favorita.

#### 🔐 Seguridad y API Keys
*   Para usar Google Gemini, ve a `Model` -> `Edit Gemini API Key`.
*   Tu clave se guarda localmente en `~/.darhisper_config.json` y nunca se comparte.
*   Los modelos locales (`mlx`) funcionan 100% offline y son totalmente privados.

---

## ❓ Solución de Problemas

| Problema | Solución |
| :--- | :--- |
| **No escribe nada** | Verifica que has dado permisos de **Accesibilidad** en *Preferencias del Sistema -> Privacidad y Seguridad*. |
| **Error al iniciar** | Asegúrate de tener un Mac con **Apple Silicon**. Borra la carpeta `~/.darhisper_config.json` para resetear la config. |
| **La primera transcripción tarda** | Es normal. La primera vez, la app descarga los modelos de IA (1-3 GB). Las siguientes serán instantáneas. |

---

## 📄 Licencia

Este proyecto es de código abierto. Siéntete libre de modificarlo, mejorarlo y compartirlo.

---
*Hecho para maximizar tu productividad.*

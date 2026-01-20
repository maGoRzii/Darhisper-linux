import sys
import threading
import time
import queue
import numpy as np
import sounddevice as sd
import mlx_whisper
import rumps
import pyperclip
import pyautogui
from pynput import keyboard
import subprocess
import os
import json
import logging
import math
import objc
from AppKit import NSWindow, NSView, NSColor, NSMakeRect, NSBorderlessWindowMask, NSFloatingWindowLevel, NSTimer, NSBezierPath
from Quartz import CGRectMake

# Setup debug logging
logging.basicConfig(
    filename='/tmp/ghosteagle_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Ghost Eagle starting up...")


# Configuration
CONFIG_FILE = os.path.expanduser("~/.ghosteagle_config.json")
SAMPLE_RATE = 16000
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

class WaveView(NSView):
    """Vista personalizada que dibuja las ondas de voz"""
    def init(self):
        self = objc.super(WaveView, self).init()
        if self is None:
            return None
        self.wave_phase = 0.0
        self.timer = None
        self.drag_start_point = None
        self.window_origin = None
        self.is_recording = False  # Estado de grabación
        return self
    
    def drawRect_(self, rect):
        """Dibuja una línea fina o ondas según el estado"""
        if not self.is_recording:
            # Modo inactivo: mostrar solo una línea fina
            # Sin fondo, solo la línea con borde
            
            # Línea en el centro
            width = self.bounds().size.width
            height = self.bounds().size.height
            center_y = height / 2
            
            # Dibujar línea horizontal con efecto de borde
            line_height = 3
            line_y = center_y - (line_height / 2)
            line_rect = NSMakeRect(1, line_y, width - 2, line_height)
            
            # Color gris oscuro para la línea (interior)
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.25, 0.25, 0.28, 1.0  # Gris oscuro
            ).setFill()
            
            line_path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                line_rect, 1.5, 1.5
            )
            line_path.fill()
            
            # Dibujar borde más claro alrededor
            NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.4, 0.4, 0.45, 0.8  # Gris más claro para el borde
            ).setStroke()
            line_path.setLineWidth_(0.5)
            line_path.stroke()
        else:
            # Modo grabando: líneas entrelazadas estilo IA moderno
            # Fondo casi invisible
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.1, 0.1, 0.1, 0.2).setFill()
            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                self.bounds(), 6, 6
            )
            path.fill()
            
            width = self.bounds().size.width
            height = self.bounds().size.height
            center_y = height / 2
            
            # Dibujar 3 líneas entrelazadas con diferentes fases y velocidades
            num_lines = 3
            
            for i in range(num_lines):
                path = NSBezierPath.bezierPath()
                
                # Configuracion de la onda
                time = self.wave_phase
                frequency = 0.05 + (i * 0.01) # Frecuencia espacial
                speed = 1.0 + (i * 0.5)       # Velocidad de oscilación
                amplitude = 15 - (i * 2)      # Altura de la onda
                phase_offset = i * (math.pi / 2) # Desfase entre líneas
                
                # Iniciar el camino
                start_y = center_y + math.sin(time * speed + phase_offset) * amplitude
                path.moveToPoint_((0, start_y))
                
                # Dibujar la curva punto por punto
                for x in range(1, int(width), 2): # Paso de 2px para eficiencia
                    # Fórmula compuesta para movimiento orgánico
                    # Seno base + variación lenta
                    wave = math.sin(x * frequency + time * speed + phase_offset)
                    # Modulación de amplitud para que los bordes se atenúen (efecto lente)
                    envelope = math.sin((x / width) * math.pi) 
                    
                    y = center_y + (wave * amplitude * envelope)
                    path.lineToPoint_((x, y))
                
                # Configurar estilo de línea
                path.setLineWidth_(2.0)
                
                # Colores grises/plateados con transparencia
                # La línea más activa es más clara
                if i == 0: # Principal
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(0.9, 0.9, 0.95, 0.9).setStroke()
                elif i == 1: # Secundaria
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(0.6, 0.6, 0.65, 0.7).setStroke()
                else: # Terciaria
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(0.4, 0.4, 0.45, 0.5).setStroke()
                    
                path.stroke()
    
    def setRecording_(self, recording):
        """Establece el estado de grabación"""
        self.is_recording = recording
        self.setNeedsDisplay_(True)
    
    def mouseDown_(self, event):
        """Evento cuando se presiona el mouse - inicia el arrastre"""
        self.drag_start_point = event.locationInWindow()
        self.window_origin = self.window().frame().origin
    
    def mouseDragged_(self, event):
        """Evento cuando se arrastra el mouse - mueve la ventana"""
        if self.drag_start_point is None or self.window_origin is None:
            return
        
        current_location = event.locationInWindow()
        from AppKit import NSScreen
        
        # Convertir a coordenadas de pantalla
        window_frame = self.window().frame()
        screen_location = self.window().convertBaseToScreen_(current_location)
        
        # Calcular nuevo origen
        new_origin_x = self.window_origin.x + (current_location.x - self.drag_start_point.x)
        new_origin_y = self.window_origin.y + (current_location.y - self.drag_start_point.y)
        
        # Mover la ventana sin restricciones
        self.window().setFrameOrigin_((new_origin_x, new_origin_y))
    
    def mouseUp_(self, event):
        """Evento cuando se suelta el mouse - finaliza el arrastre"""
        self.drag_start_point = None
        self.window_origin = None
    
    def startAnimation(self):
        """Inicia la animación"""
        if self.timer is None:
            self.timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.05, self, 'updateWave:', None, True
            )
    
    def stopAnimation(self):
        """Detiene la animación"""
        if self.timer is not None:
            self.timer.invalidate()
            self.timer = None
    
    def updateWave_(self, timer):
        """Actualiza la fase de la onda y redibuja"""
        self.wave_phase += 0.2
        self.setNeedsDisplay_(True)

class VoiceWaveWindow:
    """Ventana flotante con ondas de voz animadas"""
    def __init__(self, app):
        self.app = app  # Referencia a la aplicación rumps
        self.window = None
        self.wave_view = None
        self._show_pending = False
        self._hide_pending = False
        self._recording_pending = False
        self._recording_state = False
        
        # Configurar timer para manejar operaciones de UI en el hilo principal
        self.ui_timer = rumps.Timer(self._process_ui_operations, 0.1)
        self.ui_timer.start()
        
    def _process_ui_operations(self, sender):
        """Procesa operaciones de UI pendientes en el hilo principal"""
        try:
            # Crear ventana si es necesario (lazy creation)
            if self.window is None:
                self._create_window()
                # Mostrar la ventana inmediatamente después de crearla
                if self.window:
                    self.window.orderFront_(None)
                    self.window.makeKeyAndOrderFront_(None)
                    print("Línea indicadora mostrada")
            
            # Cambiar estado de grabación si está pendiente
            if self._recording_pending:
                if self._recording_state:
                    # Expandir ventana para ondas
                    self._expand_window()
                    if self.wave_view:
                        self.wave_view.setRecording_(True)
                        self.wave_view.startAnimation()
                    print("Ventana expandida - ondas activas")
                else:
                    # Contraer ventana a línea
                    self._contract_window()
                    if self.wave_view:
                        self.wave_view.setRecording_(False)
                        self.wave_view.stopAnimation()
                    print("Ventana contraída - modo línea")
                self._recording_pending = False
                
        except Exception as e:
            print(f"Error en _process_ui_operations: {e}")
            import traceback
            traceback.print_exc()
        
    def _create_window(self):
        """Crea la ventana flotante como línea fina (llamado desde el hilo principal)"""
        try:
            # Crear ventana pequeña (línea fina más pequeña)
            self.line_width = 50
            self.line_height = 5
            rect = NSMakeRect(0, 0, self.line_width, self.line_height)
            self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                rect,
                NSBorderlessWindowMask,
                2,  # NSBackingStoreBuffered
                False
            )
            
            # Configurar ventana
            self.window.setLevel_(NSFloatingWindowLevel)
            self.window.setOpaque_(False)
            self.window.setBackgroundColor_(NSColor.clearColor())
            
            # Hacer que aparezca en todos los espacios/escritorios
            from AppKit import NSWindowCollectionBehaviorCanJoinAllSpaces
            self.window.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces)
            
            # Posicionar en la pantalla (centro superior, pegado a la barra de herramientas)
            from AppKit import NSScreen
            screen = NSScreen.mainScreen()
            screen_frame = screen.frame()
            # Centro horizontal
            x = (screen_frame.size.width - self.line_width) / 2
            # Arriba del todo (en macOS, y alto = arriba)
            y = screen_frame.size.height - self.line_height - 5  # 5px desde el top
            self.window.setFrameOrigin_((x, y))
            
            # Crear vista de ondas
            self.wave_view = WaveView.alloc().init()
            self.wave_view.setFrame_(rect)
            
            # Agregar vista a la ventana
            self.window.setContentView_(self.wave_view)
            
            print("Línea indicadora creada exitosamente")
        except Exception as e:
            print(f"Error creando ventana: {e}")
            import traceback
            traceback.print_exc()
    
    def _expand_window(self):
        """Expande la ventana para mostrar ondas (hacia abajo desde la línea)"""
        if not self.window:
            return
        
        # Guardar posición actual
        current_pos = self.window.frame().origin
        
        # Nuevo tamaño expandido
        new_width = 100
        new_height = 50
        
        # Calcular nueva posición: centrado horizontalmente, mantener top fijo
        new_x = current_pos.x - (new_width - self.line_width) / 2
        # En macOS, para mantener el top fijo al expandir hacia abajo,
        # debemos RESTAR la diferencia de altura del Y
        new_y = current_pos.y - (new_height - self.line_height)
        
        # Redimensionar ventana
        new_frame = NSMakeRect(new_x, new_y, new_width, new_height)
        self.window.setFrame_display_(new_frame, True)
        
        # Redimensionar vista
        if self.wave_view:
            self.wave_view.setFrame_(NSMakeRect(0, 0, new_width, new_height))
    
    def _contract_window(self):
        """Contrae la ventana a línea fina"""
        if not self.window:
            return
        
        # Guardar posición actual
        current_frame = self.window.frame()
        center_x = current_frame.origin.x + current_frame.size.width / 2
        top_y = current_frame.origin.y  # Mantener posición superior
        
        # Calcular nueva posición: centrado horizontalmente, mantener top fijo
        new_x = center_x - self.line_width / 2
        # Para volver a la línea, debemos SUMAR la diferencia de altura
        # para que el top vuelva a su posición original
        new_y = top_y + (current_frame.size.height - self.line_height)
        
        # Redimensionar ventana a línea
        new_frame = NSMakeRect(new_x, new_y, self.line_width, self.line_height)
        self.window.setFrame_display_(new_frame, True)
        
        # Redimensionar vista
        if self.wave_view:
            self.wave_view.setFrame_(NSMakeRect(0, 0, self.line_width, self.line_height))
    
    def show(self):
        """Muestra ondas (inicia grabación)"""
        self._recording_state = True
        self._recording_pending = True
        
    def hide(self):
        """Oculta ondas (detiene grabación, pero mantiene línea visible)"""
        self._recording_state = False
        self._recording_pending = True


class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.audio_queue = queue.Queue()
        self.stream = None
        self.audio_data = []

    def callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        if self.recording:
            self.audio_queue.put(indata.copy())

    def start(self):
        self.recording = True
        self.audio_data = []
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, 
            channels=1, 
            callback=self.callback
        )
        self.stream.start()

    def stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Collect all data from queue
        while not self.audio_queue.empty():
            self.audio_data.append(self.audio_queue.get())
            
        if not self.audio_data:
            return None
            
        return np.concatenate(self.audio_data, axis=0)

class VoiceTranscriberApp(rumps.App):
    def __init__(self):
        super(VoiceTranscriberApp, self).__init__("🎙️")
        self.recorder = AudioRecorder()
        self.is_transcribing = False
        self.is_learning_hotkey = False
        self.learning_keys = set()
        
        # Inicializar ventana de ondas (ahora con PyObjC, no necesita hilo separado)
        self.wave_window = VoiceWaveWindow(self)
        
        # Load configuration
        self.config = self.load_config()
        self.model_path = self.config.get("model", DEFAULT_MODEL)
        self.hotkey_check = self.deserialize_hotkey(self.config.get("hotkey", ["Key.f5"]))
        
        # Pre-generate feedback sounds
        self.start_sound = self.generate_beep(880, 0.1)
        self.stop_sound = self.generate_beep(440, 0.1)
        
        # Menu items
        self.menu = [
            rumps.MenuItem("Model", icon=None, dimensions=(1, 1)),
            rumps.MenuItem("Shortcut", icon=None, dimensions=(1, 1)),
            rumps.separator
        ]
        
        # Hotkey listener initialization
        self.current_keys = set()
        
        # Submenus
        self.setup_model_menu()
        self.setup_shortcut_menu()
        
        self.setup_hotkey_listener()

    def generate_beep(self, frequency, duration=0.1, fs=SAMPLE_RATE):
        t = np.linspace(0, duration, int(fs * duration), False)
        tone = np.sin(2 * np.pi * frequency * t) * 0.1
        # Fade in/out
        envelope = np.concatenate([
            np.linspace(0, 1, int(fs * 0.01)),
            np.ones(int(fs * (duration - 0.02))),
            np.linspace(1, 0, int(fs * 0.01))
        ])
        return (tone * envelope).astype(np.float32)

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def save_config(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config, f)

    def serialize_hotkey(self, keys):
        serialized = []
        for k in keys:
            if isinstance(k, keyboard.Key):
                serialized.append(f"Key.{k.name}")
            elif isinstance(k, keyboard.KeyCode):
                # Use char if available, otherwise virtual key code
                if k.char:
                    serialized.append(f"Char.{k.char}")
                else:
                    serialized.append(f"Vk.{k.vk}")
        return serialized

    def deserialize_hotkey(self, key_strings):
        keys = set()
        for s in key_strings:
            try:
                if s.startswith("Key."):
                    keys.add(getattr(keyboard.Key, s.split(".", 1)[1]))
                elif s.startswith("Char."):
                    keys.add(keyboard.KeyCode.from_char(s.split(".", 1)[1]))
                elif s.startswith("Vk."):
                    keys.add(keyboard.KeyCode.from_vk(int(s.split(".", 1)[1])))
            except:
                pass # Ignore malformed keys
        return keys

    def setup_model_menu(self):
        models = [
            "mlx-community/whisper-tiny-mlx",
            "mlx-community/whisper-base-mlx",
            "mlx-community/whisper-small-mlx",
            "mlx-community/whisper-large-v3-turbo",
            "mlx-community/whisper-large-v3-turbo-q4"
        ]
        img_model_menu = rumps.MenuItem("Select Model")
        for m in models:
            item = rumps.MenuItem(m, callback=self.change_model)
            if m == self.model_path:
                item.state = 1
            img_model_menu.add(item)
        self.menu["Model"].add(img_model_menu)

    def change_model(self, sender):
        for item in self.menu["Model"]["Select Model"].values():
            item.state = 0
        sender.state = 1
        self.model_path = sender.title
        self.config["model"] = self.model_path
        self.save_config()
        print(f"Model switched to: {self.model_path}")

    def setup_shortcut_menu(self):
        # Determine text for current shortcut
        current_serialized = self.config.get("hotkey", ["Key.f5"])
        
        presets = {
            "F5": {keyboard.Key.f5},
            "Cmd+Opt+R": {keyboard.Key.cmd, keyboard.Key.alt, keyboard.KeyCode.from_char('r')},
            "Right Option": {keyboard.Key.alt_r}
        }
        
        shortcut_menu = rumps.MenuItem("Select Shortcut")
        
        # Add presets
        for name, keys in presets.items():
            item = rumps.MenuItem(name, callback=self.change_shortcut_preset)
            if keys == self.hotkey_check:
                item.state = 1
            shortcut_menu.add(item)
            
        shortcut_menu.add(rumps.separator)
        shortcut_menu.add(rumps.MenuItem("Record New Shortcut...", callback=self.start_recording_hotkey))
        
        self.menu["Shortcut"].add(shortcut_menu)

    def change_shortcut_preset(self, sender):
        presets = {
            "F5": {keyboard.Key.f5},
            "Cmd+Opt+R": {keyboard.Key.cmd, keyboard.Key.alt, keyboard.KeyCode.from_char('r')},
            "Right Option": {keyboard.Key.alt_r}
        }
        
        for item in self.menu["Shortcut"]["Select Shortcut"].values():
            item.state = 0
        sender.state = 1
        
        if sender.title in presets:
            self.hotkey_check = presets[sender.title]
            self.config["hotkey"] = self.serialize_hotkey(self.hotkey_check)
            self.save_config()
            print(f"Shortcut changed to: {sender.title}")

    def start_recording_hotkey(self, sender):
        self.is_learning_hotkey = True
        self.learning_keys = set()
        self.title = "⌨️"
        rumps.notification("Ghost Eagle", "Recording Shortcut", "Press your desired key combination now.")

    def setup_hotkey_listener(self):
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        if self.is_learning_hotkey:
            self.learning_keys.add(key)
            return

        if key in self.hotkey_check:
            self.current_keys.add(key)
            
        if self.current_keys == self.hotkey_check and not self.recorder.recording and not self.is_transcribing:
            print("Start recording...")
            try:
                sd.play(self.start_sound, samplerate=SAMPLE_RATE)
            except Exception as e:
                print(f"Error playing sound: {e}")
            # Mostrar ventana de ondas en lugar de cambiar el icono
            self.wave_window.show()
            self.recorder.start()

    def on_release(self, key):
        if self.is_learning_hotkey:
            # When any key is released during learning, we finalize the shortcut
            # using the set of keys that were pressed together.
            if self.learning_keys:
                self.hotkey_check = self.learning_keys.copy()
                self.config["hotkey"] = self.serialize_hotkey(self.hotkey_check)
                self.save_config()
                
                # Reset UI
                self.is_learning_hotkey = False
                self.title = "🎙️"
                rumps.notification("Ghost Eagle", "Shortcut Saved", "New shortcut has been saved.")
                
                # Update menu state (clear others)
                for item in self.menu["Shortcut"]["Select Shortcut"].values():
                    item.state = 0
            return

        if key in self.current_keys:
            self.current_keys.remove(key)
        
        # If we release any key of the hotkey combo and we are recording, stop.
        # This logic mimics "hold to record".
        # If hotkey is single key (F5), releasing F5 stops.
        # If hotkey is Cmd+Opt+R, releasing any of them stops.
        if self.recorder.recording:
            # Check if the combo is broken
            if not self.hotkey_check.issubset(self.current_keys): 
                print("Stop recording...")
                # Ocultar ventana de ondas
                self.wave_window.hide()
                audio = self.recorder.stop()
                try:
                    sd.play(self.stop_sound, samplerate=SAMPLE_RATE)
                except Exception as e:
                    print(f"Error playing sound: {e}")
                    
                if audio is not None:
                    threading.Thread(target=self.transcribe_and_paste, args=(audio,)).start()

    def transcribe_and_paste(self, audio):
        self.is_transcribing = True
        try:
            print("Transcribing...")
            # Normalize audio to float32 range [-1, 1] if needed, sounddevice usually gives float32
            # mlx_whisper expects a string file or np array.
            
            # Simple VAD/Silence check could go here, but user wants ultra-fast.
            
            # Flatten if needed (sounddevice returns [frames, channels])
            audio_flat = audio.flatten()
            
            text = mlx_whisper.transcribe(
                audio_flat, 
                path_or_hf_repo=self.model_path,
                verbose=False
            )["text"]
            
            text = text.strip()
            print(f"Transcribed: {text}")
            
            if text:
                pyperclip.copy(text)
                # Small delay to ensure clipboard is ready
                time.sleep(0.2) 
                
                print("Pasting...")
                # Try pasting using AppleScript which is often more reliable on macOS for this
                try:
                    subprocess.run(
                        ["osascript", "-e", 'tell application "System Events" to keystroke "v" using command down'], 
                        check=True,
                        capture_output=True,
                        text=True
                    )
                except subprocess.CalledProcessError as e:
                    print(f"AppleScript paste failed: {e.stderr}")
                    if "1002" in e.stderr or "permiso" in e.stderr:
                         rumps.notification(
                             "⚠️ Faltan Permisos", 
                             "Error al Pegar", 
                             "Tu Terminal necesita permisos de 'Accesibilidad' y 'Automatización' para pegar texto. Revisa Ajustes del Sistema > Privacidad."
                         )
                    
                    print("Falling back to pyautogui...")
                    pyautogui.hotkey('command', 'v')
                except Exception as as_e:
                    print(f"Unexpected error during paste: {as_e}")
                    pyautogui.hotkey('command', 'v')
                
        except Exception as e:
            print(f"Error during transcription: {e}")
            rumps.notification("Transcriber Error", "Failed to transcribe", str(e))
        finally:
            self.is_transcribing = False

if __name__ == "__main__":
    app = VoiceTranscriberApp()
    app.run()

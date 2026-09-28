"""
async_tts.py — Motor de síntesis de voz (TTS) asíncrono y desacoplado del loop de video.
Utiliza un hilo worker dedicado con cola de mensajes para garantizar CERO desfase perceptual
y evitar cualquier congelamiento en la tasa de fotogramas (FPS) de OpenCV.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pyttsx3

from src.config_dynamic import TTS_RATE, TTS_VOLUME


class AsyncTTSEngine:
    """
    Motor TTS no bloqueante basado en cola Productor-Consumidor.
    """

    def __init__(self, rate: int = TTS_RATE, volume: float = TTS_VOLUME) -> None:
        self.rate = rate
        self.volume = volume

        self._queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=10)
        self._running: bool = True
        self._last_spoken_text: str = ""
        self._last_spoken_time: float = 0.0
        self._cooldown_seconds: float = 1.2

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="AsyncTTSWorker",
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Ciclo de vida del hilo secundario dedicado a la síntesis de voz."""
        # Inicializar pyttsx3 dentro del mismo hilo que lo ejecutará (requerido por COM en Windows)
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)

            # Buscar voz en español si está disponible
            voices = engine.getProperty("voices")
            for v in voices:
                if "spanish" in v.name.lower() or "es" in getattr(v, "languages", [""])[0].lower():
                    engine.setProperty("voice", v.id)
                    break
        except Exception as e:
            print(f"[ERROR TTS] No se pudo inicializar pyttsx3: {e}")
            engine = None

        while self._running:
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:  # Señal de parada
                break

            if engine is not None and text.strip():
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as ex:
                    print(f"[ERROR TTS] Error durante la reproducción de voz: {ex}")

            self._queue.task_done()

        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    def speak(self, text: str, min_interval: float = 1.0) -> bool:
        """
        Encola un texto para ser sintetizado de forma inmediata y asíncrona.

        Args:
            text: Palabra o frase a reproducir.
            min_interval: Intervalo mínimo entre reproducciones de la misma palabra para evitar eco.

        Returns:
            True si fue encolado con éxito, False si fue descartado por cooldown o cola llena.
        """
        now = time.time()
        text_clean = text.strip()

        if not text_clean:
            return False

        # Evitar repetir la misma palabra si se dispara consecutivamente
        if text_clean == self._last_spoken_text and (now - self._last_spoken_time) < min_interval:
            return False

        try:
            self._queue.put_nowait(text_clean)
            self._last_spoken_text = text_clean
            self._last_spoken_time = now
            return True
        except queue.Full:
            print(f"[AVISO TTS] Cola saturada; descartando: '{text_clean}'")
            return False

    def stop(self) -> None:
        """Detiene el hilo del motor y libera recursos."""
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)


if __name__ == "__main__":
    print("[TEST TTS] Probando reproducción asíncrona no bloqueante...")
    tts = AsyncTTSEngine()
    tts.speak("Hola")
    print("[TEST TTS] Llamada speak() retornó inmediatamente sin congelar la consola.")
    time.sleep(1.5)
    tts.speak("Gracias")
    time.sleep(1.5)
    tts.stop()
    print("[TEST TTS] Prueba finalizada exitosamente.")

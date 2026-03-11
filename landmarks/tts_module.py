import threading
import queue
import importlib


class SpeechSynthesizer:
    """Text-to-speech wrapper with queue-based async speech."""

    def __init__(self, rate=180, volume=1.0):
        self._queue = None
        self._engine = None
        self._thread = None
        self._rate = rate
        self._volume = volume
        self._last_spoken = ""
        self._last_spoken_time = 0.0
        self._running = False

        try:
            pyttsx3 = importlib.import_module("pyttsx3")
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", rate)
            self._engine.setProperty("volume", volume)
        except Exception:
            self._engine = None
            return

        self._queue = queue.Queue()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        """Background thread that processes speech queue."""
        while self._running:
            try:
                text = self._queue.get(timeout=0.1)
                if text is None:
                    break
                if self._engine:
                    self._engine.say(text)
                    self._engine.runAndWait()
            except queue.Empty:
                continue
            except Exception:
                pass

    def speak(self, text, interval=1.5):
        """
        Queue text for speech.
        Speaks immediately if text changed, otherwise respects interval.
        """
        if self._queue is None or self._engine is None:
            return False

        import time
        now = time.time()
        should_speak = (text != self._last_spoken) or ((now - self._last_spoken_time) >= interval)

        if should_speak:
            try:
                self._queue.put(text, block=False)
                self._last_spoken = text
                self._last_spoken_time = now
                return True
            except queue.Full:
                pass
        return False

    def close(self):
        """Stop the speech thread and cleanup."""
        self._running = False
        if self._queue:
            try:
                self._queue.put(None, block=False)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=1.0)


# Simple factory function for quick setup
def create_synthesizer(rate=180, volume=1.0):
    """Create and return a SpeechSynthesizer. Returns None if pyttsx3 unavailable."""
    synth = SpeechSynthesizer(rate=rate, volume=volume)
    return synth if synth._engine else None

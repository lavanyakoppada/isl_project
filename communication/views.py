# # from django.shortcuts import render

# # # Create your views here.
# from django.shortcuts import render
# from django.http import JsonResponse
# import spacy
# import nltk
# from nltk.corpus import stopwords

# nltk.download('stopwords')
# nlp = spacy.load("en_core_web_sm")

# STOP_WORDS = set(stopwords.words("english")) - {"i", "you", "he", "she", "we", "they"}
# AUX_VERBS = {"am", "is", "are", "was", "were", "be", "been", "being"}

# def text_to_isl_gloss(text):
#     doc = nlp(text.lower())
#     subject, obj, verb, adj = [], [], [], []

#     for token in doc:
#         if token.is_punct or token.text in STOP_WORDS:
#             continue
#         if token.dep_ in ("nsubj", "nsubjpass"):
#             subject.append(token.lemma_)
#         elif token.lemma_ in AUX_VERBS:
#             continue
#         elif token.pos_ == "VERB":
#             verb.append(token.lemma_)
#         elif token.pos_ == "ADJ":
#             adj.append(token.lemma_)
#         elif token.pos_ in ("NOUN", "PROPN", "PRON"):
#             obj.append(token.lemma_)

#     isl = subject + adj + obj + verb
#     return [word.upper() for word in isl]

# def home(request):
#     return render(request, "home.html")

# def speechtosign(request):
#     return render(request, "speechtosign.html")

# def convert_speech(request):
#     if request.method == "POST":
#         import json
#         data = json.loads(request.body)
#         text = data.get("text", "")
#         gloss = text_to_isl_gloss(text)
#         return JsonResponse({"gloss": gloss})
#     return JsonResponse({"error": "Invalid request"}, status=400)




from django.shortcuts import render
from django.http import JsonResponse
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import spacy
import nltk
from nltk.corpus import stopwords
import subprocess
import sys
import os
from django.conf import settings
import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
import time
import importlib.util

# -------------------------
# NLP SETUP
# -------------------------
nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")

STOP_WORDS = set(stopwords.words("english")) - {
    "i", "you", "he", "she", "we", "they"
}

AUX_VERBS = {
    "am", "is", "are", "was", "were", "be", "been", "being"
}

TIME_WORDS = {
    "today", "tomorrow", "yesterday",
    "morning", "evening", "night"
}

WH_WORDS = {
    "what", "where", "when", "why", "how", "who"
}

# -------------------------
# TEXT → ISL GLOSS
# -------------------------
def text_to_isl_gloss(text):
    """
    ISL Order:
    TIME + SUBJECT + ADJECTIVE + OBJECT + VERB + WH
    """
    doc = nlp(text.lower())

    time = []
    subject = []
    adj = []
    obj = []
    verb = []
    wh = []

    for token in doc:
        if token.is_punct:
            continue

        # WH words
        if token.lemma_ in WH_WORDS:
            wh.append(token.lemma_)
            continue

        # TIME words (IMPORTANT: do NOT return early)
        if token.ent_type_ in ("DATE", "TIME") or token.text in TIME_WORDS:
            time.append(token.text)
            continue

        # SUBJECT (pronouns + nouns)
        if token.dep_ in ("nsubj", "nsubjpass"):
            subject.append(token.text)
            continue

        # Ignore auxiliary verbs
        if token.lemma_ in AUX_VERBS:
            continue

        # ADJECTIVES
        if token.pos_ == "ADJ":
            adj.append(token.text)
            continue

        # MAIN VERBS
        if token.pos_ == "VERB":
            verb.append(token.text)
            continue

        # OBJECTS
        if token.pos_ in ("NOUN", "PROPN", "PRON"):
            if token.text not in STOP_WORDS:
                obj.append(token.text)
            continue

    isl = time + subject + adj + obj + verb + wh

    return [word.upper() for word in isl]

# -------------------------
# PAGES
# -------------------------
def home(request):
    return render(request, "home.html")

def speechtosign(request):
    return render(request, "speechtosign.html")

def signtospeech(request):
    return render(request, "signtospeech.html")

def play_signs(request):
    gloss = request.GET.get("gloss", "")
    gloss_list = gloss.split(",") if gloss else []
    return render(request, "play_signs.html", {"gloss": gloss_list})

# -------------------------
# API
# -------------------------
@csrf_exempt
def convert_speech(request):
    if request.method == "POST":
        data = json.loads(request.body)
        text = data.get("text", "")
        gloss = text_to_isl_gloss(text)
        return JsonResponse({"gloss": gloss})

    return JsonResponse({"error": "Invalid request"}, status=400)

# -------------------------
# Recording: start/stop
# -------------------------
REC_PROC = None
STREAM_ACTIVE = False
STOP_STREAM = False
LAST_GLOSS = ""

def _paths():
    project_root = settings.BASE_DIR.parent
    landmarks_dir = os.path.join(project_root, "sign_to_speech", "landmarks")
    if not os.path.exists(landmarks_dir):
        # Fallback to old path just in case
        landmarks_dir = os.path.join(project_root, "landmarks")
    
    script = os.path.join(landmarks_dir, "recorded_predictions.py")
    models_dir = os.path.join(landmarks_dir, "models")
    output_file = os.path.join(landmarks_dir, "recorded_predictions.txt")
    stop_flag = os.path.join(landmarks_dir, "STOP_RECORDING.flag")
    return script, models_dir, output_file, stop_flag

@csrf_exempt
def record_start(request):
    global REC_PROC
    global LAST_GLOSS
    global STOP_STREAM
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=400)
    script, models_dir, output_file, stop_flag = _paths()
    if not os.path.exists(script):
        return JsonResponse({"error": "script_not_found", "path": script}, status=500)
    if not os.path.exists(models_dir):
        return JsonResponse({"error": "models_dir_missing", "path": models_dir}, status=500)
    try:
        if os.path.exists(stop_flag):
            os.remove(stop_flag)
    except Exception:
        pass
    LAST_GLOSS = ""
    STOP_STREAM = False
    try:
        with open(output_file, "w") as f:
            f.write("")
    except Exception:
        pass
    if REC_PROC and REC_PROC.poll() is None:
        return JsonResponse({"status": "already_running", "pid": REC_PROC.pid})
    
    # Ensure any previous stream flag is cleared
    STOP_STREAM = False
    
    cmd = [
        sys.executable,
        script,
        "--models-dir", models_dir,
        "--output-file", output_file,
        "--stop-flag", stop_flag,
    ]
    try:
        return JsonResponse({"status": "ready"})
    except Exception as e:
        return JsonResponse({"error": "spawn_failed", "detail": str(e)}, status=500)

@csrf_exempt
def record_stop(request):
    global REC_PROC
    global STOP_STREAM
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=400)
    _, _, _, stop_flag = _paths()
    try:
        with open(stop_flag, "w") as f:
            f.write("stop")
    except Exception:
        pass
    STOP_STREAM = True
    return JsonResponse({"status": "stopping"})

def _simple_translate(gloss_content: str) -> str:
    text = (gloss_content or "").strip().lower()
    if not text:
        return ""
    tokens = [t for t in text.split() if t != "break"]
    phrase = " ".join(tokens)
    if not phrase:
        return ""
    if "how are you" in phrase:
        return "How are you?"
    if phrase == "thank":
        return "Thank you."
    nouns = {"team", "attendance", "sun", "home", "school", "work", "office", "place", "seat"}
    for wh in ("where", "what", "who", "when", "why", "how"):
        if tokens and tokens[0] == wh and len(tokens) >= 2:
            rest = " ".join(tokens[1:])
            first = tokens[1]
            if first in nouns:
                rest = f"the {rest}"
            sent = f"{wh} is {rest}".strip()
            if not sent.endswith("?"):
                sent += "?"
            return sent[0].upper() + sent[1:]
    sent = phrase.capitalize()
    if not sent.endswith((".", "!", "?")):
        sent += "."
    return sent
# -------------------------
# Singleton Model Loader
# -------------------------
class SignModelLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SignModelLoader, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.id_to_label = None
            cls._instance.mean = None
            cls._instance.std = None
            cls._instance.landmarker = None
            cls._instance.last_models_dir = None
        return cls._instance

    def load(self, models_dir):
        if self.model is not None and self.last_models_dir == models_dir:
            return
            
        print(f"--- [SignModelLoader] Loading Models from {models_dir} ---")
        try:
            self.model = tf.keras.models.load_model(os.path.join(models_dir, "ann_landmarks.keras"))
            with open(os.path.join(models_dir, "label_map.json"), "r") as f:
                label_to_id = json.load(f)
            self.id_to_label = {int(v): k for k, v in label_to_id.items()}
            norm = np.load(os.path.join(models_dir, "normalization.npz"))
            self.mean = norm["mean"]
            self.std = norm["std"]
            
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            p = os.path.join(models_dir, "hand_landmarker.task")
            if not os.path.exists(p):
                p = os.path.join("models", "hand_landmarker.task")
                
            base = mp_python.BaseOptions(model_asset_path=p)
            opts = vision.HandLandmarkerOptions(
                base_options=base, 
                num_hands=2, 
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                running_mode=vision.RunningMode.VIDEO
            )
            self.landmarker = vision.HandLandmarker.create_from_options(opts)
            self.last_models_dir = models_dir
            print("--- [SignModelLoader] Successfully Loaded ---")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

def _norm_vec(pts):
    if not pts:
        return None
    arr = np.array([[lm.x, lm.y, lm.z] for lm in pts])
    wrist = arr[0]
    arr = arr - wrist
    s = np.linalg.norm(arr[9])
    if s > 0:
        arr = arr / s
    return arr.flatten()

def _landmarks_to_vector(result):
    left = None
    right = None
    if hasattr(result, "hand_landmarks") and result.hand_landmarks:
        for idx, handedness in enumerate(result.handedness):
            label = handedness[0].category_name.lower()
            lms = result.hand_landmarks[idx]
            vec = _norm_vec(lms)
            if label == "left":
                left = vec
            elif label == "right":
                right = vec
    if left is None and right is not None:
        left = right
        right = None
    if left is None:
        return None
    zeros = [0.0] * 63
    final = np.concatenate([left, (right if right is not None else zeros)])
    return final.astype(np.float32)

def _gen_frames(models_dir, output_file):
    global STOP_STREAM
    # Prevent expensive reloading
    loader = SignModelLoader()
    loader.load(models_dir)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("--- [Camera] Failed to open camera ---")
        return

    prob_buffer = []
    recorded_signs = []
    last_stable_label = None
    稳定_count = 0
    last_top_id = -1
    
    capturing = False
    capture_start = 0.0
    # MediaPipe detection_for_video requires monotonically increasing timestamps.
    # Using time.time() * 1000 provides a absolute timestamp that won't reset
    # even if the generator is restarted.
    
    try:
        while True:
            if STOP_STREAM:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            # Use absolute milliseconds to ensure monotonicity across calls
            current_ms = int(time.time() * 1000)
            
            try:
                r = loader.landmarker.detect_for_video(mp_image, current_ms)
            except Exception as e:
                print(f"--- [MediaPipe] Detection error: {e} ---")
                # If timestamp is the issue, we might need a small offset or retry
                continue

            x = _landmarks_to_vector(r)
            
            display_text = "Searching for hands..."
            conf_val = 0.0
            
            # Visual cues for hands
            if hasattr(r, "hand_landmarks") and r.hand_landmarks:
                for hand_landmarks in r.hand_landmarks:
                    h, w = frame.shape[:2]
                    for lm in hand_landmarks:
                        px, py = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)
                        
            if x is not None:
                if not capturing:
                    capturing = True
                    capture_start = time.time()
                    prob_buffer.clear()
                    
                x = (x - loader.mean) / loader.std
                probs = loader.model.predict(x.reshape(1, -1), verbose=0)[0]
                prob_buffer.append(probs)
                if len(prob_buffer) > 15: # Aligned with smooth_window in scripts
                    prob_buffer.pop(0)
                    
                avg_probs = np.mean(prob_buffer, axis=0)
                top_idx = np.argsort(avg_probs)[-3:][::-1]
                current_top_id = top_idx[0]
                current_conf = avg_probs[current_top_id]
                
                if current_top_id == last_top_id:
                    稳定_count += 1
                else:
                    稳定_count = 0
                last_top_id = current_top_id
                
                # Align with hold_sec=1.5 and stable_frames=12 from scripts
                ready = (time.time() - capture_start) >= 1.5
                if ready and 稳定_count >= 12 and current_conf >= 0.45:
                    label_name = loader.id_to_label[current_top_id]
                    conf_val = current_conf
                    display_text = f"LOCKED: {label_name}"
                    if label_name != last_stable_label:
                        recorded_signs.append(label_name)
                        last_stable_label = label_name
                else:
                    display_text = "Analyzing movement..."
            else:
                prob_buffer.clear()
                稳定_count = 0
                last_top_id = -1
                capturing = False
                last_stable_label = None # Reset when hand leaves to allow re-detect
                
            # Draw Overlay
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 60), (40, 40, 40), -1)
            cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if conf_val > 0:
                bw = int(conf_val * 200)
                cv2.rectangle(frame, (20, 65), (20 + bw, 75), (0, 255, 0), -1)
            signs_str = " | ".join(recorded_signs[-4:])
            cv2.putText(frame, f"Stored Seq: {signs_str}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytearray(buffer) + b"\r\n")
    finally:
        print("--- [Camera] Releasing resources ---")
        cap.release()
        STOP_STREAM = False
        if recorded_signs:
            content = " ".join(recorded_signs)
            globals()["LAST_GLOSS"] = content
            try:
                with open(output_file, "w") as f:
                    f.write(content)
            except Exception:
                pass


def record_stream(request):
    global STREAM_ACTIVE
    if STREAM_ACTIVE:
        pass
    STREAM_ACTIVE = True
    _, models_dir, output_file, _ = _paths()
    resp = StreamingHttpResponse(_gen_frames(models_dir, output_file), content_type="multipart/x-mixed-replace; boundary=frame")
    return resp

# -------------------------
# Gloss → English sentence
# -------------------------
@csrf_exempt
def gloss_sentence(request):
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request"}, status=400)
    
    script, _, gloss_file, _ = _paths()
    landmarks_dir = os.path.dirname(script)
    
    gloss_content = globals().get("LAST_GLOSS", "")
    if not gloss_content:
        if os.path.exists(gloss_file):
            try:
                with open(gloss_file, "r") as f:
                    gloss_content = f.read().strip()
            except Exception:
                gloss_content = ""
                
    if not gloss_content:
        return JsonResponse({"sentence": "No signs detected.", "gloss": ""})

    module_path = os.path.join(landmarks_dir, "gloss_to_speech.py")
    try:
        # Add landmarks_dir to sys.path so it can find tts_module
        if landmarks_dir not in sys.path:
            sys.path.append(landmarks_dir)
            
        spec = importlib.util.spec_from_file_location("gloss_to_speech", module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sentence = mod.pure_nlp_processor(gloss_content)
    except Exception as e:
        print(f"Extraction error: {e}")
        sentence = _simple_translate(gloss_content) or gloss_content
    if not sentence or sentence.strip().lower() == gloss_content.strip().lower():
        simple = _simple_translate(gloss_content)
        if simple:
            sentence = simple
    resp = JsonResponse({"sentence": sentence, "gloss": gloss_content})
    resp["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp["Pragma"] = "no-cache"
    return resp

@csrf_exempt
def tts_speak(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request"}, status=400)
    try:
        data = json.loads(request.body)
        text = (data.get("text") or "").strip()
    except Exception:
        text = ""
    if not text:
        return JsonResponse({"error": "no_text"}, status=400)
    try:
        from landmarks.tts_module import create_synthesizer
        synth = create_synthesizer(rate=180)
        if not synth:
            return JsonResponse({"error": "tts_unavailable"}, status=500)
        ok = synth.speak(text)
        time.sleep(0.1)
        return JsonResponse({"status": "speaking" if ok else "queued"})
    except Exception as e:
        return JsonResponse({"error": "tts_failed", "detail": str(e)}, status=500)

@csrf_exempt
def gloss_code(request):
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request"}, status=400)
    
    script, _, _, _ = _paths()
    landmarks_dir = os.path.dirname(script)
    module_path = os.path.join(landmarks_dir, "gloss_to_speech.py")
    if not os.path.exists(module_path):
        return JsonResponse({"code": ""})
    try:
        with open(module_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return JsonResponse({"error": "read_failed", "detail": str(e)}, status=500)
    return JsonResponse({"code": content})
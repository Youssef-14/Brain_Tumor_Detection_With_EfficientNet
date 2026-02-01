import os
import threading
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Cache des modèles chargés par chemin absolu
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()

# chemin par défaut relatif au dépôt (conservé comme fallback)
DEFAULT_MODEL_NAME = "model.h5"
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", DEFAULT_MODEL_NAME)


class DummyModel:
    """Fallback minimal: renvoie prédiction neutre et logge le problème."""

    def predict(self, x, *args, **kwargs):
        print("WARNING: DummyModel used because real models failed to load.")
        # Retourne une probabilité proche de 0 (pas de tumeur)
        return np.array([[0.0]])


def _first_model_in_model_dir():
    """Retourne le chemin absolu du premier fichier .h5 (ordre alphabétique, insensible à la casse)
    présent dans le dossier 'models', ou None si aucun trouvé.
    """
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    try:
        files = [f for f in os.listdir(model_dir)
                 if os.path.isfile(os.path.join(model_dir, f)) and (f.lower().endswith('.h5') or f.lower().endswith('.keras'))]
        if not files:
            return None
        files.sort(key=lambda s: s.lower())
        return os.path.abspath(os.path.join(model_dir, files[0]))
    except Exception:
        return None


def _resolve_model_path(model_name_or_path):
    """Résout un nom de modèle (ex: 'VGG_model.h5') ou un chemin vers un chemin absolu existant.

    Retourne le chemin absolu si trouvé, sinon None.
    """
    # Si aucune sélection fournie -> choisir le premier .h5 alphabétique dans models/
    if not model_name_or_path:
        first = _first_model_in_model_dir()
        if first:
            return first
        # fallback vers l'ancien fichier nominal si présent
        if os.path.exists(DEFAULT_MODEL_PATH):
            return os.path.abspath(DEFAULT_MODEL_PATH)
        return None

    # Chemin absolu fourni
    if os.path.isabs(model_name_or_path) and os.path.exists(model_name_or_path):
        return os.path.abspath(model_name_or_path)

    # Chercher dans le dossier 'models' du projet
    candidate = os.path.join(os.path.dirname(__file__), "models", model_name_or_path)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)

    # Chercher chemin relatif simple 'models/xxx'
    candidate2 = os.path.join("models", model_name_or_path)
    if os.path.exists(candidate2):
        return os.path.abspath(candidate2)

    # Si on reçoit déjà un chemin relatif qui existe
    if os.path.exists(model_name_or_path):
        return os.path.abspath(model_name_or_path)

    return None


def _load_saved_model(model_name_or_path=None):
    """Charge (et met en cache) un modèle Keras à partir d'un nom ou chemin.

    Retourne un objet modèle (ou DummyModel en cas d'erreur).
    """
    resolved = _resolve_model_path(model_name_or_path)
    if resolved is None:
        print(f"Model not found for '{model_name_or_path}', using DummyModel as fallback.")
        return DummyModel()

    # Utiliser le cache pour éviter de recharger le même modèle plusieurs fois
    with _MODEL_CACHE_LOCK:
        if resolved in _MODEL_CACHE:
            # cache hit
            # print utile pour debug
            print(f"Model cache hit: {resolved}")
            return _MODEL_CACHE[resolved]

    # cache miss -> tenter de charger
    try:
        print(f"Loading models from: {resolved}")
        model = load_model(resolved, compile=False)
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE[resolved] = model
        print(f"Modèle chargé avec succès : {resolved}")
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle '{resolved}':", e)
        print("Tentative de contournement : utilisation d'un DummyModel (fallback).")
        dummy = DummyModel()
        with _MODEL_CACHE_LOCK:
            # mémoriser le fallback pour éviter de retenter inutilement
            _MODEL_CACHE[resolved] = dummy
        return dummy


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def check(model_name_or_path, input_img):
    """Prédit si l'image contient une tumeur en utilisant le modèle demandé.

    model_name_or_path: nom de fichier (ex: 'VGG_model.h5') ou chemin vers le modèle.
    input_img: nom de fichier dans le dossier 'images/' (ex: '5.jpg').

    Retourne un dict:
      - pour les modèles binaires: {'mode':'binary','prediction':bool,'score':float}
      - pour les modèles multiclasses: {'mode':'multiclass','label':str,'index':int,'probs': [floats]}

    En cas d'erreur, retourne {'mode':'binary','prediction':False,'score':0.0} par défaut.
    """
    print(f"Using models: {model_name_or_path} for image: {input_img}")

    model = _load_saved_model(model_name_or_path)

    # Chargement et préparation de l'image
    img_path = os.path.join(os.path.dirname(__file__), "images", input_img)
    # fallback : si l'image n'est pas dans package, regarder chemin relatif
    if not os.path.exists(img_path):
        img_path = os.path.join("images", input_img)

    try:
        img = image.load_img(img_path, target_size=(224, 224))
    except Exception as e:
        print(f"Erreur chargement image {img_path} :", e)
        # Retourner False si image non chargée
        return {'mode': 'binary', 'prediction': False, 'score': 0.0}

    img = image.img_to_array(img)
    # Assurer le bon dtype et normalisation si le modèle attend preprocess_input
    try:
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

        img = preprocess_input(img)
    except Exception:
        # si preprocess_input indisponible, normaliser entre 0 et 1
        img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    print("Input shape pour prédiction :", img.shape)
    try:
        output = model.predict(img)
    except Exception as e:
        print("Erreur lors de models.predict:", e)
        # fallback: considérer comme négatif
        return {'mode': 'binary', 'prediction': False, 'score': 0.0}

    print("Raw models output :", output)

    # Post-traitement: détecter sortie binaire vs multiclasses
    try:
        out_arr = np.asarray(output)
        # Normaliser forme
        flat = out_arr.flatten()
        if flat.size > 1:
            # multiclass
            probs = flat.astype('float64')
            # si ne ressemble pas à des probabilités, appliquer softmax
            s = probs.sum()
            if not (np.isfinite(s) and s > 0 and np.isclose(s, 1.0, rtol=1e-3)):
                probs = _softmax(probs)
            # Mapping des labels par défaut si 4 classes
            if probs.size == 4:
                labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
            else:
                labels = [f'class_{i}' for i in range(probs.size)]
            idx = int(np.argmax(probs))
            label = labels[idx]
            print(f"Multiclass prediction: {label} (index {idx}) probs={probs}")
            return {'mode': 'multiclass', 'label': label, 'index': idx, 'probs': probs.tolist(), 'labels': labels}
        else:
            # binary output
            val = float(flat[0])
            # si sortie est dans [0,1], l'utiliser comme score, sinon le transformer
            score = val
            if not (0.0 <= score <= 1.0):
                # appliquer sigmoid
                score = 1.0 / (1.0 + np.exp(-score))
            prediction = bool(score >= 0.5)
            print(f"Binary prediction: {prediction} (score={score})")
            return {'mode': 'binary', 'prediction': prediction, 'score': float(score)}
    except Exception as e:
        print("Erreur traitement sortie modèle :", e)
        return {'mode': 'binary', 'prediction': False, 'score': 0.0}

import os
from flask import Flask, render_template, request
from predictor import check


author = 'MAATOUG ET ATAWA'

app = Flask(__name__, static_folder="images")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_ROOT, 'models')


@app.route('/')
@app.route('/index')
def index():
    # Lister les fichiers .h5 dans le dossier models pour alimenter le <select>
    model_list = []
    default_model = None
    try:
        model_list = [f for f in os.listdir(MODEL_DIR) if os.path.isfile(os.path.join(MODEL_DIR, f)) and (f.lower().endswith('.h5') or f.lower().endswith('.keras'))]
        # Trier insensible à la casse et choisir le premier comme default
        model_list.sort(key=lambda s: s.lower())
        if model_list:
            default_model = model_list[0]
    except Exception as e:
        print('Erreur lecture dossier models:', e)
    return render_template('upload.html', model_list=model_list, default_model=default_model)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    filename = None
    for file in request.files.getlist('file'):
        print(file)
        filename = file.filename
        print(filename)
        # utiliser os.path.join pour éviter les doubles slashes et problèmes de plateforme
        dest = os.path.join(target, filename)
        print(dest)
        file.save(dest)

    # Récupérer le modèle sélectionné (si présent), sinon utiliser None pour la valeur par défaut
    model_name = request.form.get('model_name')
    print('Model selected:', model_name)

    status = check(model_name, filename)

    return render_template('complete.html', image_name=filename, predvalue=status, model_name=model_name)

# Correction : guard doit être "__main__"
if __name__ == "__main__":
    app.run(port=4555, debug=True)

from flask import Flask, request, jsonify
from deepface import DeepFace
from sklearn.metrics import precision_score
import os
import tempfile

app = Flask(__name__)


def save_image_to_temp_file(image):
    _, temp_file_path = tempfile.mkstemp()
    image.save(temp_file_path)
    return temp_file_path


def predict_with_multiple_models(image1_path, image2_path):
    models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace']
    positive_results = 0

    for model_name in models:
        result = DeepFace.verify(image1_path, image2_path, model_name=model_name, enforce_detection=True)

        print(f"- The model is {model_name}\n- The model result is {result['verified']}\n{str(result)}\n")

        if 'verified' in result and result['verified'].item():
            positive_results += 1

    return positive_results > len(models) / 2


@app.route('/predict', methods=['POST'])
def predict():
    image1 = request.files.get('image1')
    image2 = request.files.get('image2')

    image1_path = save_image_to_temp_file(image1)
    image2_path = save_image_to_temp_file(image2)

    try:
        result = predict_with_multiple_models(image1_path, image2_path)
        # Converts numpy bool to native bool
        result = {"verified": result}
    except ValueError as e:
        return jsonify({"error": str(e)})

    os.remove(image1_path)
    os.remove(image2_path)

    return jsonify(result)


@app.route('/precision', methods=['GET'])
def calculate_precision():
    # Assuming image_pairs_and_labels is a list of tuples,
    # where each tuple is a pair of image paths and a true match status.
    # [('path_to_image1a', 'path_to_image1b', True), ('path_to_image2a', 'path_to_image2b', False), ...]
    image_pairs_and_labels = ...

    predictions = []
    true_labels = []

    for image1_path, image2_path, is_match in image_pairs_and_labels:
        result = predict_with_multiple_models(image1_path, image2_path)
        predictions.append(result)  # Converts numpy bool to native bool
        true_labels.append(is_match)

    precision = precision_score(true_labels, predictions)

    return jsonify({"precision": precision})


if __name__ == "__main__":
    app.run(port=5000)

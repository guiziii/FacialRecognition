
# DeepFace Flask API

DeepFace Flask API provides a RESTful interface for facial analysis and verification tasks, utilizing the DeepFace library for deep learning-based face recognition and attribute analysis. This API supports image comparison, verifying identities, and evaluating the precision of the verification process.

## Features

- Facial attribute analysis including age, gender, race, and emotion.
- Face verification to determine if two faces belong to the same person.
- Precision calculation for verification processes.

## Requirements

- Python 3.6+
- Flask
- DeepFace
- scikit-learn

Ensure you have the required dependencies by installing them through `pip`:

```bash
pip install flask deepface sklearn
```

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/guiziii/FacialRecognition.git
cd FacialRecognition
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Start the Flask server by running:

```bash
python app.py
```

The API will be available at `http://localhost:5000`.

### Endpoints

#### `/predict`

- **Method:** POST
- **Description:** Compares two images and predicts if they belong to the same person.
- **Payload:** `multipart/form-data` with two image files named `image1` and `image2`.

#### `/precision`

- **Method:** GET
- **Description:** Calculates the precision of the face verification process over a predefined set of image pairs and labels.

## Example Requests

You can use tools like `curl` or Postman to interact with the API. Here's an example using `curl`:

```bash
curl -X POST -F "image1=@path_to_image1.jpg" -F "image2=@path_to_image2.jpg" http://localhost:5000/predict
```

## Contributing

Contributions to the DeepFace Flask API are welcome. Please ensure you follow the provided coding standards and contribute to the existing functionalities.

## License

[MIT License](LICENSE)

Please note that the use of the DeepFace library and other dependencies might be subject to their own licenses.

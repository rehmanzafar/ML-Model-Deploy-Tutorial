import requests


def test_model():
    data = {
        "data": [0.038, 0.050, 0.061, 0.022, -0.044, -0.034, -0.043, -0.002, 0.019, -0.017]
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    print("Prediction result:", response.json())


if __name__ == '__main__':
    test_model()

import requests

def check_tensorboard_status(url="http://localhost:6006"):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error checking TensorBoard status: {e}")
        return False

tensorboard_url = "http://localhost:6006" if check_tensorboard_status() else "TensorBoard not available"

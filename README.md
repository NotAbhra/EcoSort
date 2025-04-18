# EcoSort
EcoSort is a deep learning-based waste classification system that uses a VGG-style convolutional neural network to classify waste into bio-degradable and non-bio-degradable categories. It processes a large dataset of images and provides real-time predictions.

## Features
- **Deep Learning Model:** Utilizes a custom-built CNN based on the VGG architecture for high-accuracy classification.
- **Large Dataset Support:** Designed to process and classify a substantial number of waste images efficiently.
- **Automated Preprocessing:** Automatically resizes images and normalizes data for optimal model performance.
- **User-Friendly Prediction:** Allows users to input an image path and receive classification results instantly.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- OpenCV (cv2)
- NumPy
- scikit-learn
- Matplotlib

## Installation
Clone the repository:
```sh
git clone https://github.com/NotAbhra/EcoSort.git
```

Navigate to the project directory:
```sh
cd EcoSort
```

Install the required dependencies:
```sh
pip install -r requirements.txt
```

## Usage
Run the script and provide an image path:
```sh
python EcoSort.py --image path/to/image.jpg
```
The system will display the image and predict its category.

## Example Output
```sh
Enter the path to the image: test_waste.jpg
Prediction: bio-degradable
```

## Screenshot
![EcoSort Example](https://github.com/user-attachments/assets/78127b49-63f4-4b4c-907b-7abd8f243f02)

## Notes
- The dataset consists of a large collection of labeled waste images.
- The model can be further fine-tuned for improved accuracy.

## License
This project is licensed under the MIT License.

## Contributing
Feel free to fork the repository and submit pull requests. For issues or feature requests, please open an issue in the GitHub repository.


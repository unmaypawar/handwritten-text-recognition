# handwritten-text-recognition

## About

<p>This project is dedicated to the development of a sophisticated handwritten text recognition system leveraging Convolutional Neural Networks (CNNs). The primary objective is to accurately identify and interpret handwritten characters, subsequently transforming them into editable digital documents. Through meticulous training on a dataset rich in handwritten samples, the CNN model harnesses intricate spatial hierarchies within the input data, enabling precise character recognition. This approach not only enhances the efficiency of converting handwritten content into digital format but also contributes to the automation of tasks previously reliant on manual transcription.</p>

<p>At the heart of this project lies the intricate architecture of CNNs, which excel in capturing and understanding patterns in image data. By harnessing the power of deep learning, the system learns to discern the nuances of handwritten characters, even amidst variations in style, size, and orientation. Through iterative training and refinement, the CNN model becomes adept at recognizing a diverse range of handwritten text, ensuring robust performance across different contexts and writing styles. This adaptability is crucial in real-world scenarios where handwritten documents may vary widely in quality and presentation.</p>

<p>Furthermore, to showcase the capabilities of this system, a web-based demo has been developed, allowing users to experience firsthand the process of handwritten text recognition and document conversion. Through the intuitive interface of the web demo, users can upload handwritten documents and witness the CNN model in action as it accurately identifies and converts the handwritten characters into editable digital format.</p>

## Installation

```
conda create --name htr --file requirements.txt
```

## Usage

### Dataset

Download IAM dataset from [here](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)

### Training

```
python3 main.py --mode train --data_dir path/to/IAM
```

### Command line arguments

* `--mode`: select between "train", "validate" and "infer".
* `--batch_size`: batch size.
* `--data_dir`: directory containing IAM dataset (with subdirectories `img` and `gt`).
* `--fast`: use LMDB to load images faster.
* `--line_mode`: train reading text lines instead of single words.
* `--img_file`: image that is used for inference.

### Demo

* From command line:
```
python3 main.py --img_file ../data/line.png
```

* From web:
```
python3 main.py --infer_by_web 1
```
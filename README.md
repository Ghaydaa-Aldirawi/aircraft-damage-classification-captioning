✈️ Aircraft Damage Classification & Captioning

A two-stage pipeline for aircraft inspection images:

Classification: a VGG16 transfer-learning model labels surface damage as dent or crack.
Captioning & summarization: a pre-trained BLIP vision-language Transformer generates a caption and a short summary of the image.

📚 Course context: Final project of Introduction to Deep Learning & Neural Networks with Keras (IBM, Coursera). The project scenario, dataset, and notebook template were provided by the course; the implementation, training, and evaluation in this repository are my own work on top of that template.

🎯 Objective

Manual aircraft inspection is time-consuming and prone to human error. This project explores how pre-trained models can automatically classify visible damage and describe it in natural language, as a learning exercise in transfer learning and vision-language models.

📦 Dataset

Provided by the course (downloaded automatically by the notebook as aircraft_damage_dataset_v1). Image file names indicate a Roboflow export, and the dataset is reported by other projects to be licensed CC BY 4.0.

Split	Images
Train	300
Validation	96
Test	50

Classes: crack and dent (Keras assigns crack = 0, dent = 1). Images are resized to 224×224 and rescaled to [0, 1]; no data augmentation is applied.

🧠 Approach
1. Classification (feature extraction)
VGG16 (ImageNet weights, include_top=False) with all layers frozen, followed by Flatten
Custom head: Dense(512) → Dropout(0.3) → Dense(512) → Dropout(0.3) → Dense(1, sigmoid)
Adam (learning rate 1e-4), binary cross-entropy, batch size 32, 5 epochs, random seed 42
Training and validation accuracy/loss curves are plotted, and the model is evaluated on the test generator
2. Captioning & summarization (BLIP)
Model: Salesforce/blip-image-captioning-base (Hugging Face Transformers, PyTorch backend)
The model is wrapped in a custom Keras layer, BlipCaptionSummaryLayer, that calls BLIP through tf.py_function
Two prompts select the output type:
Caption: "This is a picture of"
Summary: "This is a detailed photo showing"
Caption and summary are generated for test images of dents and displayed alongside the image
📊 Evaluation
The classifier reports test loss and accuracy through model.evaluate on the test generator.
The course's evaluation cell computes steps = test_samples // batch_size, so with a 50-image test set it scores a single batch of 32 images rather than all 50.
A helper function also plots one test image with its true and predicted label (index 1 of the first test batch).
BLIP is a general-purpose model that was not trained on aircraft damage, so its captions are generic and may not mention the damage itself. The notebook outputs show the generated caption and summary for the example image.
🛠️ Tech Stack

Python · TensorFlow / Keras · PyTorch · Hugging Face Transformers (BLIP) · Pillow · NumPy · Matplotlib

🚀 How to Run

The notebook downloads the dataset (from a course-hosted link) and the pre-trained models automatically, so an internet connection is required. The BLIP model is roughly 1 GB. After downloading or cloning this repository, open a terminal in its folder and run:

bash
pip install tensorflow torch transformers pillow matplotlib numpy
jupyter notebook

Then open the notebook inside the project folder and run all cells. A GPU runtime such as Google Colab is recommended.

📁 Repository Structure
├── Aircraft Damage Classification & Captioning System/   # Notebook and script
└── README.md
🔮 Future Improvements
Add data augmentation and fine-tune the top VGG16 block
Evaluate on the entire test set and add a confusion matrix
Fine-tune BLIP on inspection-specific captions so descriptions mention the damage type
Add more damage classes (corrosion, scratches, missing fasteners)
🙏 Acknowledgments

Project template, scenario, and dataset provided by IBM Skills Network through the Introduction to Deep Learning & Neural Networks with Keras course (Coursera). Pre-trained models: VGG16 (Keras Applications) and BLIP (Salesforce).

👩‍💻 Author

Ghaydaa-Aldirawi

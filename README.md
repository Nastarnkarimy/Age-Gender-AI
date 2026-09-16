# 🧠 Age & Gender AI

A deep learning computer vision application for **simultaneous age estimation and gender classification** from facial images.

This project uses a **multi-task CNN architecture** to learn two related tasks from the same facial features:

* **Gender Classification** → Male / Female
* **Age Estimation** → Predicted age in years

The trained model is integrated into an interactive **Streamlit** application for real-time image inference.

---

## 🚀 Demo

![Age & Gender AI Demo](demo.png)

The application allows users to upload a facial image and receive:

* Predicted age
* Predicted gender
* Gender confidence score

---

## 🏗️ Architecture

Instead of training two completely independent models, this project uses a **shared CNN feature extractor with two output branches**.

```text
                         Input Image
                       128 × 128 × 3
                              │
                              ▼
                  ┌─────────────────────┐
                  │   CNN Feature       │
                  │     Extractor       │
                  │                     │
                  │ Conv2D + BN + ReLU  │
                  │ MaxPooling          │
                  │ Conv2D + BN + ReLU  │
                  │ MaxPooling          │
                  │ Conv2D + BN + ReLU  │
                  │ MaxPooling          │
                  │ Conv2D + BN + ReLU  │
                  │ MaxPooling          │
                  └──────────┬──────────┘
                             │
                       Shared Features
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
        ┌───────────────┐         ┌───────────────┐
        │ Gender Branch │         │   Age Branch  │
        │               │         │               │
        │ Dense +       │         │ Dense +       │
        │ Dropout       │         │ Dropout       │
        │               │         │               │
        │ Sigmoid       │         │ Linear        │
        └───────┬───────┘         └───────┬───────┘
                │                         │
                ▼                         ▼
          Male / Female             Estimated Age
```

---

## 🎯 Why Multi-Task Learning?

Age and gender are both strongly related to facial features.

Instead of forcing the model to learn two independent representations, the architecture shares the convolutional feature extractor between the two tasks.

```text
                    Facial Image
                         │
                         ▼
               Shared Visual Features
                    /           \
                   /             \
                  ▼               ▼
             Gender Task       Age Task
           Classification     Regression
```

This allows the model to learn a common representation while maintaining separate prediction heads for each task.

---

## 📊 Dataset

The project uses the **UTKFace dataset**.

The dataset contains facial images with metadata encoded in the filename:

```text
[age]_[gender]_[race]_[date].jpg
```

For example:

```text
25_0_2_20170116174525125.jpg
```

The relevant labels are extracted from the first two fields.

### Gender Mapping

```text
0 → Male
1 → Female
```

### Age

The dataset contains a wide range of ages, approximately:

```text
1 – 116 years
```

The dataset contains more than **23,000 facial images**.

---

## 🧠 Model Configuration

### Input

```text
128 × 128 × 3
```

### Convolutional Backbone

The CNN contains four convolutional stages:

```text
Conv2D(32)
     ↓
BatchNormalization
     ↓
ReLU
     ↓
MaxPooling

Conv2D(64)
     ↓
BatchNormalization
     ↓
ReLU
     ↓
MaxPooling

Conv2D(128)
     ↓
BatchNormalization
     ↓
ReLU
     ↓
MaxPooling

Conv2D(256)
     ↓
BatchNormalization
     ↓
ReLU
     ↓
MaxPooling
```

The extracted features are then passed to two independent fully connected branches.

### Gender Output

```python
Dense(1, activation="sigmoid")
```

Loss:

```text
Binary Crossentropy
```

Metric:

```text
Accuracy
```

### Age Output

```python
Dense(1, activation="linear")
```

Loss:

```text
Mean Squared Error
```

Metric:

```text
Mean Absolute Error
```

---

## 🔬 Training Pipeline

```text
UTKFace Dataset
       │
       ▼
Filename Parsing
       │
       ├──────────────► Age Labels
       │
       └──────────────► Gender Labels
                         │
                         ▼
                 Image Preprocessing
                         │
                         ▼
                    128 × 128 × 3
                         │
                         ▼
                  Train / Validation
                         │
                         ▼
                  Multi-Task CNN
                    /          \
                   /            \
                  ▼              ▼
             Gender Loss      Age Loss
                   \            /
                    \          /
                     ▼        ▼
                    Training
                         │
                         ▼
                   Saved Model
                         │
                         ▼
                  Streamlit App
```

---

## 🖼️ Image Preprocessing

Input images are:

1. Converted to RGB
2. Resized to `128 × 128`
3. Converted to NumPy arrays
4. Converted to `float32`
5. Normalized to `[0, 1]`
6. Expanded with a batch dimension

```python
image = image.convert("RGB")

image = image.resize(
    (128, 128),
    Image.Resampling.LANCZOS
)

image = np.array(image)

image = image.astype(
    np.float32
) / 255.0

image = np.expand_dims(
    image,
    axis=0
)
```

---

## 🌐 Streamlit Application

The trained model is integrated into a Streamlit interface.

The application pipeline is:

```text
Upload Image
      │
      ▼
RGB Conversion
      │
      ▼
Resize → 128 × 128
      │
      ▼
Normalize → [0,1]
      │
      ▼
CNN Inference
      │
      ├───────────────┐
      ▼               ▼
Gender Prediction   Age Prediction
      │               │
      ▼               ▼
Confidence          Age Value
      │               │
      └───────┬───────┘
              ▼
          UI Results
```

---

## 📁 Project Structure

```text
Age-Gender-AI/
│
├── app.py
│
├── training.ipynb
│
├── best_model/
│   └── age_gender_mobilenetv2.keras
│
├── demo.png
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

### Files

| File               | Description                    |
| ------------------ | ------------------------------ |
| `app.py`           | Streamlit application          |
| `training.ipynb`   | Model development and training |
| `best_model/`      | Saved trained model            |
| `demo.png`         | Application screenshot         |
| `requirements.txt` | Python dependencies            |
| `README.md`        | Project documentation          |

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Age-Gender-AI.git
```

Move into the project directory:

```bash
cd Age-Gender-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

Upload a face image and the model will return the estimated age and predicted gender.

---

## 📦 Requirements

Main technologies used in this project:

```text
Python
TensorFlow
Keras
NumPy
Pillow
Scikit-learn
Streamlit
```

---

## 📈 Example Output

```text
AI Prediction

Age:
19 years

Gender:
Male

Gender Confidence:
82.2%
```

The confidence score represents the model's confidence in its binary gender prediction.

---

## 🛠️ Technologies

| Technology   | Purpose              |
| ------------ | -------------------- |
| Python       | Programming          |
| TensorFlow   | Deep Learning        |
| Keras        | Model development    |
| NumPy        | Numerical processing |
| Pillow       | Image processing     |
| Scikit-learn | Dataset splitting    |
| Streamlit    | Web application      |

---

## 🔮 Future Improvements

Possible improvements for future versions include:

* Face detection before classification
* Webcam-based inference
* Better age estimation through improved loss design
* Data augmentation
* Transfer learning with a stronger pretrained backbone
* Model optimization for faster CPU inference
* Confidence estimation for age predictions
* Better handling of underrepresented age groups
* Deployment optimization

---

## ⚠️ Limitations

Age estimation is inherently more difficult than binary gender classification because age is a continuous variable and facial appearance varies significantly between individuals.

The dataset also contains an uneven distribution across age ranges, which can affect performance for less represented age groups.

Predictions should therefore be considered **model estimates**, not exact measurements.

---

## 📚 Learning Objectives

This project was developed to practice several core deep learning and computer vision concepts:

* Convolutional Neural Networks
* Multi-task Learning
* Image preprocessing
* Classification
* Regression
* Train/validation splitting
* Model evaluation
* TensorFlow / Keras
* Streamlit deployment
* Deep Learning inference pipelines

---

## 👤 Author

**Nastaran Karimi**

AI / Machine Learning Developer

Interested in:

```text
Machine Learning
Deep Learning
Computer Vision
Generative AI
AI Applications
```

---

## 📄 License

This project is intended for educational and portfolio purposes.

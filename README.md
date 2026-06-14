# 👤 AutoCluster
### AI Powered Photo Clustering Using Face Recognition and DBSCAN

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![DeepFace](https://img.shields.io/badge/DeepFace-Facenet512-green?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat&logo=streamlit)
![DBSCAN](https://img.shields.io/badge/ML-DBSCAN-orange?style=flat)

---

## 🚀 What is AutoCluster?

AutoCluster is an AI-powered photo organizer that automatically groups photos by person — just like Google Photos, but built from scratch using Python.

Upload any collection of photos and AutoCluster will:
- Detect every face automatically
- Convert faces into mathematical fingerprints
- Group similar faces together using machine learning
- Display results in a clean organized interface

---

## 🎯 Problem it Solves

People store thousands of photos but finding photos of a specific person is time consuming and frustrating. AutoCluster solves this by automatically organizing photos by person with zero manual effort.
Upload Photos

↓

Face Detection (MTCNN)

↓

Feature Extraction (Facenet512 → 512D embedding)

↓

DBSCAN Clustering (groups similar faces)

↓

Organized Results by Person

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **DeepFace** | Face detection and recognition |
| **Facenet512** | Face embedding model (512D vectors) |
| **MTCNN** | Face detector backend |
| **DBSCAN** | Unsupervised clustering algorithm |
| **Scikit-learn** | Machine learning implementation |
| **Streamlit** | Web interface |
| **NumPy** | Numerical computations |

---

## 🧠 Key Concepts

### Face Embeddings
Every face is converted into 512 numbers that mathematically represent facial features like eye distance, nose shape, and jawline. Similar faces produce similar numbers.

### Why DBSCAN over KMeans?
- No need to specify number of people beforehand
- Handles noise and outliers automatically
- Works perfectly for unknown group sizes

### Cosine Similarity
Used instead of Euclidean distance for better face comparison — measures angle between face vectors, not straight line distance.

---

## 📁 Project Structure

AutoCluster/

│

├── app.py          → Main Streamlit application

├── uploaded/       → Temporary photo storage

└── README.md       → Project documentation

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/jyothika-ds/AutoCluster.git
cd AutoCluster
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install deepface streamlit scikit-learn opencv-python numpy pandas tf-keras
pip install protobuf==6.31.1
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📊 Results

- ✅ Detects faces in JPG, JPEG, PNG formats
- ✅ Groups photos of same person automatically
- ✅ Handles multiple people in same photo collection
- ✅ Marks unrecognized/blurry photos separately

---

## ⚠️ Limitations

- Struggles with black and white photos
- Very blurry or far away faces may not be detected
- Visually similar people may occasionally be grouped together
- Processing time increases with number of photos

---

## 🔮 Future Enhancements

- [ ] Cloud storage integration (Google Drive, AWS S3)
- [ ] Real time clustering as photos are uploaded
- [ ] Mobile application (Android/iOS)
- [ ] Smart search — "Show all photos of John"
- [ ] Export clusters as ZIP folders

---

## 👩‍💻 Developer

**Jyothika Suresh**
- 📍 Bengaluru, India
- 💼 Aspiring AIML Engineer
- 🔗 [LinkedIn](https://linkedin.com/in/jyothika-suresh-78770a2a8)
- 🐙 [GitHub](https://github.com/jyothika-ds)

---

## ⭐ If you found this useful, please star the repo!
---

## ⚙️ How it Works

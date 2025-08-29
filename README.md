## 📌**SafeSpace: Digital Behaviour Monitor**  

## 📖 Overview and Purpose  

The rise of online platforms has created new avenues for expression but also increased the risk of **cyberbullying**, which negatively impacts mental health and social well-being.  
SafeSpace is a **machine learning and NLP-based framework** designed to detect cyberbullying behaviour in **real-time**, particularly focusing on **Hinglish (Hindi + English)** text.  

The purpose of this project is to:  
- Detect and classify online bullying behaviour in digital conversations.  
- Provide **instant feedback** to discourage harmful content.  
- Contribute towards creating **safe, inclusive, and empathetic online environments**.  

![Demo](https://github.com/VaishnavThorwat/SafeSpace__Digital_Behaviour_Monitor/blob/main/Assets/Screenshot%202025-08-29%20180927.png)
---

## 👨‍💻 Contributor's  

- [@Vaishnav Thorwat](https://github.com/VaishnavThorwat)
- [@Vedant Tanpure](https://github.com/VRT47pro)
- [@Vishal Maurya](https://github.com/VishalMaurya7)
- [@Akash Didwagh](https://github.com/Akashdidwagh)

---

## 📂 Data Sources  

- **Hinglish Bullying Detection Dataset** (public repository)  
  - Contains text entries labeled as *bullying* or *non-bullying*.  
  - Includes multilingual content with colloquial Hinglish phrases.  
- Additional preprocessing resources:  
  - `stopwords.txt` – custom stopword list  
  - `final_dataset_hinglish.csv` – cleaned dataset used in training  

---

## ⚙️ Methodology  

The methodology followed in this project consists of:  

1. **Data Cleaning & Preprocessing**  
   - Removal of special characters, punctuation, extra spaces  
   - Stopword removal  
   - Tokenization using TensorFlow’s Keras Tokenizer  
   - Padding sequences to a maximum length of 200 tokens  

2. **Model Architecture** – Hybrid CNN-BiLSTM  
   - **Embedding Layer** → Converts words into 128-dimensional vectors  
   - **CNN Layer** → Extracts local features from text  
   - **Max Pooling** → Reduces dimensionality  
   - **BiLSTM Layer** → Captures long-term sequential dependencies  
   - **Dense + Softmax Layer** → Classifies as *bullying* or *non-bullying* 
   
![Untitled diagram-2024-10-08-203246.png](https://github.com/VaishnavThorwat/SafeSpace__Digital_Behaviour_Monitor/blob/main/Assets/Untitled%20diagram-2024-10-08-203246.png)

3. **Training Setup**  
   - Optimizer: Adam (learning rate = 0.001)  
   - Loss: Sparse categorical crossentropy  
   - Epochs: 10 | Batch Size: 32  

4. **Evaluation Metrics**  
   - Accuracy, Precision, Recall, F1-Score  

---

## 📊 Results and Visualizations  

- **Training Accuracy**: 99.66%  
- **Validation Accuracy**: 91.02%  
- **Precision**: 91.45%  
- **Recall**: 90.65%  
- **F1-Score**: 91.05%  
  
![./Assets/Screenshot 2025-08-29180927.png](https://github.com/VaishnavThorwat/SafeSpace__Digital_Behaviour_Monitor/blob/main/Assets/Screenshot%202025-03-28%20112254.png)

These results show that the model is **highly effective** in detecting bullying behaviour while minimizing false positives.
### Key Insights  
- The CNN-BiLSTM hybrid model is effective in handling **multilingual and informal Hinglish text**.  
- Most misclassifications occurred with **ambiguous or idiomatic expressions**.  

### System Architecture

![Untitled diagram-2024-10-08-211800.png](https://github.com/VaishnavThorwat/SafeSpace__Digital_Behaviour_Monitor/blob/main/Assets/Untitled%20diagram-2024-10-08-211800.png)

---
## 💻 Installation and Setup  

### Prerequisites  
- Python 3.8+  
- Jupyter Notebook  
- Libraries: `numpy`, `pandas`, `scikit-learn`, `tensorflow/keras`, `nltk`  

### Setup Instructions  

```bash
# Clone the repository
git clone https://github.com/VaishnavThorwat/SafeSpace__Digital_Behaviour_Monitor.git
cd SafeSpace__Digital_Behaviour_Monitor

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

---

## ▶️ Usage  

- **Train or retrain the model**: Run `SafeSpaceModel.ipynb`  
- **Simulate client-side inputs**: Run `Client.ipynb`  
- **Run server-side classification**: Run `Server.ipynb`  
- Pretrained model (`CNNBILSTM.pkl`) and TF-IDF vocabulary are provided for direct inference.  

---

## ⚠️ Limitations and Future Work  

### Limitations  
- Performance decreases with **ambiguous or idiomatic Hinglish expressions**.  
- Limited to Hinglish dataset – multilingual scalability needs improvement.  

### Future Enhancements  
- Integrating **transformer models (BERT, DistilBERT)** for better contextual handling.  
- Extending to **regional languages** (Marathi, Bengali, etc.).  
- Developing a **real-time dashboard** with visualization of bullying patterns.  
- Embedding **ethical safeguards** – data anonymization, privacy-first architecture.  

---

## 📎 Optional Information  

- **File Structure**  
  - `SafeSpaceModel.ipynb` – Model training & evaluation  
  - `Client.ipynb` – Simulated client interactions  
  - `Server.ipynb` – Backend inference pipeline  
  - `CNNBILSTM.pkl` – Pre-trained model  
  - `tfidf_vector_vocabulary.pkl` – Vocabulary file  
  - `final_dataset_hinglish.csv` – Training dataset  
  - `stopwords.txt` – Custom stopword list  

- **Research Paper**  
  - Title: *SafeSpace: Digital Behaviour Monitor*  
  - Published: Terna Engineering College, 2025  

---

## 📜 License  

Licensed under the **MIT License**. See [LICENSE](LICENSE) for details.  

---

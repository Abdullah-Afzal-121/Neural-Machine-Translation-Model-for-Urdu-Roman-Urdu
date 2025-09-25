# Urdu → Roman Urdu Neural Machine Translation (NMT)

This project implements a **Neural Machine Translation (NMT) system** to convert **Urdu sentences** into **Roman Urdu** using a **Seq2Seq (BiLSTM Encoder–Decoder)** model.  
It includes training scripts, evaluation notebooks, and a **Streamlit web app** for interactive translation.

---

## 🚀 Features
- Seq2Seq architecture with **BiLSTM encoder and decoder**
- **BPE Tokenization** for Urdu and Roman Urdu text
- Support for GPU/CPU training and inference
- Pretrained model checkpoint loading
- **Streamlit UI** for real-time Urdu → Roman Urdu translation
- Deployable on **Google Colab** with public access (via LocalTunnel/Ngrok)

---

## 📂 Project Structure
urdu-roman-nmt/
│── notebooks/ # Jupyter/Colab notebooks for training & testing
│── app.py # Streamlit translation app
│── untitled7.py # Core model + training code
│── phase4_model_and_data.pth # Saved model checkpoint (not included in repo)
│── requirements.txt # Dependencies
│── README.md # Project documentation

---

## ⚙️ Installation

Clone the repository:
```bash
git clone https://github.com/your-username/urdu-roman-nmt.git
cd urdu-roman-nmt
pip install -r requirements.txt
streamlit run app.py
!pip install streamlit
!npm install -g localtunnel
!streamlit run app.py & npx localtunnel --port 8501
🖥️ Streamlit Interface

Enter Urdu text in the input box

Click Translate

Get Roman Urdu translation instantly 🎉

📊 Model Details

Encoder: BiLSTM with 2 layers

Decoder: LSTM with 4 layers

Embedding Dimension: 256

Hidden Dimension: 512

Dropout: 0.3

Training: CrossEntropy Loss + Adam Optimizer

🤝 Contributing

Pull requests are welcome!
If you find issues, please open an Issue.

📜 License

This project is licensed under the MIT License.

---

👉 Next Step: Do you want me to also generate a **requirements.txt** file (with `torch`, `streamlit`, `sentencepiece` etc.) so your repo is fully ready to push?

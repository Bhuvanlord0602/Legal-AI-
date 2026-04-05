# ⚖️ Kannada Legal AI System

🚀 NLP-powered Legal Document Analyzer for Kannada Language  

---

## 📌 Overview
Kannada Legal AI System is an end-to-end NLP application that processes Kannada legal documents and converts them into structured insights such as summaries, entities, and legal explanations.

This project combines:
- Classical NLP (tokenization, TF-IDF)
- Machine Learning
- Large Language Models (LLMs)

---

## ✨ Features
- Upload legal documents (PDF / Image)
- Extract Kannada text using OCR
- Clean and normalize text
- Custom Kannada tokenizer (character-level)
- Encode text into numerical form
- TF-IDF feature extraction
- Entity extraction (dates, key info)
- Automatic summarization
- AI-powered legal explanation
- Interactive UI using Streamlit

---

## 🧠 NLP Pipeline
Document Upload → Text Extraction → Preprocessing → Tokenization → Encoding → Feature Extraction → Model → LLM → Output

---

## 📂 Project Structure
kannada_legal_ai/
│
├── app.py
├── tokenizer.py
├── preprocessing.py
├── feature_extraction.py
├── model.py
├── utils.py
├── llm.py
├── requirements.txt
└── data/

---

## ⚙️ Installation
1. Clone Repository  
git clone https://github.com/Bhuvanlord0602/Legal-AI-.git  
cd Legal-AI-  

2. Install Dependencies  
pip install -r requirements.txt  

3. Install Tesseract OCR (Kannada)  
sudo apt install tesseract-ocr-kan  

---

## 🔑 API Setup (Gemini)
1. Go to: https://aistudio.google.com/  
2. Generate API key  
3. Add in llm.py:  
genai.configure(api_key="YOUR_API_KEY")  

---

## ▶️ Run the Application
streamlit run app.py  

---

## 🖥️ Application Workflow
Upload → Extract → Clean → Tokenize → Encode → TF-IDF → Model → Explain → Output  

---

## 🧪 Tech Stack
Python | Streamlit | Scikit-learn | Tesseract OCR | NLP | Gemini API  

---

## 🧠 Key Innovations
- Custom Kannada tokenizer (handles matras & modifiers)  
- Hybrid system: Classical NLP + LLM reasoning  
- Focus on low-resource language (Kannada)  
- Transparent pipeline (step-by-step output)  

---

## 🚀 Future Enhancements
- IndicBERT / MuRIL integration  
- Legal clause classification  
- RAG-based chatbot  
- Voice input (Kannada speech-to-text)  
- Cloud deployment  

---

## ⚠️ Disclaimer
This project is for educational purposes only and should not be considered legal advice.  

---

## 👨‍💻 Author
Bhuvan M  

---

## 🌟 Support
⭐ Star | 🍴 Fork | 📢 Share  

---

## 💡 Inspiration
Using AI to make legal systems more accessible and understandable.

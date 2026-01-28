## Clinical Decision Support System (CDSS) 🏥🤖 

During my research, I faced a huge challenge: analyzing patient data to predict disease severity and ICU admission risk was slow, complicated, and often lacked transparency. I wanted a solution that was fast, accurate, and explainable.

So, I built this Clinical Decision Support System (CDSS),  a Python-powered AI tool that helps healthcare professionals make data-driven decisions effortlessly.

## 🚀 What It Does

Predicts disease severity from patient data

Estimates ICU admission risk

Shows why each prediction was made using SHAP explainability

Interactive web interface built with Streamlit

Think of it as your AI co-pilot in healthcare, turning complex data into clear insights.

## 💡 Features

Explainable AI: SHAP shows exactly which factors influenced predictions

Fast & efficient: Make predictions in seconds

Modular design: Easily adapt models for new datasets or diseases

Open-source: Explore, experiment, and improve

## 🛠 Installation

Clone the repo:

git clone https://github.com/IrfanaAslam/clinical-decision-support-system.git
cd clinical-decision-support-system


Create a virtual environment:

python -m venv venv


Activate it:

Windows:

.\venv\Scripts\activate


macOS/Linux:

source venv/bin/activate


Install dependencies:

pip install --upgrade pip
pip install -r requirements.txt

## 🎯 Usage

Run the Streamlit app:

streamlit run app.py


Open the URL provided in your browser, input patient data, and get predictions instantly, along with an explanation of why the AI thinks this patient may need ICU care or has severe disease.

## 📂 File Structure
├── app.py                 # Streamlit app
├── inference.py           # Prediction logic
├── model.py               # Load & preprocess models
├── models/                # Trained ML models & encoders
├── notebooks/             # Experimentation notebooks
├── scripts/               # Training & prediction scripts
├── transforms.py          # Data preprocessing
└── requirements.txt       # Python dependencies

## 🌟 Benefits

Saves time: Quick predictions from patient data

Transparent: Understand AI decisions with SHAP

Scalable: Can adapt to other diseases or datasets

Open-source: Perfect for research, learning, and collaboration

## 👩‍💻 About Me

Hi! I’m Irfana Aslam, a researcher and developer passionate about AI in healthcare. I build tools that combine machine learning, data science, and practical applications to solve real-world problems.

I’m always open to collaborations, feedback, and exciting research opportunities.

## 📫 Reach me at:

Email: irfanaaslam69@gmail.com

LinkedIn: www.linkedin.com/in/irfana-aslam-b26751176

## 🤝 Contributing

I welcome contributions! Fork the repo, experiment, and submit a PR. Your ideas and improvements are valuable.

## 📜 License

MIT License ,  see LICENSE for details.

## 🙏 Acknowledgements

Python & Streamlit

SHAP for explainable AI

Scikit-learn for ML models



# 🤖 Multi-Agent Data Analyst

A **Multi-Agent Data Analysis System** with **LLM-powered agents**, running fully **locally** with `Mistral 7B Instruct v0.3` — no API required, 100% on-device! 🚀

---

## ✨ Features

✅ Modular **multi-agent pipeline**:
- Data Ingestion
- Preprocessing
- EDA
- Modeling
- Evaluation
- LLM Reporting

✅ **LLM Agents** run locally using:
- `llama-cpp-python`
- `Mistral 7B Instruct v0.3` GGUF

✅ Interactive **Streamlit UI**  
✅ Full CLI support  
✅ Runs entirely on **Mac Intel i5 + 16 GB RAM**

---

## 🗂 Project Structure
```text
multi_agent_data_analyst/
├── agents/ # LLM agents
├── orchestrator/ # Pipeline orchestrator
├── docs/ # Input CSV files
├── models/ # Local GGUF models (gitignored)
├── outputs/ # Outputs / reports
├── app.py # Streamlit app
├── run_pipeline.py # CLI runner
├── requirements.txt # Dependencies
├── .gitignore # Git ignore config
└── README.md # This file
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/multi_agent_data_analyst.git
cd multi_agent_data_analyst
```
### 2️⃣ Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### 3️⃣ Add Local Model

Place your Mistral 7B Instruct v0.3 GGUF model in:
```bash
models/mistral-7b-instruct-v0.3.Q5_K_M.gguf
```
## 🚀 Running the App

### 🖥️ Run Streamlit UI
```bash
streamlit run app.py
```
### ⚙️ Run CLI Pipeline
```bash
python3 run_pipeline.py
```

## ⚙️ Dependencies

See requirements.txt:

llama-cpp-python==0.2.64
pandas
numpy
scikit-learn
matplotlib
seaborn
ydata-profiling
streamlit
python-dotenv

## 🤖 Agent Architecture

| Agent                 | Purpose                         |
| --------------------- | ------------------------------- |
| PreprocessingLLMAgent | Suggest preprocessing steps     |
| ModelingLLMAgent      | Suggest model & hyperparameters |
| ReportingLLMAgent     | Generate final report           |

LLM Backed:

llama-cpp-python + Mistral 7B Instruct v0.3

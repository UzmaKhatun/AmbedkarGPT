# 📜 AmbedkarGPT - AI RAG System
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ambedkar-gpt.streamlit.app/)

Welcome! This is a **Retrieval-Augmented Generation (RAG)** system designed to answer questions based on the speeches and writings of Dr. B.R. Ambedkar. 

Instead of making up answers, this AI looks up specific text from provided documents, reads them, and creates an accurate answer for you.

---

## 📂 Project Structure (What are these files?)

Here is a simple breakdown of the files you see in the folder:

### **1. The Core Logic (`src/` folder)**
*   **`ingest_data.py`**: The "Librarian." It reads the text files, chops them into small pieces (chunks), turns them into numbers (embeddings), and saves them in a database.
*   **`rag_engine.py`**: The "Brain." It connects the database to the LLM (Llama-3 via Groq). It handles the logic of taking your question, finding the right text, and generating an answer.
*   **`metrics.py`**: The "Teacher." It contains all the math formulas (ROUGE, BLEU, Cosine) used to grade how smart the AI is.

### **2. The Interfaces (How you use it)**
*   **`app.py`**: The **Web App**. Run this to see a beautiful chat interface in your browser (using Streamlit).
*   **`main.py`**: The **Command Line Tool**. A simple version that runs in your black terminal window.

### **3. The Setup & Evaluation**
*   **`data_setup.py`**: The "Generator." You run this **first**. It creates the `data/` folder with the speeches and the `test_dataset.json` file from the assignment PDF.
*   **`evaluation.py`**: The "Exam." This script runs a scientific experiment. It tests the AI against 25 distinct questions, tries different chunk sizes, and saves the scores.
*   **`test_dataset.json`**: The list of 25 questions and correct answers used for testing.
*   **`test_results.json`**: The raw scores generated after running the evaluation.

### **4. Configuration**
*   **`requirements.txt`**: A list of all the Python tools (libraries) this project needs.
*   **`.env`**: A secret file where we keep the API Key.

---

## 🚀 How to Setup and Run

Follow these steps exactly, and it will work!

### Step 1: Create the Environment (The Venv)
We use a virtual environment so this project doesn't mess with your other Python projects.

**For Windows:**
```bash
# 1. Create the venv
python -m venv venv

# 2. Activate it (You will see (venv) appear in your terminal)
venv\Scripts\activate
```
**For Mac/Linux:**
```
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
Now that the venv is active, install the required tools.
```
pip install -r requirements.txt
```

### Step 3: Set API Key
Create a file named .env and paste your Groq API key inside:
```
GROQ_API_KEY=gsk_your_actual_api_key_here
```

### Step 4: Prepare Data
Run this script to generate the text files and test data.
```
python data_setup.py
```

### Step 5: Load the "Brain"
Run this to read the files and build the database.
```
python -m src.ingest_data
```

## 🎮 How to Use
Option A: The Web App (Recommended)
```
streamlit run app.py
```

Option B: The Command Line
```
python main.py
```

## 🧪 How to Run the Evaluation
To grade the system and generate the test_results.json file:
```
python evaluation.py
```

Note: This might take 2-3 minutes as it rebuilds the database 3 times to test different strategies.

---

## 👤 Author & Contact

**Developed by:** Uzma Khatun  
**Project Type:** RAG (Retrieval-Augmented Generation) Prototype

*   📧 **Email:** uzmakhatun0205@gmail.com
*   🔗 **LinkedIn:** https://www.linkedin.com/in/uzma-khatun-88b990334/
*   🐙 **GitHub:** https://github.com/UzmaKhatun

*Open to feedback and collaboration. Feel free to reach out!*

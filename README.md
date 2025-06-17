# Book Acquisitions Workflow Automation

### Project Overview

This project automates the book acquisitions workflow for donated books using computer vision and AI. It extracts metadata from book cover images, verifies against library holdings, and generates complete catalog records including ISBNs and call numbers.

---

## Setup Instructions (For Windows) Using CMD or Powershell to work on the project
### 1. Clone the Repository

If you're starting from a Git repo:

```bash
git clone https://github.com/GM-Sniper/Book_Acquisitions_Automation
cd Book_Acquisitions_Automation
```
---

###  2. Create and Activate a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

You should now see `(venv)` in your terminal.

---

### 3. Install Required Packages when you are inside the venv on powershell

```bash
pip install -r requirements.txt --upgrade
```
---

### 4. Run the Streamlit App

Make sure you are in the project root directory and your virtual environment is activated. Then run:

```bash
streamlit run src/UI/app.py
```

This will launch the web app in your browser for uploading and preprocessing book cover images.

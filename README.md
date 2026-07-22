# Healthcare-AI-Dashboard

An interactive prototype dashboard for procurement analytics and visualization developed during my Clinical Engineering internship

> **Note:** This repository contains a demonstration version of the project. All confidential procurement data from my internship has been removed and replaced with sample data. Certain features, including AI integration and outlier detection, are prototype implementations and remain areas for future development.

---

## Project Overview

Clinical engineering departments manage large amounts of procurement data for medical equipment and supplies. Analyzing this information manually can make it difficult to identify purchasing trends, unusual spending patterns, and opportunities for optimization.

This project explores AI can assist in this process by providing an interactive dashboard that allows users to:

- Analyze procurement spending
- Visualize purchasing trends
- Compare suppliers and cost centers
- Generate AI-assisted summaries

---

## Features

- 📊 Interactive Streamlit dashboard
- 🧹 Automated data cleaning pipeline
- 📈 Procurement spending visualizations
- 🏢 Supplier and cost center analysis
- 📋 Executive reporting
- 🤖 Prototype Claude AI integration
- 🚨 Prototype statistical outlier detection

---

## Project Structure

```text
Healthcare-AI-Dashboard/
│
├── app.py                 # Streamlit dashboard
├── data.py                # Data loading
├── cleanData.py           # Data cleaning pipeline
├── analyzeData.py         # Procurement analytics
├── graph.py               # Dashboard visualizations
├── outlierDetect.py       # Statistical outlier detection
├── report.py              # AI-generated reports
│
├── dataset/               # Sample demonstration dataset
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Anthropic Claude API

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Sanyu24/Clinical-Engineering-AI-Dashboard.git
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Current Status

This project is an active prototype developed as part of my internship.

The dashboard, analytics and visualization pipeline are functional. AI-assisted reporting and statistical outlier detection have been implemented as prototypes and are continuing to be refined.


---

## Acknowledgements

This project was inspired by my Clinical Engineering internship, where I explored how software engineering, data analytics and artificial intelligence can support engineering workflows and decision making. 

---

## Author

**Sanyu Karathody**

Electrical & Computer Systems Engineering Student  
Rensselaer Polytechnic Institute

GitHub: https://github.com/Sanyu24
LinkedIn: https://www.linkedin.com/in/sanyu-karathody

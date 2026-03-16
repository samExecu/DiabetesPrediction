# Diabetes Prediction App

A simple web app that takes some basic health information about a person and tells you whether they might have diabetes or not. It was built using Python and Streamlit.

---

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

---

## About

This app was made as a course project to practice data analysis and machine learning. Think of it like this, you feed the app some numbers (like your blood sugar level or body weight), and the app uses patterns it learned from a dataset to make a guess.

It was built to help us understand how data analysis and machine learning work in practice, not to be used as a real medical tool.

> **Disclaimer:** The data used to train this app is not reliable enough for real medical tests. The results can be wrong. Do not use this app to make any health decisions. Always talk to a real doctor.

---

## Installation

You will need Python installed on your computer. If you do not have it, download it from [python.org](https://www.python.org/).

**Step 1 — Get the code**

Download this project to your computer by running this in your terminal:

```bash
git clone https://github.com/samExecu/DiabetesPrediction.git
cd DiabetesPrediction
```

**Step 2 — Install the dependencies**

The app needs some extra tools to run. Install them all at once with this command:

```bash
pip install -r requirements.txt
```

This command goes through that list and installs everything automatically, so you do not have to do it one by one.

---

## Usage

Once everything is installed, start the app by running:

```bash
streamlit run Home.py
```

After running that command, the app will open in your web browser. There will be a Menu on the left where you can switch between pages, Just go to the Prediction Page and just fill in the health details it asks for and hit the predict button. It will then tell you its best guess.

---

## License

This project is licensed under the [MIT License](LICENSE).

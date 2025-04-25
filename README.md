# Automl-Leaderboard-App
🏆 AutoML Leaderboard Studio
Upload any tabular CSV, click Train Leaderboard, and instantly compare several “out-of-the-box” classifiers—no notebooks, no code.

🚀 What it does

Step	Action
1	Upload a CSV & choose a binary target column.
2	The app auto-detects numeric vs. categorical features and builds a common preprocessing pipeline (impute ▸ scale ▸ one-hot).
3	Trains four popular models
• Logistic Regression
• Gradient Boosting
• Random Forest
• LightGBM (if available)
4	Scores each model on Accuracy, F1, ROC-AUC and shows a sortable leaderboard.
5	Displays a SHAP bar chart of the champion model’s global feature importance.
6	Lets you download the champion model.pkl and the scored CSV with predicted probabilities.
Proof-of-concept – one train/valid split, no hyper-parameter search or fairness tests.
Need enterprise AutoML pipelines & experiment tracking? → drtomharty.com/bio

✨ Highlights
Single Python file (automl_leaderboard_app.py)

Works on CPU-only—ideal for Streamlit Cloud’s free tier

Transparent sklearn pipelines—no hidden magic

Optional LightGBM support (auto-skips if wheel unavailable)

🛠️ Requirements
nginx
Copy
Edit
streamlit
pandas
numpy
scikit-learn
lightgbm      # optional but recommended
shap
plotly
(All CPU wheels—no GPU needed.)

💻 Quick start (local)
bash
Copy
Edit
git clone https://github.com/THartyMBA/automl-leaderboard-studio.git
cd automl-leaderboard-studio
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run automl_leaderboard_app.py
Go to http://localhost:8501, upload a CSV, and explore.

☁️ Free deployment on Streamlit Cloud
Push the repo (public or private) to GitHub.

At streamlit.io/cloud, click New app → select repo/branch → Deploy.

Done—share your public URL!

🗂️ Repo layout
kotlin
Copy
Edit
automl_leaderboard_app.py   ← entire app
requirements.txt
README.md                    ← this file
📜 License
CC0 1.0 – public-domain dedication. Attribution appreciated but not required.

🙏 Acknowledgements
Streamlit – effortless data apps

scikit-learn – the ML backbone

LightGBM – gradient-boosting goodness

SHAP – model explainability made easy

Plotly – beautiful interactive charts

Build a leaderboard, pick a winner, ship the model—enjoy! 🎉

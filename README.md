# Malaria Severity Prediction API

Predicts whether a patient has **severe malaria** (1) or not (0) based on
age, sex, and 15 clinical symptoms. Built with scikit-learn + Flask, containerised
with Docker, and ready to deploy to Render or AWS.

---

## Model summary

| Item              | Detail                              |
|-------------------|-------------------------------------|
| Algorithm         | Gradient Boosting Classifier        |
| Features          | 17 (age, sex, 15 binary symptoms)   |
| Training samples  | 269 (+ SMOTE balancing)             |
| Test ROC-AUC      | 0.6464                              |
| Test Accuracy     | 61.8%                               |
| Class balance fix | SMOTE (inside cross-validation)     |

Top predictors by feature importance: `age`, `headace`, `Convulsion`,
`bitter_tongue`, `jundice`, `cocacola_urine`.

---

## Project structure

```
malaria-app/
├── train.py           # train & save the model
├── app.py             # Flask REST API
├── Dockerfile         # container recipe
├── requirements.txt   # Python dependencies
├── Malaria-Data.csv   # training data (not committed to git in prod)
└── model/
    ├── pipeline.joblib  # trained pipeline (scaler + SMOTE + model)
    └── features.joblib  # ordered feature name list
```

---

## Quickstart (local)

```bash
# 1. Create virtual environment
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python train.py

# 4. Start the API
python app.py
```

---

## API endpoints

### GET /health
Liveness check.
```json
{"status": "ok", "model_loaded": true}
```

### GET /info
Returns expected feature names and model description.

### POST /predict
Send a JSON body with all 17 features. Returns prediction + probability.

**Request:**
```json
{
  "age": 5,
  "sex": 1,
  "fever": 1,
  "cold": 1,
  "rigor": 1,
  "fatigue": 1,
  "headace": 1,
  "bitter_tongue": 0,
  "vomitting": 0,
  "diarrhea": 1,
  "Convulsion": 1,
  "Anemia": 0,
  "jundice": 0,
  "cocacola_urine": 1,
  "hypoglycemia": 1,
  "prostraction": 0,
  "hyperpyrexia": 1
}
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Severe Malaria",
  "probability_severe": 0.9114,
  "severity_risk": "HIGH"
}
```

`severity_risk` levels: `LOW` (< 0.40) · `MEDIUM` (0.40–0.70) · `HIGH` (≥ 0.70)

### GET /stats
Live request counts since server start.

---

## Docker (local container)

```bash
docker build -t malaria-app .
docker run -p {desired-port}:{desired-port} malaria-app
```

---

## Deploy to Render (free tier)

```bash
git init && git add . && git commit -m "initial deploy"
git remote add origin https://github.com/YOUR_USERNAME/malaria-app.git
git push -u origin main
```

1. Go to **render.com** → New + → Web Service
2. Connect your GitHub repo
3. Environment: **Docker**
4. Instance type: **Free**
5. Click **Create Web Service**

Render builds and deploys automatically. Your live URL will be:
`https://malaria-app.onrender.com`

---

## Deploy to AWS (App Runner)

```bash
# Configure AWS CLI
aws configure

# Push image to ECR
aws ecr create-repository --repository-name malaria-app
docker tag malaria-app:latest <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/malaria-app
docker push <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/malaria-app
```

Then in the AWS Console: **App Runner → Create service → Container registry → ECR**.

---

## Feature encoding

| Feature         | Values             |
|-----------------|--------------------|
| age             | numeric (years)    |
| sex             | 0 = female, 1 = male |
| All symptoms    | 0 = absent, 1 = present |

Symptoms: `fever`, `cold`, `rigor`, `fatigue`, `headace`, `bitter_tongue`,
`vomitting`, `diarrhea`, `Convulsion`, `Anemia`, `jundice`, `cocacola_urine`,
`hypoglycemia`, `prostraction`, `hyperpyrexia`

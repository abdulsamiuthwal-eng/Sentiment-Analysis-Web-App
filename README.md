# AuraSentiment — End-to-End Sentiment Analysis Web Application

AuraSentiment is a premium, portfolio-grade Machine Learning based Sentiment Analysis application that classifies the sentiment of text (tweets, reviews, or general comments) into **Positive 😊**, **Neutral 😐**, or **Negative 😞** categories. 

The project is developed as part of the **DEVFORGE AI/ML Internship**.

### 🌐 Live Deployments
- **Live Frontend (Vercel):** [https://sentiment-analysis-web-app-gcgl.vercel.app/](https://sentiment-analysis-web-app-gcgl.vercel.app/)
- **Live Backend API (Render):** [https://sentiment-analysis-web-app-7odn.onrender.com/](https://sentiment-analysis-web-app-7odn.onrender.com/)
- **API Documentation (Swagger UI):** [https://sentiment-analysis-web-app-7odn.onrender.com/docs](https://sentiment-analysis-web-app-7odn.onrender.com/docs)
- **API Documentation (ReDoc):** [https://sentiment-analysis-web-app-7odn.onrender.com/redoc](https://sentiment-analysis-web-app-7odn.onrender.com/redoc)

---

## 🚀 Key Features
- **Exploratory Data Analysis (EDA):** Insights and statistical analysis of Twitter sentiment data.
- **Robust NLP Pipeline:** Step-by-step cleaning (URLs, punctuation, numbers, special characters removal) and advanced text tokenization, stopword pruning, and **WordNet Lemmatization**.
- **Model Evaluation Suite:** Evaluates and compares Logistic Regression, Multinomial Naive Bayes, and Random Forest models.
- **FastAPI Microservice:** Fast, production-ready REST API featuring health check, input validation, automatic OpenAPI documentation (Swagger/ReDoc), and dynamic model probability calculation (Confidence Scores).
- **Premium Glassmorphic UI:** A modern responsive frontend designed with tailored HSL color palettes, custom typography, backdrop blurs, and animated transition states.

---

## 📂 Project Structure

```text
├── dataset/
│   ├── twitter_training.csv      # Dataset used for training and model selection
│   └── twitter_validation.csv    # Unseen final validation dataset
├── training/
│   ├── train_model.ipynb        # Main documented training notebook
│   └── extend_notebook.py       # Helper script to programmatically extend the notebook
├── backend/
│   ├── models/
│   │   ├── model.pkl            # Serialized Logistic Regression Model
│   │   └── vectorizer.pkl       # Serialized TF-IDF Vectorizer
│   ├── main.py                  # Main FastAPI application router
│   ├── schemas.py               # Pydantic validation schemas
│   └── utils.py                 # Core text cleaning and preprocessing functions
├── frontend/
│   ├── index.html               # Main HTML UI structure
│   ├── style.css                # Premium responsive glassmorphic stylesheet
│   └── app.js                   # Client-side integration logic (Fetch API)
├── requirements.txt             # Unified Python dependencies
├── render.yaml                  # Render deployment configuration
├── vercel.json                  # Vercel deployment configuration
└── README.md                    # Project documentation
```

---

## 🛠️ Backend Setup & Run Instructions

1. **Activate the Virtual Environment:**
   - On Windows (PowerShell):
     ```powershell
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

2. **Install Dependencies:**
   Ensure all packages are installed:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI Server:**
   Launch the backend server using Uvicorn:
   ```bash
   uvicorn backend.main:app --reload
   ```
   The API will start running locally at: `http://127.0.0.1:8000`

---

## 🌐 API Documentation

FastAPI provides automatic interactive API documentations. You can view them at:
- **Swagger UI:** `http://127.0.0.1:8000/docs` (or live at [https://sentiment-analysis-web-app-7odn.onrender.com/docs](https://sentiment-analysis-web-app-7odn.onrender.com/docs))
- **ReDoc:** `http://127.0.0.1:8000/redoc` (or live at [https://sentiment-analysis-web-app-7odn.onrender.com/redoc](https://sentiment-analysis-web-app-7odn.onrender.com/redoc))

### Endpoints

#### 1. Health Check
- **URL:** `GET /`
- **Description:** Checks if the API is active.
- **Example Response:**
  ```json
  {
    "message": "Sentiment Analysis API Running"
  }
  ```

#### 2. Classify Sentiment
- **URL:** `POST /predict`
- **Description:** Preprocesses the input text, applies TF-IDF vectorization, and predicts its sentiment along with the model's confidence probability.
- **Content-Type:** `application/json`
- **Example Request Payload:**
  ```json
  {
    "text": "I love this internship!"
  }
  ```
- **Example Response Payload:**
  ```json
  {
    "text": "I love this internship!",
    "prediction": "Positive",
    "confidence": "98.91%"
  }
  ```

---

## 💻 Frontend Setup & Instructions

The frontend is built using clean, vanilla **HTML5**, **CSS3**, and **Vanilla JavaScript** (no external framework build step required).

1. **How to run the UI:**
   Simply open [frontend/index.html](file:///d:/DEVFORGE_INTERNSHIP/Sentiment-Analysis/frontend/index.html) directly in any web browser.
   - Alternatively, you can serve the frontend folder using any local web server extension (e.g., Live Server in VS Code) or Python's HTTP server:
     ```bash
     python -m http.server 8080 --directory frontend
     ```
     Then navigate to: `http://localhost:8080`

2. **Integration details:**
   The frontend uses the browser's native **Fetch API** to send requests asynchronously to the live backend hosted on Render: `https://sentiment-analysis-web-app-7odn.onrender.com/predict`. It handles loading states, error fallback displays, and dynamically applies positive (green), neutral (orange), and negative (red) UI styles based on the predicted category.

---

## 📸 Screenshots Placeholders

### 1. Splash Screen
<img width="1918" height="968" alt="SPLASH_T1" src="https://github.com/user-attachments/assets/195746c2-77b7-4af1-9394-e6a35551655f" />


### 2. 🏠 Home Screen
<img width="1898" height="972" alt="HOME" src="https://github.com/user-attachments/assets/03e9d5f5-b6d5-488f-8f85-767877abdfb5" />


### 3. 😊 Positive Prediction
<img width="1901" height="1075" alt="POSITIVE" src="https://github.com/user-attachments/assets/c647a8e9-041f-4e5c-a680-5d70ce945777" />



### 4. 😐 Neutral Prediction
<img width="1897" height="1078" alt="NEUTRAL" src="https://github.com/user-attachments/assets/f473f83c-61d1-46f2-91a1-1ba3e90b336c" />



### 5. 😞 Negative Prediction
<img width="1900" height="1077" alt="NEGATIVE" src="https://github.com/user-attachments/assets/993c15c4-af21-4249-bc3d-218af5b73831" />



---

## 🔮 Future Improvements
1. **Transformer Models:** Upgrade from Logistic Regression to a pre-trained Transformer (like DistilBERT) for superior semantic parsing.
2. **Emoji Mapping:** Parse and tokenize emojis to preserve their sentiment scores instead of discarding them.
3. **Database Integration:** Save history of user query predictions for analytics and active learning feedback loops.

import os
import joblib
import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import PredictionRequest, PredictionResponse
from backend.utils import preprocess_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Sentiment Analysis API",
    description="FastAPI backend serving a Logistic Regression model for Sentiment Analysis.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local frontend testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve model and vectorizer paths dynamically relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

# Load serialized ML pipeline components
try:
    logger.info(f"Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logger.info(f"Loading vectorizer from: {VECTORIZER_PATH}")
    vectorizer = joblib.load(VECTORIZER_PATH)
    logger.info("Model and vectorizer loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model/vectorizer artifacts: {str(e)}", exc_info=True)
    raise RuntimeError(f"ML artifacts load failure: {str(e)}")

@app.get("/", status_code=status.HTTP_200_OK)
def health_check():
    """
    API Health Check endpoint.
    """
    return {"message": "Sentiment Analysis API Running"}

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict(request: PredictionRequest):
    """
    Predicts the sentiment of the provided text (Positive, Neutral, or Negative)
    and returns a confidence score.
    """
    # Validation: Ensure text is not empty or whitespace only
    if not request.text or not request.text.strip():
        logger.warning("Empty or whitespace-only prediction request received.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text cannot be empty or whitespace only."
        )
    
    try:
        # Preprocess input text
        processed = preprocess_text(request.text)
        
        # Transform using loaded TF-IDF vectorizer
        vectorized = vectorizer.transform([processed])
        
        # Predict sentiment
        prediction = model.predict(vectorized)[0]
        
        # Calculate confidence score if the model supports probability estimation
        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vectorized)[0]
            classes = model.classes_
            if prediction in classes:
                pred_idx = list(classes).index(prediction)
                confidence_val = probs[pred_idx]
                confidence = f"{confidence_val * 100:.2f}%"
        
        logger.info(f"Input: '{request.text}' | Prediction: {prediction} | Confidence: {confidence}")
        
        return PredictionResponse(
            text=request.text,
            prediction=prediction,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction logic failure: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred during prediction processing: {str(e)}"
        )

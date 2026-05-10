ML Pipeline: NYC Airbnb Price Prediction
This project builds a complete machine learning pipeline to predict Airbnb listing prices in New York City. It includes data cleaning, feature engineering, model training, evaluation, slice metrics, model documentation, API deployment, and containerization. The goal is to demonstrate the full lifecycle of an ML model from raw data to a production-ready service.
The model is trained using a RandomForestRegressor and uses both categorical and numerical features. After training, the model is exported and served through a FastAPI application. The API exposes a /predict endpoint that accepts listing details and returns a price prediction.
This project demonstrates the following skills:
- Building an end-to-end ML pipeline
- Training and evaluating a supervised learning model
- Computing slice metrics for fairness and performance analysis
- Creating a model card for documentation
- Deploying a model using FastAPI
- Containerizing an application with Docker
- Writing clear instructions for running and testing the API
Project Structure
Project-Build-an-ML-Pipeline-Starter/ components/ train_random_forest/ model_export/model.pkl serve/ app.py requirements.txt Dockerfile README.md
Installation
Install dependencies: pip install -r requirements.txt
Model
The model uses the following features: neighbourhood_group room_type minimum_nights number_of_reviews reviews_per_month calculated_host_listings_count availability_365
The trained model is stored at: components/train_random_forest/model_export/model.pkl
Running the FastAPI App
From the project root: uvicorn serve.app:app --reload
Open the interactive API docs: http://localhost:8000/docs
Inside the Udacity workspace, use the proxy URL shown in the workspace.
Example Prediction Request
POST /predict
Body: { "neighbourhood_group": "Brooklyn", "room_type": "Private room", "minimum_nights": 2, "number_of_reviews": 15, "reviews_per_month": 1.5, "calculated_host_listings_count": 1, "availability_365": 120 }
Example Response: { "prediction": [73.16] }
Docker
Build the image: docker build -t ml_api .
Run the container: docker run -p 8000:8000 ml_api
Then open: http://localhost:8000/docs
How to Run Everything
- Install dependencies using pip install -r requirements.txt
- Start the FastAPI server using uvicorn serve.app:app --reload
- Open the API documentation in your browser
- Send a POST request to the /predict endpoint with the required fields
- (Optional) Build and run the Docker container using the commands above
- Use the containerized API the same way through the /docs interface

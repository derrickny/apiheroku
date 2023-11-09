Diabetes Prediction Web API

This project implements a web API for predicting diabetes risk based on a trained machine learning model. The model, a neural network, has been trained on a dataset containing information such as pregnancies, glucose levels, blood pressure, and more.

Features:

FastAPI Backend: Utilizes FastAPI, a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.
Scikit-Learn Model: The project incorporates a trained Scikit-Learn model for accurate diabetes predictions.
Model Deployment: A deployed version of the model ready to be integrated into web applications.
Swagger Documentation: The API comes with Swagger UI documentation for easy exploration and understanding of available endpoints.
Test Data: Includes sample test data for validating predictions.
How to Use:

Send a POST request to the /diabetes_prediction endpoint with input parameters in the request body.
Receive predictions indicating whether a person is diabetic or not.
Technologies Used:

FastAPI
Scikit-Learn
Python
Pandas
Numpy
Usage:

Clone the repository:
bash
Copy code
git clone https://github.com/your-username/diabetes-prediction-api.git
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Run the FastAPI application:
bash
Copy code
uvicorn main:app --reload
Access Swagger documentation:
Visit http://127.0.0.1:8000/docs in your browser.

Make predictions:
Use your preferred API testing tool or library to send POST requests to http://127.0.0.1:8000/diabetes_prediction with input parameters

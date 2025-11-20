# USED CAR PRICE PREDICTION

This project predicts used car prices using a machine learning model trained on real-world data web-scraped from Cars24. After cleaning and preprocessing the dataset, a regression model was built to estimate prices based on brand, model, year, kilometers driven, fuel type, and engine capacity. The application is developed using FastAPI with a clean HTML/CSS interface for entering car details and receiving instant predictions. The final model and app are deployed on Render, showcasing end-to-end skills in web scraping, ML modeling, API development, and cloud deployment.

### Live API Link : https://car-price-prediction-6-lnha.onrender.com

![AppImage](appImage.png)

### Data Collection (Web Scraping)
* Scraped real car listings using Selenium + BeautifulSoup.
* Extracted important attributes such as brand, model, year, kilometers driven, fuel type, transmission, engine power, location, and price.
* Cleaned and saved the dataset into a structured CSV for further processing.

### Machine Learning Pipeline
#### Built a complete regression workflow including:
##### Train/Test Split
##### Data preprocessing
##### OneHot encoding for categorical features
##### Feature scaling
##### Model training (Multiple Linear Regression)
##### Evaluation using R², MAE, MSE

### FastAPI Backend Development
Created REST API endpoints using FastAPI.
Built a /predict endpoint that takes car features and returns predicted price.
Integrated the model + pipeline into an ASGI application.

### Deployment
Deployed the FastAPI application on Render Cloud using:
Gunicorn/Uvicorn server
Proper folder structure for static/templates
Production configuration
App runs live and can be accessed publicly.





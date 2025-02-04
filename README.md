# ML Zoomcamp 2024 Capstone Project 2 - Car Insurance Regressor

A capstone project from [ml-zoomcamp](!http://mlzoomcamp.com/) 2024. 

The project showed the application of what we have learned in the course
1. Machine Learning Model: Regression. 
2. Deployment in an web application using Flask
3. Containerization using Docker


This project using the [Car Insurance Premium Dataset](!https://www.kaggle.com/datasets/govindaramsriram/car-insurance-premium-dataset) dataset which can be downloaded from [kaggle](!https://www.kaggle.com/datasets/govindaramsriram/car-insurance-premium-dataset).

The target variable is insurance premium.

We used this dataset to train a regression model for this capstone project. Based on the feature given, the model will predict how much insurance premium a customer should be charged.

Conda is used as virtual environment and the depedencies needed was provided in [requirement.txt](requirements.txt) file.
```bash
conda create --name ml-zoomcamp
conda activate ml-zoomcamp
pip install -r requirements.txt
```

We used Docker container and build an image from the [Dockerfile](Dockerfile)
```bash
docker build -t capstone-project2 .
```

Run the docker image using this command:
```bash
docker run -it --rm -p 9696:9696 capstone-project2
```

Flags explanations:
* `--name`: give the name to the created conda environment
* `-r`: installing pip packages listed in the requirements.txt file
* `-t`: specifying tag name
* `-it`: allowing access to the terminal inside Docker image
* `--rm`: removing the image from the system after we have exited the image
* `-p`: mapping the port `9696` in our machine to the port `9696` inside Docker image  

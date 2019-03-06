# Disaster Response Pipeline Project

The Disaster Response Pipeline allows to organize messages from social media in emergency categories that allows different organizations to respond request from people in need in a given emergency disaster event. It uses Natural Language Processing and Machine Learning Model to perform such categorizations.

### Scripts:
* **data/process_data.py**: This script loads raw data from cvs input files and clean, transform build a dataset that will be used by the ML Pipeline to run predictions. It implements Natural Language Processing strategies to transform the messages in a structured dataset that will serve as a training set for the ML Model.
* **models/train_classifier.py**: Builds the Machine Learning pipeline that will be used to predict the categories in which a given message belongs to.
* **models/test_model.py**: Ad hoc script to test the model and compare the results from the web implementation.
* **app/run.py**: Script for launch the web application. It also provides basic plotting information from the messages received and their classifications.
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    - To test the pipeline
        `python models/test_model.py`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Web Application:
#### Home
  ![Home](https://raw.githubusercontent.com/mrugeles/mrugeles.github.io/master/images/home.png)

#### Message categorization
  ![Predictions](https://raw.githubusercontent.com/mrugeles/mrugeles.github.io/master/images/prediction.png)

#### Dashboard
![Dashboard](https://raw.githubusercontent.com/mrugeles/mrugeles.github.io/master/images/dashboard.png)

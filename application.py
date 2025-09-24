# Importing the Required Libraries
import flask
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

# Initializing the Flask application
application = Flask(__name__)

# Loading trained model
model = load_model("models/model.h5")

# Compiling the trained model
model.compile()

# Mapping dictionaries for categorical variables
gender_map = {'Male': 1, 'Female': 0}
yes_no_map = {'Yes': 1, 'No': 0}
multiple_lines_map = {'Yes': 1, 'No': 0}
internet_map = {'No internet service': 0, 'DSL': 1, 'Fiber optic': 2}
contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
payment_map = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer': 2, 'Credit card': 3}

# Home page route
@application.route('/')
def home():
    return render_template('home.html')

# Prediction route
@application.route('/predict', methods=['POST'])
def predict():
    # Collecting form inputs
    gender = request.form['gender']
    senior_citizen = request.form['senior_citizen']
    partner = request.form['partner']
    dependents = request.form['dependents']
    tenure = float(request.form['tenure'])
    phone_service = request.form['phone_service']
    multiple_lines = request.form['multiple_lines']
    internet_service = request.form['internet_service']
    online_security = request.form['online_security']
    online_backup = request.form['online_backup']
    device_protection = request.form['device_protection']
    tech_support = request.form['tech_support']
    streaming_tv = request.form['streaming_tv']
    streaming_movies = request.form['streaming_movies']
    contract = request.form['contract']
    paperless_billing = request.form['paperless_billing']
    payment_method = request.form['payment_method']
    monthly_charges = float(request.form['monthly_charges'])
    total_charges = float(request.form['total_charges'])

    # Encoding categorical features
    input_features = [
        gender_map[gender], yes_no_map[senior_citizen], yes_no_map[partner], yes_no_map[dependents], tenure,
        yes_no_map[phone_service], multiple_lines_map[multiple_lines], internet_map[internet_service],
        yes_no_map[online_security], yes_no_map[online_backup], yes_no_map[device_protection], yes_no_map[tech_support],
        yes_no_map[streaming_tv], yes_no_map[streaming_movies], contract_map[contract],
        yes_no_map[paperless_billing], payment_map[payment_method], monthly_charges, total_charges
    ]

    # Converting to NumPy array
    input_data = np.array([input_features], dtype=np.float32)

    # Making prediction
    prediction = model.predict(input_data)
    churn_probability = float(prediction[0][0])

    # Determining churn status based on target variable {Yes, No}
    if churn_probability > 0.5:
        churn_status = "Churn: Yes (Customer is likely to churn)"
        next_step = "Consider offering retention discounts or personalized support."
    else:
        churn_status = "Churn: No (Customer is not likely to churn)"
        next_step = "Focus on maintaining service quality and engagement."

    # Rendering results page with all inputs (grouped for readability)
    return render_template(
        'results.html',
        gender=gender, senior_citizen=senior_citizen, partner=partner, dependents=dependents,
        tenure=tenure, phone_service=phone_service, multiple_lines=multiple_lines, internet_service=internet_service,
        online_security=online_security, online_backup=online_backup, device_protection=device_protection, tech_support=tech_support,
        streaming_tv=streaming_tv, streaming_movies=streaming_movies, contract=contract, paperless_billing=paperless_billing,
        payment_method=payment_method, monthly_charges=monthly_charges, total_charges=total_charges,
        churn_status=churn_status, next_step=next_step, prediction_value=f"{churn_probability:.2f}"
    )

# Running the app
if __name__ == '__main__':
    application.run(debug=True)
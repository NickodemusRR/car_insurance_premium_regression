import pickle
import pandas as pd

from flask import Flask, request, jsonify

model_file = "model.bin"
with open(model_file, "rb") as f_in:
    scaler_x, scaler_y, model = pickle.load(f_in)

app = Flask("insurance")

@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()

    cust_df = pd.DataFrame([customer])

    X = scaler_x.transform(cust_df)
    y_pred_scaler = model.predict(X)
    y_pred = scaler_y.inverse_transform(y_pred_scaler.reshape(-1, 1)).ravel()

    result = {
        "Premium": round(y_pred[0],3),
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)

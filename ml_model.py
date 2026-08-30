import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# --------------------------------
# 1. Load the dataset
# --------------------------------

df = pd.read_csv("payment_dataset.csv")

print("Dataset loaded successfully!")
print("Total rows:", len(df))


# --------------------------------
# 2. Convert text into numbers
# --------------------------------

encoder = LabelEncoder()

df["payment_method"] = encoder.fit_transform(df["payment_method"])
df["failure_reason"] = encoder.fit_transform(df["failure_reason"])
df["customer_type"] = encoder.fit_transform(df["customer_type"])


# --------------------------------
# 3. Select input features
# --------------------------------

X = df[
    [
        "amount",
        "payment_method",
        "failure_reason",
        "previous_transactions",
        "previous_failures",
        "retry_count",
        "customer_type"
    ]
]


# --------------------------------
# 4. Select target
# --------------------------------

y = df["recovered"].map({
    "Yes": 1,
    "No": 0
})


# --------------------------------
# 5. Split the data
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------
# 6. Create ML model
# --------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# --------------------------------
# 7. Train the model
# --------------------------------

model.fit(X_train, y_train)


# --------------------------------
# 8. Test the model
# --------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model trained successfully!")
print("Model Accuracy:", accuracy)


# --------------------------------
# 9. Test one new payment
# --------------------------------

new_payment = [[
    1499,   # amount
    3,      # payment method
    1,      # failure reason
    20,     # previous transactions
    1,      # previous failures
    0,      # retry count
    0       # customer type
]]

prediction = model.predict(new_payment)

probability = model.predict_proba(new_payment)

print("\nNew Payment Prediction:")

if prediction[0] == 1:
    print("Payment is likely to be recovered")
else:
    print("Payment is unlikely to be recovered")

print(
    "Recovery Probability:",
    round(probability[0][1] * 100, 2),
    "%"
)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# =====================================================
# 1. LOAD DATASET
# =====================================================

df = pd.read_csv("payment_dataset.csv")

print("Dataset loaded!")
print("Total transactions:", len(df))


# =====================================================
# 2. CONVERT TEXT DATA INTO NUMBERS
# =====================================================

payment_encoder = LabelEncoder()
failure_encoder = LabelEncoder()
customer_encoder = LabelEncoder()

df["payment_method"] = payment_encoder.fit_transform(
    df["payment_method"]
)

df["failure_reason"] = failure_encoder.fit_transform(
    df["failure_reason"]
)

df["customer_type"] = customer_encoder.fit_transform(
    df["customer_type"]
)


# =====================================================
# 3. SELECT FEATURES
# =====================================================

features = [
    "amount",
    "payment_method",
    "failure_reason",
    "previous_transactions",
    "previous_failures",
    "retry_count",
    "customer_type"
]

X = df[features]


# =====================================================
# 4. TARGET
# =====================================================

y = df["recovered"].map({
    "Yes": 1,
    "No": 0
})


# =====================================================
# 5. TRAIN RANDOM FOREST MODEL
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("ML model trained successfully!")


# =====================================================
# 6. ADVANCED AI RECOVERY AGENT
# =====================================================

MAX_RETRIES = 2


def recovery_agent(transaction, probability):

    amount = transaction["amount"]
    retry_count = transaction["retry_count"]
    failure_reason = transaction["failure_reason"]
    customer_type = transaction["customer_type"]

    # -------------------------------------------------
    # RULE 1: RETRY LIMIT
    # -------------------------------------------------

    if retry_count >= MAX_RETRIES:

        return {
            "action": "STOP",
            "reason": (
                "Maximum retry limit reached. "
                "Further retries are stopped to reduce "
                "customer friction."
            )
        }

    # -------------------------------------------------
    # RULE 2: HIGH PROBABILITY
    # -------------------------------------------------

    if probability >= 70:

        return {
            "action": "RETRY",
            "reason": (
                f"Recovery probability is {probability:.1f}%. "
                f"Payment amount is ₹{amount:.0f}. "
                f"Failure type is {failure_reason}. "
                "A limited retry is recommended."
            )
        }

    # -------------------------------------------------
    # RULE 3: MEDIUM PROBABILITY
    # -------------------------------------------------

    elif probability >= 40:

        return {
            "action": "PAYMENT LINK",
            "reason": (
                f"Recovery probability is {probability:.1f}%. "
                f"Customer type is {customer_type}. "
                "A payment link is preferred instead of "
                "another automatic retry."
            )
        }

    # -------------------------------------------------
    # RULE 4: LOW PROBABILITY
    # -------------------------------------------------

    else:

        return {
            "action": "STOP",
            "reason": (
                f"Recovery probability is only {probability:.1f}%. "
                "Another attempt may create unnecessary "
                "customer friction, so the agent stops."
            )
        }


# =====================================================
# 7. TEST THE ML + AI AGENT
# =====================================================

print("\n==============================================")
print("        ML + AI RECOVERY AGENT")
print("==============================================")


for index in range(10):

    transaction = df.iloc[index]

    # Get transaction features
    input_data = pd.DataFrame(
        [transaction[features]]
    )

    # -------------------------------------------------
    # ML PREDICTION
    # -------------------------------------------------

    probability = (
        model.predict_proba(input_data)[0][1]
        * 100
    )

    # -------------------------------------------------
    # AI AGENT DECISION
    # -------------------------------------------------

    decision = recovery_agent(
        transaction,
        probability
    )

    # -------------------------------------------------
    # DISPLAY
    # -------------------------------------------------

    print("\n----------------------------------------------")

    print(
        "Transaction:",
        transaction["transaction_id"]
    )

    print(
        "Amount: ₹",
        transaction["amount"]
    )

    print(
        "ML Recovery Probability:",
        round(probability, 2),
        "%"
    )

    print(
        "AI Action:",
        decision["action"]
    )

    print(
        "AI Explanation:",
        decision["reason"]
    )
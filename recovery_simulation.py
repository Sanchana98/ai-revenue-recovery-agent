import pandas as pd
import random

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("payment_dataset.csv")

print("Dataset loaded!")
print("Total transactions:", len(df))


# ==========================================
# SIMULATION SETTINGS
# ==========================================

MAX_RETRIES = 2


# ==========================================
# RECOVERY SIMULATION
# ==========================================

def simulate_recovery(transaction, probability, action):

    retry_count = transaction["retry_count"]

    # --------------------------------------
    # STOP ACTION
    # --------------------------------------

    if action == "STOP":

        return "NOT ATTEMPTED"


    # --------------------------------------
    # RETRY LIMIT
    # --------------------------------------

    if retry_count >= MAX_RETRIES:

        return "NOT ATTEMPTED"


    # --------------------------------------
    # SIMULATE RECOVERY
    # --------------------------------------

    random_number = random.randint(1, 100)

    if random_number <= probability:

        return "SUCCESS"

    else:

        return "FAILED"


# ==========================================
# SIMULATE ALL TRANSACTIONS
# ==========================================

results = []

for index, transaction in df.iterrows():

    # --------------------------------------
    # SIMPLE ML-LIKE RECOVERY SCORE
    # --------------------------------------

    probability = 50

    if transaction["previous_transactions"] >= 10:
        probability += 15

    if transaction["previous_failures"] <= 1:
        probability += 15

    if transaction["retry_count"] == 0:
        probability += 10

    if transaction["failure_reason"] == "Bank Timeout":
        probability += 10

    if transaction["failure_reason"] == "Insufficient Balance":
        probability -= 20

    if transaction["failure_reason"] == "Card Declined":
        probability -= 15


    # Keep probability between 0 and 100

    probability = max(0, min(100, probability))


    # --------------------------------------
    # AI AGENT DECISION
    # --------------------------------------

    if transaction["retry_count"] >= MAX_RETRIES:

        action = "STOP"

    elif probability >= 70:

        action = "RETRY"

    elif probability >= 40:

        action = "PAYMENT LINK"

    else:

        action = "STOP"


    # --------------------------------------
    # SIMULATE RECOVERY
    # --------------------------------------

    result = simulate_recovery(
        transaction,
        probability,
        action
    )


    # --------------------------------------
    # REVENUE RECOVERED
    # --------------------------------------

    if result == "SUCCESS":

        recovered_amount = transaction["amount"]

    else:

        recovered_amount = 0


    results.append([
        transaction["transaction_id"],
        transaction["amount"],
        transaction["failure_reason"],
        transaction["retry_count"],
        round(probability, 2),
        action,
        result,
        recovered_amount
    ])


# ==========================================
# CREATE RESULTS DATAFRAME
# ==========================================

results_df = pd.DataFrame(
    results,
    columns=[
        "transaction_id",
        "amount",
        "failure_reason",
        "retry_count",
        "recovery_probability",
        "ai_action",
        "simulation_result",
        "recovered_amount"
    ]
)


# ==========================================
# BUSINESS METRICS
# ==========================================

revenue_at_risk = results_df["amount"].sum()

revenue_recovered = results_df[
    results_df["simulation_result"] == "SUCCESS"
]["recovered_amount"].sum()

successful_transactions = len(
    results_df[
        results_df["simulation_result"] == "SUCCESS"
    ]
)

attempted_transactions = len(
    results_df[
        results_df["simulation_result"] != "NOT ATTEMPTED"
    ]
)


if attempted_transactions > 0:

    recovery_rate = (
        successful_transactions /
        attempted_transactions
    ) * 100

else:

    recovery_rate = 0


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("       AI RECOVERY SIMULATION")
print("==========================================")

print(
    "\nRevenue at Risk: ₹",
    round(revenue_at_risk, 2)
)

print(
    "Revenue Recovered: ₹",
    round(revenue_recovered, 2)
)

print(
    "Successful Recoveries:",
    successful_transactions
)

print(
    "Recovery Attempts:",
    attempted_transactions
)

print(
    "Recovery Rate:",
    round(recovery_rate, 2),
    "%"
)


# ==========================================
# ACTION SUMMARY
# ==========================================

print("\n==========================================")
print("          AI ACTION SUMMARY")
print("==========================================")

print(
    results_df["ai_action"].value_counts()
)


# ==========================================
# SAVE RESULTS
# ==========================================

results_df.to_csv(
    "final_recovery_results.csv",
    index=False
)

print("\nResults saved to:")
print("final_recovery_results.csv")


# ==========================================
# SHOW FIRST 10 RESULTS
# ==========================================

print("\n==========================================")
print("       FIRST 10 TRANSACTIONS")
print("==========================================")

print(
    results_df.head(10).to_string(index=False)
)
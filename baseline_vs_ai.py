import pandas as pd
import random

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("payment_dataset.csv")

print("Dataset loaded:", len(df))


# ==========================================
# BASELINE STRATEGY
# ==========================================

baseline_recovered = 0
baseline_revenue = 0
baseline_attempts = 0


for _, row in df.iterrows():

    # Baseline blindly retries every payment once
    baseline_attempts += 1

    probability = 50

    if row["failure_reason"] == "Bank Timeout":
        probability = 60

    elif row["failure_reason"] == "Insufficient Balance":
        probability = 30

    elif row["failure_reason"] == "Card Declined":
        probability = 35

    # Simulate recovery

    if random.randint(1, 100) <= probability:

        baseline_recovered += 1
        baseline_revenue += row["amount"]


# ==========================================
# AI STRATEGY
# ==========================================

ai_recovered = 0
ai_revenue = 0
ai_attempts = 0

MAX_RETRIES = 2


for _, row in df.iterrows():

    probability = 50

    # Customer history

    if row["previous_transactions"] >= 10:
        probability += 15

    if row["previous_failures"] <= 1:
        probability += 15

    # Retry history

    if row["retry_count"] == 0:
        probability += 10

    # Failure reason

    if row["failure_reason"] == "Bank Timeout":
        probability += 10

    elif row["failure_reason"] == "Insufficient Balance":
        probability -= 20

    elif row["failure_reason"] == "Card Declined":
        probability -= 15

    probability = max(0, min(100, probability))


    # AI decision

    if row["retry_count"] >= MAX_RETRIES:

        action = "STOP"

    elif probability >= 70:

        action = "RETRY"

    elif probability >= 40:

        action = "PAYMENT LINK"

    else:

        action = "STOP"


    # Recovery attempt

    if action != "STOP":

        ai_attempts += 1

        if random.randint(1, 100) <= probability:

            ai_recovered += 1
            ai_revenue += row["amount"]


# ==========================================
# CALCULATE RECOVERY RATES
# ==========================================

baseline_rate = (
    baseline_recovered /
    baseline_attempts
) * 100


ai_rate = (
    ai_recovered /
    ai_attempts
) * 100


# ==========================================
# AI IMPROVEMENT
# ==========================================

additional_revenue = (
    ai_revenue -
    baseline_revenue
)


if baseline_revenue > 0:

    revenue_improvement = (
        additional_revenue /
        baseline_revenue
    ) * 100

else:

    revenue_improvement = 0


# ==========================================
# PRINT RESULTS
# ==========================================

print("\n============================================")
print("          BASELINE vs AI COMPARISON")
print("============================================")


print("\n🟠 BASELINE STRATEGY")

print(
    "Recovery attempts:",
    baseline_attempts
)

print(
    "Successful recoveries:",
    baseline_recovered
)

print(
    "Revenue recovered: ₹",
    f"{baseline_revenue:,.0f}"
)

print(
    "Recovery rate:",
    round(baseline_rate, 2),
    "%"
)


print("\n🤖 AI RECOVERY STRATEGY")

print(
    "Recovery attempts:",
    ai_attempts
)

print(
    "Successful recoveries:",
    ai_recovered
)

print(
    "Revenue recovered: ₹",
    f"{ai_revenue:,.0f}"
)

print(
    "Recovery rate:",
    round(ai_rate, 2),
    "%"
)


print("\n============================================")
print("             AI IMPROVEMENT")
print("============================================")


print(
    "Additional revenue recovered: ₹",
    f"{additional_revenue:,.0f}"
)

print(
    "Revenue improvement:",
    round(revenue_improvement, 2),
    "%"
)


# ==========================================
# SAVE COMPARISON DATA
# ==========================================

comparison = pd.DataFrame({

    "Strategy": [
        "Baseline",
        "AI Agent"
    ],

    "Recovery Attempts": [
        baseline_attempts,
        ai_attempts
    ],

    "Successful Recoveries": [
        baseline_recovered,
        ai_recovered
    ],

    "Revenue Recovered": [
        baseline_revenue,
        ai_revenue
    ],

    "Recovery Rate": [
        round(baseline_rate, 2),
        round(ai_rate, 2)
    ]
})


comparison.to_csv(
    "baseline_ai_comparison.csv",
    index=False
)


print("\nComparison saved to:")
print("baseline_ai_comparison.csv")

print("\n")
print(comparison)
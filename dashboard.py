import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Revenue Recovery",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# CUSTOM UI
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
}

.header {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(90deg, #111827, #2563eb);
    color: white;
    margin-bottom: 25px;
}

.header h1 {
    margin-bottom: 5px;
}

.header p {
    margin: 0;
}

.section-title {
    font-size: 22px;
    font-weight: bold;
    margin-top: 25px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

try:

    df = pd.read_csv("payment_dataset.csv")

except FileNotFoundError:

    st.error(
        "payment_dataset.csv not found. "
        "Make sure it is in the same folder as dashboard.py."
    )

    st.stop()


# =========================================================
# ENCODE DATA FOR ML
# =========================================================

payment_encoder = LabelEncoder()
failure_encoder = LabelEncoder()
customer_encoder = LabelEncoder()


df["payment_method_encoded"] = (
    payment_encoder.fit_transform(
        df["payment_method"]
    )
)


df["failure_reason_encoded"] = (
    failure_encoder.fit_transform(
        df["failure_reason"]
    )
)


df["customer_type_encoded"] = (
    customer_encoder.fit_transform(
        df["customer_type"]
    )
)


# =========================================================
# ML FEATURES
# =========================================================

features = [
    "amount",
    "payment_method_encoded",
    "failure_reason_encoded",
    "previous_transactions",
    "previous_failures",
    "retry_count",
    "customer_type_encoded"
]


X = df[features]


y = df["recovered"].map({
    "Yes": 1,
    "No": 0
})


# =========================================================
# TRAIN RANDOM FOREST
# =========================================================

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


model.fit(
    X_train,
    y_train
)


# =========================================================
# AI RECOVERY AGENT
# =========================================================

MAX_RETRIES = 2


def recovery_agent(transaction, probability):

    retry_count = transaction["retry_count"]

    amount = transaction["amount"]

    failure_reason = transaction["failure_reason"]

    customer_type = transaction["customer_type"]


    # ---------------------------------------------
    # RETRY LIMIT
    # ---------------------------------------------

    if retry_count >= MAX_RETRIES:

        return (
            "STOP",
            "Maximum retry limit reached. "
            "Further retries are stopped to reduce "
            "unnecessary customer friction."
        )


    # ---------------------------------------------
    # HIGH PROBABILITY
    # ---------------------------------------------

    if probability >= 70:

        return (
            "RETRY",
            f"Recovery probability is {probability:.1f}%. "
            f"The payment amount is ₹{amount:.0f}. "
            f"The failure reason is {failure_reason}. "
            "A limited retry is recommended."
        )


    # ---------------------------------------------
    # MEDIUM PROBABILITY
    # ---------------------------------------------

    elif probability >= 40:

        return (
            "PAYMENT LINK",
            f"Recovery probability is {probability:.1f}%. "
            f"The customer is classified as {customer_type}. "
            "A payment link is recommended instead of "
            "another automatic retry."
        )


    # ---------------------------------------------
    # LOW PROBABILITY
    # ---------------------------------------------

    else:

        return (
            "STOP",
            f"Recovery probability is only {probability:.1f}%. "
            "Another attempt may create unnecessary "
            "customer friction, so the agent stops."
        )


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="header">

<h1>🤖 AI Revenue Recovery</h1>

<p>
Intelligent failed-payment recovery and revenue protection system
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Controls")

actions = [
    "All",
    "RETRY",
    "PAYMENT LINK",
    "STOP"
]


selected_action = st.sidebar.selectbox(
    "AI Recovery Action",
    actions
)


# =========================================================
# RUN ML + AGENT FOR ALL TRANSACTIONS
# =========================================================

probabilities = []

ai_actions = []

ai_explanations = []


for index, transaction in df.iterrows():

    input_data = pd.DataFrame(
        [[
            transaction["amount"],
            transaction["payment_method_encoded"],
            transaction["failure_reason_encoded"],
            transaction["previous_transactions"],
            transaction["previous_failures"],
            transaction["retry_count"],
            transaction["customer_type_encoded"]
        ]],
        columns=features
    )


    probability = (
        model.predict_proba(input_data)[0][1]
        * 100
    )


    action, explanation = recovery_agent(
        transaction,
        probability
    )


    probabilities.append(probability)

    ai_actions.append(action)

    ai_explanations.append(explanation)


df["recovery_probability"] = probabilities

df["ai_action"] = ai_actions

df["ai_explanation"] = ai_explanations


# =========================================================
# FILTER
# =========================================================

if selected_action != "All":

    filtered_df = df[
        df["ai_action"] == selected_action
    ]

else:

    filtered_df = df.copy()


# =========================================================
# BUSINESS METRICS
# =========================================================

revenue_at_risk = filtered_df["amount"].sum()


ai_recovered = filtered_df[
    filtered_df["ai_action"] != "STOP"
]["amount"].sum()


total_transactions = len(filtered_df)


if revenue_at_risk > 0:

    recovery_rate = (
        ai_recovered /
        revenue_at_risk
    ) * 100

else:

    recovery_rate = 0


# =========================================================
# BUSINESS OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-title">📊 Business Overview</div>',
    unsafe_allow_html=True
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "💰 Revenue at Risk",
        f"₹{revenue_at_risk:,.0f}"
    )


with col2:

    st.metric(
        "🤖 AI Recovery Value",
        f"₹{ai_recovered:,.0f}"
    )


with col3:

    st.metric(
        "📈 Recovery Rate",
        f"{recovery_rate:.2f}%"
    )


with col4:

    st.metric(
        "💳 Transactions",
        f"{total_transactions:,}"
    )


# =========================================================
# AI ACTIONS
# =========================================================

st.markdown(
    '<div class="section-title">🤖 AI Recovery Actions</div>',
    unsafe_allow_html=True
)


action_counts = filtered_df[
    "ai_action"
].value_counts()


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "🔄 Retry",
        action_counts.get("RETRY", 0)
    )


with col2:

    st.metric(
        "🔗 Payment Link",
        action_counts.get("PAYMENT LINK", 0)
    )


with col3:

    st.metric(
        "🛑 Stop",
        action_counts.get("STOP", 0)
    )


st.bar_chart(action_counts)


# =========================================================
# FAILURE REASON ANALYSIS
# =========================================================

st.markdown(
    '<div class="section-title">⚠️ Revenue at Risk by Failure Reason</div>',
    unsafe_allow_html=True
)


failure_revenue = (
    filtered_df
    .groupby("failure_reason")["amount"]
    .sum()
)


st.bar_chart(failure_revenue)


# =========================================================
# TRANSACTION SEARCH
# =========================================================

st.markdown(
    '<div class="section-title">🔍 Transaction Search</div>',
    unsafe_allow_html=True
)


search_id = st.text_input(
    "Enter Transaction ID",
    placeholder="Example: TXN00025"
)


if search_id:

    search_id = search_id.strip().upper()


    result = df[
        df["transaction_id"] == search_id
    ]


    if result.empty:

        st.warning(
            "❌ Transaction not found."
        )


    else:

        transaction = result.iloc[0]


        st.success(
            f"Transaction {search_id} found!"
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "💰 Amount",
                f"₹{transaction['amount']:,.0f}"
            )


        with col2:

            st.metric(
                "🎯 ML Probability",
                f"{transaction['recovery_probability']:.1f}%"
            )


        with col3:

            st.metric(
                "🤖 AI Action",
                transaction["ai_action"]
            )


        with col4:

            st.metric(
                "🔄 Retry Count",
                transaction["retry_count"]
            )


        st.markdown("### 🧠 AI Explanation")


        st.info(
            transaction["ai_explanation"]
        )


        st.markdown("### 📋 Transaction Details")


        st.dataframe(
            result[
                [
                    "transaction_id",
                    "customer_id",
                    "amount",
                    "payment_method",
                    "failure_reason",
                    "previous_transactions",
                    "previous_failures",
                    "retry_count",
                    "customer_type",
                    "recovery_probability",
                    "ai_action"
                ]
            ],
            width="stretch",
            hide_index=True
        )


# =========================================================
# TRANSACTION TABLE
# =========================================================

st.markdown(
    '<div class="section-title">📋 All AI Decisions</div>',
    unsafe_allow_html=True
)


st.dataframe(
    filtered_df[
        [
            "transaction_id",
            "amount",
            "failure_reason",
            "recovery_probability",
            "ai_action"
        ]
    ],
    width="stretch",
    hide_index=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        🤖 AI Revenue Recovery System |
        Track 3 Prototype
    </div>
    """,
    unsafe_allow_html=True
)
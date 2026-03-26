import pandas as pd
import requests
import json
import os

# 1. LOAD DATA

all_transactions = pd.read_csv('transactions.csv')
all_cashflow     = pd.read_csv('daily_cashflow.csv')

# 2. BUSINESS SELECTION
available_biz = sorted(all_transactions['business_id'].unique().tolist())

print("\n" + "="*50)
print("   FinBot - AI Financial Assistant")
print("="*50)
print("\n Available Businesses:")
for i, biz in enumerate(available_biz, 1):
    print(f"   {i}. {biz}")

while True:
    choice = input("\nEnter Business ID (EXAMPLE: BIZ_001): ").strip().upper()
    if choice in available_biz:
        SELECTED_BIZ = choice
        print(f"\n Selected: {SELECTED_BIZ}")
        break
    else:
        print(f" '{choice}' not found. Please enter a valid Business ID from the list above.")

# 3. FILTER DATA FOR SELECTED BUSINESS


df = all_transactions[all_transactions['business_id'] == SELECTED_BIZ].copy()

# Cashflow: filter by business if column exists, else use global
if 'business_id' in all_cashflow.columns:
    cashflow_df = all_cashflow[all_cashflow['business_id'] == SELECTED_BIZ].copy()
else:
    cashflow_df = all_cashflow.copy()

# Cashflow summary
total_income    = cashflow_df['total_income'].sum()
total_expense   = cashflow_df['total_expense'].sum()
avg_cashflow    = cashflow_df['net_cashflow'].mean()
current_balance = cashflow_df['cumulative_balance'].iloc[-1]

# Transaction analysis for selected business
expenses_df = df[df['type'] == 'expense']
income_df   = df[df['type'] == 'income']

biz_total_expense = expenses_df['amount'].sum()
biz_total_income  = income_df['amount'].sum()
biz_net_profit    = biz_total_income - biz_total_expense

top_category   = expenses_df.groupby('category')['amount'].sum().idxmax()
top_cat_amount = expenses_df.groupby('category')['amount'].sum().max()

expense_breakdown = (
    expenses_df.groupby('category')['amount']
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

# Recent 5 transactions
recent_txns = df.sort_values('date', ascending=False).head(5)[
    ['date', 'description', 'amount', 'type', 'category']
]

# Month-wise breakdown (fix: lowercase column names after unstack)
df['month'] = pd.to_datetime(df['date']).dt.strftime('%B %Y')
monthly_summary = (
    df.groupby(['month', 'type'])['amount']
    .sum()
    .unstack(fill_value=0)
    .reset_index()
)
monthly_summary.columns = [str(c).lower() for c in monthly_summary.columns]  # BUG FIX

# Build context strings
expense_lines = "\n".join(
    [f"  - {cat}: ₹{amt:,.0f}" for cat, amt in expense_breakdown.items()]
)

monthly_lines = "\n".join(
    [f"  - {row['month']}: Income ₹{row.get('income', 0):,.0f} | Expense ₹{row.get('expense', 0):,.0f}"
     for _, row in monthly_summary.iterrows()]
)

recent_lines = "\n".join(
    [f"  - {row['date']} | {row['description']} | ₹{row['amount']:,.0f} ({row['type']} - {row['category']})"
     for _, row in recent_txns.iterrows()]
)

#ANOMALY DETECTION
def detect_anomalies(expenses_df):
    anomalies = []

    # Find transactions 3x above category average
    cat_avg = expenses_df.groupby('category')['amount'].mean()

    for _, row in expenses_df.iterrows():
        avg = cat_avg.get(row['category'], 0)
        if avg > 0 and row['amount'] > 3 * avg:
            anomalies.append({
                "date": row['date'],
                "description": row['description'],
                "amount": row['amount'],
                "category": row['category'],
                "normal_avg": round(avg, 2),
                "times_higher": round(row['amount'] / avg, 1)
            })

    return anomalies[:5]  # top 5 anomalies

anomalies = detect_anomalies(expenses_df)
anomaly_lines = "\n".join([
    f"  {a['date']} | {a['description']} | ₹{a['amount']:,.0f} "
    f"({a['times_higher']}x above normal ₹{a['normal_avg']:,.0f} avg)"
    for a in anomalies
]) or "  No anomalies detected."

FINANCIAL_CONTEXT = f"""
=== FINANCIAL SUMMARY FOR {SELECTED_BIZ} ===

Business Income  : ₹{biz_total_income:,.0f}
Business Expense : ₹{biz_total_expense:,.0f}
Net Profit       : ₹{biz_net_profit:,.0f}
Current Balance  : ₹{current_balance:,.0f}
Avg Daily CF     : ₹{avg_cashflow:,.0f}

Top Expense Category: {top_category} (₹{top_cat_amount:,.0f})

Top 5 Expense Categories:
{expense_lines}

Monthly Breakdown:
{monthly_lines}

Recent Transactions:
{recent_lines}

Anomaly Alerts:
{anomaly_lines}

Total Transactions for {SELECTED_BIZ}: {len(df):,}
Date Range: {df['date'].min()} to {df['date'].max()}
"""

print("\n" + FINANCIAL_CONTEXT)
print(" Data loaded successfully!\n")

# 4. HUGGING FACE API SETUP

import os
HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    raise EnvironmentError(
        "\n[ERROR] HF_API_KEY not set!\n"
    )

# model for financial chat
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct:cerebras"
HF_URL   = "https://router.huggingface.co/v1/chat/completions"
SYSTEM_INSTRUCTION = f"""You are FinBot, an expert AI financial assistant for small businesses.
You are assisting: {SELECTED_BIZ}

REAL FINANCIAL DATA:
{FINANCIAL_CONTEXT}

RULES:
1. Always use EXACT numbers from the data — never guess
2. Format amounts as ₹X,XX,XXX
3. If asked about anomalies/alerts, highlight the anomaly alerts section
4. Give 2-3 specific actionable tips, not generic advice
5. If asked something not in the data, say "I don't have that data"
6. Keep responses under 150 words
7. Be direct — give the answer first, then explanation

You can answer questions about:
- Income, expenses, profit, balance
- Category-wise spending breakdown
- Monthly trends
- Unusual spending alerts
- Cost reduction advice
- Cash flow health"""

# HUGGING FACE CHAT FUNCTION

def ask_finbot(user_message: str, chat_history: list) -> str:
    """Send a message to HuggingFace Inference API (OpenAI-compatible endpoint)."""

    # Build message list: system prompt + history + new user message
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]

    for turn in chat_history:
        messages.append({
            "role": turn["role"],
            "content": turn["content"]
        })

    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            HF_URL,
            headers={
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=60  # HF free tier can be slow — give it extra time
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()

        elif response.status_code == 503:
            return " Model is loading on HuggingFace servers. Please wait 20 seconds and try again."

        elif response.status_code == 429:
            return " Rate limit reached. HuggingFace free tier allows limited requests. Wait a minute and retry."

        else:
            return f"API Error {response.status_code}: {response.text}"

    except requests.exceptions.Timeout:
        return "Request timed out. HuggingFace free tier can be slow — please try again."
    except requests.exceptions.ConnectionError:
        return " Connection failed. Please check your internet connection."
    except KeyError:
        return " Unexpected response format from API. Try again or switch to a different model."
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# 6. CHATBOT LOOP

def run_chatbot():
    print("="*50)
    print(f"   FinBot — Assisting {SELECTED_BIZ}")
    print("="*50)
    print(f"   Model : {HF_MODEL}")
    print("   Type 'quit' to exit | 'clear' to reset chat\n")

    chat_history = []  # stores {"role": "user"/"assistant", "content": "..."}

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("FinBot: Goodbye! Stay financially healthy 💰")
            break

        if user_input.lower() == 'clear':
            chat_history = []
            print("FinBot: Chat history cleared. Starting fresh!\n")
            continue

        print("FinBot: Thinking...", end="\r")

        reply = ask_finbot(user_input, chat_history)

        # Update conversation history (standard OpenAI format)
        chat_history.append({"role": "user",      "content": user_input})
        chat_history.append({"role": "assistant",  "content": reply})

        # Keep last 10 turns only to avoid token limit
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]

        print(f"FinBot: {reply}\n")


if __name__ == "__main__":
    run_chatbot()

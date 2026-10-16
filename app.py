import os
import gradio as gr
from groq import Groq

SYSTEM_PROMPT = """
You are the customer support assistant for BrightMart Nigeria.

Be polite, helpful, and concise.
Use only the business information below.
Never invent prices, delivery dates, or refund approvals.
If you do not know an answer, recommend speaking with a human agent.

Business information:
Products: Household items, electronics, and personal care products.
Payment: Approved bank transfers and methods listed at checkout.
Delivery: Delivery times depend on the customer's location.
Returns: Support must confirm whether an item qualifies for a return.
Support hours: Monday to Friday, 9 AM to 5 PM.

Never ask customers to share passwords, PINs, or OTPs.
"""

ESCALATION_KEYWORDS = [
    "failed payment",
    "payment failed",
    "money deducted",
    "wrong charge",
    "fraud",
    "complaint",
    "speak to a human",
    "talk to an agent",
    "human agent",
    "order not arrived",
    "not received my order",
]

def respond(message, history):
    message_lower = message.lower()

    if any(keyword in message_lower for keyword in ESCALATION_KEYWORDS):
        return (
            "I'm sorry you're experiencing this issue. "
            "This may require further investigation by our support team. "
            "Please contact a human support representative for assistance. "
            "For your safety, never share your password, PIN, or OTP."
        )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "The assistant is not configured yet. Please contact the administrator."

    try:
        client = Groq(api_key=api_key)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Use this visitor's conversation history for follow-up questions.
        for item in history or []:
            if isinstance(item, dict):
                if item.get("role") in ("user", "assistant") and item.get("content"):
                    messages.append({
                        "role": item["role"],
                        "content": item["content"]
                    })
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                user_message, assistant_message = item
                if user_message:
                    messages.append({"role": "user", "content": str(user_message)})
                if assistant_message:
                    messages.append({"role": "assistant", "content": str(assistant_message)})

        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.3,
        )

        return response.choices[0].message.content

    except Exception:
        return (
            "Sorry, I couldn't process your request right now. "
            "Please try again later or contact a human support representative."
        )

demo = gr.ChatInterface(
    fn=respond,
    title="BrightMart AI Customer Support",
    description=(
        "Welcome to BrightMart Nigeria! Ask about products, payments, "
        "delivery, or returns. This is a demonstration using fictional business data."
    ),
    examples=[
        "What products do you sell?",
        "How can I pay for my order?",
        "What are your delivery times?",
        "Can I return a product?",
    ],
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)

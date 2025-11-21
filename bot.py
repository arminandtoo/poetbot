from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

from joblib import load
import os

vectorizer_path = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.joblib")
model_path = os.path.join(os.path.dirname(__file__), "svm_model.joblib")

vectorizer = load(vectorizer_path)
clf = load(model_path)

def predict_poet(poem_text):
    try:
        X_vec = vectorizer.transform([poem_text])
        pred_label = clf.predict(X_vec)[0]
        return pred_label
    except Exception as e:
        return None

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    if user_message.strip().lower() in ("سلام", "start", "/start"):
        await update.message.reply_text("سلام! به ربات تشخیص شاعر شعر خوش اومدی 🌹\n" +
            "یک قطعه شعر برام بفرست تا حدس بزنم شاعرش کیه.")
    else:
        result = predict_poet(user_message)
        if result is None:
            await update.message.reply_text("متاسفم! این شعر در دیتابیس نبود یا خطایی رخ داد.")
        else:
            await update.message.reply_text(f"حدس من: {result}")

if __name__ == "__main__":
    app = ApplicationBuilder().token("8553159544:AAEQTbfwax_EPcfbrosgrp3DCMesqQHDx9U").build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

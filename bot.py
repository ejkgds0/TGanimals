import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

# Загружаем .env
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")

# Обработчик любых сообщений
async def on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ctx.bot.send_chat_action(update.effective_chat.id, 'typing')
    await update.message.reply_text("Привет! Я TGanimals‑бот. Пришли мне фото животного — я определю, кто это.")

# Точка входа
if __name__ == '__main__':
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.ALL, on_message))
    print("Bot is running...")
    app.run_polling()

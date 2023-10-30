import telebot
import os
from dotenv import load_dotenv
import model

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

bot = telebot.TeleBot(TOKEN)
bot.set_webhook()

@bot.message_handler(commands=['start'])
def start(message):
    """
    Bot will introduce itself upon /start command, and prompt user for his request
    """
    try:
        # selected_language = None

        # send this msg auto when user starts the bot
        start_message = "Hello, I'm James! Ask me anything about HealthServe or migrant workers in Singapore and I will help!"
        bot.send_message(message.chat.id, start_message)
        
        # get bot to prompt for language (give only3 options: English, Chinese, Bangla in the form of buttons)
        language_message = "The current default language is English. Please type in another language should you wish to change it."
        bot.send_message(message.chat.id, language_message)

    except Exception as e:
        bot.send_message(
            message.chat.id, 'Sorry, something seems to gone wrong! Please try again later!')
        
@bot.message_handler(commands=['clear'])
def clear(message):
    """
    Bot will clear memory upon /clear command
    """
    model.clear()
    bot.send_message(message.chat.id, "Memory cleared!")


@bot.message_handler(content_types=['text'])
def send_text(message):
    response = model.getResponse(message.text)
    bot.send_message(message.chat.id, response)

def main():
    """Runs the Telegram Bot"""
    print("Starting Telegram Bot...")
    bot.infinity_polling()


if __name__ == '__main__':
    main()
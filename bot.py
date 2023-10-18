import telebot
import os
from dotenv import load_dotenv
import model

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

bot = telebot.TeleBot(TOKEN)
bot.set_webhook()

# Add a global variable to store the selected language
selected_language = ""

@bot.message_handler(commands=['start'])
def start(message):
    """
    Bot will introduce itself upon /start command, and prompt user for his request
    """
    try:
        # Start bot introduction
        start_message = "Hello, I'm James! Ask me anything about HealthServe or migrant workers in Singapore and I will help!"
        bot.send_message(message.chat.id, start_message)
        # get bot to prompt for language (give only3 options: English, Chinese, Bangla in the form of buttons)
        language_message = "What language would you like to use?"
        markup = telebot.types.ReplyKeyboardMarkup(row_width=3)
        itembtn1 = telebot.types.KeyboardButton('English')
        itembtn2 = telebot.types.KeyboardButton('Chinese')
        itembtn3 = telebot.types.KeyboardButton('Bangla')
        markup.add(itembtn1, itembtn2, itembtn3)
        bot.send_message(message.chat.id, language_message, reply_markup=markup)
        
        # tell bot to report back the language selected
        # Provide a translated confirmation message based on the selected language
        confirmation_messages = {
            "English": "You've selected English. What questions would you like to ask in this language?",
            "Chinese": "你选择了中文。您想用这种语言问什么问题？",
            "Bengali": "আপনি বেঙ্গালি ভাষা নির্বাচন করেছেন। আপনি কি এই ভাষায় কোন প্রশ্ন করতে চান?"
        }

        confirm_message = confirmation_messages.get(selected_language, "You've selected a language. What questions would you like to ask in this language?")
        bot.send_message(message.chat.id, confirm_message)

        
        # Store language choice in a global variable that the model can access
        global selected_language
        selected_language = message.text

    except Exception as e:
        bot.send_message(
            message.chat.id, 'Sorry, something seems to gone wrong! Please try again later!')


@bot.message_handler(content_types=['text'])
def send_text(message):
    if not hasattr(model, "language"):
        # If the model doesn't have a "language" attribute, set it to the selected language
        setattr(model, "language", selected_language)

    response = model.getResponse(message.text)
    bot.send_message(message.chat.id, response)

def main():
    """Runs the Telegram Bot"""
    print("Starting Telegram Bot...")
    bot.infinity_polling()


if __name__ == '__main__':
    main()
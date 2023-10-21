import telebot
import os
from dotenv import load_dotenv
import model

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

bot = telebot.TeleBot(TOKEN)
bot.set_webhook()

selected_language = None

@bot.message_handler(commands=['start'])
def start(message):
    """
    Bot will introduce itself upon /start command, and prompt user for his request
    """
    try:
        global selected_language
        selected_language = "English" # default language is English

        # send this msg auto when user starts the bot
        start_message = "Hello, I'm James! Ask me anything about HealthServe or migrant workers in Singapore and I will help!"
        bot.send_message(message.chat.id, start_message)
        # get bot to prompt for language (give only3 options: English, Chinese, Bangla in the form of buttons)
        language_message = "What language would you like to use?"
        bot.send_message(message.chat.id, language_message)

        markup = telebot.types.ReplyKeyboardMarkup(row_width=3)
        itembtn1 = telebot.types.KeyboardButton('English')
        itembtn2 = telebot.types.KeyboardButton('Chinese')
        itembtn3 = telebot.types.KeyboardButton('Bangla')
        # add buttons to markup
        markup.add(itembtn1, itembtn2, itembtn3)
        # send markup to user
        bot.send_message(message.chat.id, reply_markup=markup)
        
        
        # tell bot to report back the language selected
        # Provide a translated confirmation message based on the selected language
        confirmation_messages = {
            "English": "You've selected English. What questions would you like to ask in this language?",
            "Chinese": "你选择了中文。您想用这种语言问什么问题？",
            "Bangla": "আপনি বেঙ্গালি ভাষা নির্বাচন করেছেন। আপনি কি এই ভাষায় কোন প্রশ্ন করতে চান?"
        }
        # base on the selected language, send the corresponding confirmation message
        # only shown after user click on language button
        # await user input
        if selected_language != None: 
            bot.send_message(message.chat.id, confirmation_messages[selected_language])

        # add a clear memory button
        clear_memory = telebot.types.KeyboardButton('Clear Memory')
        markup.add(clear_memory)


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
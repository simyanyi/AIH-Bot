# AIH-Bot

We are from G2T1 of COR2221 AI & Humanity. As part of our project, we are required to development a chatbot to help HealthServe employees handle questions regarding migrant workers in Singapore. <br>

However, implementing a whole new web application may be too tedious and out-of-scope, so we have provided code to implement your code to a telegram bot instead!

## Steps to follow: 

### 1. Clone to your repository

On your preferred IDE, open the folder that you wish you put the project in, and proceed to run the following in your shell:

```
git clone https://github.com/simyanyi/AIH-Bot.git
```

And afterwhich,

```
cd AIH-Bot
```

### 2. Creating the environment variables

On the `AIH-Bot` directory, create a `.env` file or use an existing one (if you have from the lab). Open the file to the following:

```
OPENAI_API_KEY=sk-YOUR_OPENAI_APIKEY
LANGSMITH_API_KEY=ls__YOUR_LS_APIKEY
TELEGRAM_BOT_TOKEN=YOUR_BOTTOKEN
```

Insert your OpenAI, LangChain API Key, and Telegram API.

We will talk about how to get your Telegram bot token in the next step.

### 3. Get your telegram API key (if you do not have one prior)

You will need the API key to connect to a bot. This requires you to navigate to [BotFather](https://t.me/BotFather). Do refer to [this video](https://www.youtube.com/watch?v=aNmRNjME6mE&ab_channel=SmartBotsLand) should you need help to gather the API key.

After you receive the API key, save it into the `.env` file.

### 4. Install all dependencies

Windows: 
Run the following on your terminal (Command Prompt)

```
pip install -r requirements.txt
```

Mac: Run the following on your terminal (zsh)
```
pip3 install -r requirements.txt
```

### 5. Add source documents
Create a research folder under `AIH-Bot` directory and drag the downloaded source documents into the folder.

### 6. How to run our code

There are two Python files that will be running the show, `bot.py` and `model.py`.

- `bot.py` will assist in receiving and sending out responses. 

- `model.py` currently consists of only 2 functions. `getResponses(question)` which takes in the user's input and should return the message that we would like to return to the user. `clear()` helps clear conversational memory.

## 7. Test the bot

Time for you to try our bot out!

Windows: Run the following on your terminal (Command Prompt)
```
python bot.py
```

Mac: Run the following on your terminal (zsh)
```
python3 bot.py
```

If everything works, it should produce the following:
```
Loading configuration...
Successfully loaded! Starting bot...
```

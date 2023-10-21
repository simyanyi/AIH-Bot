import os
import openai
import sys
import numpy as np
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from bot import selected_language


def getResponse(question: str) -> str:
    global selected_language

    load_dotenv('./.env')

    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

    loader = PyPDFDirectoryLoader("./research/")
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    splits = text_splitter.split_documents(pages)

    # Your experiment can start from this code block which loads the vector store into variable vectordb
    embedding = OpenAIEmbeddings()

    # Reference https://github.com/hwchase17/chroma-langchain/blob/master/persistent-qa.ipynb
    persist_directory = './research/vectordb'

    # Perform embeddings and store the vectors
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory # Writes to local directory
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key='question',
        output_key='answer'
    )

    # Code below will enable tracing so we can take a deeper look into the chain
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
    os.environ["LANGCHAIN_PROJECT"] = "Chatbot"

    # Define parameters for retrival via similarity score threshold (change k, score_threshold as needed)
    retriever = vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .8, "k": 10})

    # Define parameters for retrival via mmr
    # retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 10})


    # Define llm model

    # use user input to determine language
    language = selected_language
    
    llm_name = "gpt-3.5-turbo-16k"
    llm = ChatOpenAI(model_name=llm_name, temperature=0)
    
    # Define template prompt
    if question == "English":
        template = """You are a friendly chatbot helping to answer questions by employees at HealthServe regarding Singapore's migrant workers' healthcare. Use the following pieces of context to answer the question at the end.
        {context}
        Question: {question}
        Helpful Answer in English Language: """
        context = "You are a cheerful bot who is nice and friendly, and aims to help answer questions from HealthServe employees"
    
    elif question == "Chinese":
        template = """您是一个友好的聊天机器人，帮助回答 HealthServe 员工有关新加坡移民工人医疗保健的问题。使用以下上下文来回答最后的问题。
        {context}
        Question: {question}
        Helpful Answer in Chinese Language: """
        context = "您是一个开朗的机器人，友善且友善，旨在帮助回答 HealthServe 员工的问题"
    
    else:
        template = """আপনি একজন বন্ধুত্বপূর্ণ চ্যাটবট যিনি সিঙ্গাপুরের অভিবাসী কর্মীদের স্বাস্থ্যসেবা সংক্রান্ত HealthServe-এর কর্মীদের প্রশ্নের উত্তর দিতে সাহায্য করেন। শেষে প্রশ্নের উত্তর দিতে নিচের প্রসঙ্গগুলো ব্যবহার করুন।
        {context}
        Question: {question}
        Helpful Answer in Bangla Language: """
        context = "আপনি একজন প্রফুল্ল বট যিনি সুন্দর এবং বন্ধুত্বপূর্ণ, এবং হেলথসার্ভের কর্মীদের প্রশ্নের উত্তর দিতে সাহায্য করার লক্ষ্য রাখেন"


    your_prompt = PromptTemplate.from_template(template, context=context)

    # Execute chain
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        combine_docs_chain_kwargs={"prompt": your_prompt},
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
        memory=memory
    )

    # Evaluate your chatbot with questions
    result = qa({"question": question})

    # if clear memory button is pressed, clear memory
    if question == "Clear Memory":
        memory.clear()
        return "Memory cleared!"

    print("Response: \n",result["answer"])
    return result['answer']
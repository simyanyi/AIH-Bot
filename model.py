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
from tabulate import tabulate

load_dotenv('./.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')


def getResponse(question: str) -> str:  

    print("Question: ", question)

    loader = PyPDFDirectoryLoader("./research/")
    pages = loader.load()
    
    for i in range(len(pages)):
        sentence = pages[i].page_content
        sentence = sentence.replace('\n', ' ')
        sentence = sentence.replace('\t', ' ')
        pages[i].page_content = sentence

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    splits = text_splitter.split_documents(pages)

    embedding = OpenAIEmbeddings()

    # Reference https://github.com/hwchase17/chroma-langchain/blob/master/persistent-qa.ipynb
    persist_directory = './research/vectordb'

    # Perform embeddings and store the vectors
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory # Writes to local directory in G Drive
    )
    # print(vectordb._collection.count())

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
    
    k=5
    # Define parameters for retrival
    retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": k})
    # retriever=vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .8, "k": k})    
    
    # Define llm model
    llm_name = "gpt-3.5-turbo-16k"
    llm = ChatOpenAI(model_name=llm_name, temperature=0)

    # Define template prompt
    template = """You are a friendly chatbot, named James, helping to answer questions by employees at HealthServe regarding Singapore's migrant workers' healthcare. Use the following pieces of context to answer the question at the end.
        {context}
        Question: {question}
        Helpful Answer in the question's language: """
    context = "You are a cheerful bot who is nice and friendly, and aims to help answer questions from HealthServe employees"
    
    your_prompt = PromptTemplate.from_template(template, context=context)

    # Execute chain
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        combine_docs_chain_kwargs={"prompt": your_prompt},
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
        memory=memory,

    )

    # Evaluate your chatbot with questions
    result = qa({"question": question})

    # getting source documents
    table_data = []

    docs = vectordb.similarity_search(question, k)
    for i in range(k):
        metadata = docs[i].metadata
        page_content = docs[i].page_content
        source = metadata['source']
        page = metadata['page']
        chunk_data = [i, source, page, page_content]
        table_data.append(chunk_data)
        
    table_headers = ["Chunk", "Source", "Page", "Content"]
    table_str = tabulate(table_data, headers=table_headers, tablefmt="grid")

    print(table_str)
    print("Response: ", result['answer'])

    if table_data == []:
        return "Sorry, I don't understand your question. Please try again or refer to MOM's website for more information."

    return result['answer']

# model memory clear
def clear():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key='question',
        output_key='answer'
    )
    memory.clear()
    return "Memory cleared!"
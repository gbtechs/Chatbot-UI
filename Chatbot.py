from openai import OpenAI
import streamlit as st
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
import os

api_key = os.environ.get('OPENAI_API_KEY')

if api_key:
    print("API Key found")
else:
    print("OPENAI_API_KEY environment variable is not set.")

class ShippingAssistant:
    def __init__(self, path_to_xlsx):
        self.path_to_xlsx = path_to_xlsx
        self.loader = UnstructuredExcelLoader(path_to_xlsx, mode="elements")
        self.docs = self.loader.load()
        self.docs = filter_complex_metadata(self.docs)
        self.vectorstore = Chroma.from_documents(documents=self.docs, embedding=OpenAIEmbeddings(api_key))
        self.retriever = self.vectorstore.as_retriever()
        self.mem = ""

    def ask_query(self, query, carrier_name, shipment_id, departure_city, destination_city):
        self.mem = f"Assistant: Hello {carrier_name}, We’re pleased to inform you that you are the winning bidder on {shipment_id} from {departure_city} to {destination_city} picking etc etc"
        rag_template = """
        You are a shipping assistant for question-answering tasks. Use the following pieces of context to answer the question. If you don't know the answer, just refuse to answer it.and reply to greetings. Use three sentences maximum and keep the answer concise.
        CONTEXT:
        ```
            {docs}
        ```
        PREVIOUS DISCUSSION:
        ```
            {history}
        ```
        QUERY:
        ```
            {query}
        ```
        ANSWER:
        """
        prompt = PromptTemplate(
            partial_variables={"history": self.mem, "docs": self.docs},
            input_variables=["query"],
            template=rag_template,
        )
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
        rag_chain = prompt | llm | StrOutputParser()

        resp = rag_chain.invoke({"query": f"{query}"})
        self.mem += f"\nUser: {query}"
        self.mem += f"\nAssistant: {resp}"

        print(self.mem)
        return resp



path_to_xlsx = "Database ChatGPT (1).xlsx"
assistant = ShippingAssistant(path_to_xlsx)



with st.sidebar:
    #openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    user_name = st.text_input("User Name", key="name", type="default")
    company_name = st.text_input("company Name", key="company", type="default")
    
if not (user_name and company_name):
    st.info("Please add Your Name and Company name ")
else:
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": f"hi {user_name} We're pleased to have awarded {company_name} the pickUp to drop route"}]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        # client = OpenAI()
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        #response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
        response1 = assistant.ask_query(prompt, "Carrier 1", "123", "NYC", "LAX",)
        print(response1)
        #msg = response.choices[0].message.content
        msg=response1
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
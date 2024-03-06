import pandas as pd
import streamlit as st
from langchain_community.document_loaders import UnstructuredExcelLoader
# from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata

class ShippingAssistant:
    def __init__(self, path_to_xlsx):
        self.path_to_xlsx = path_to_xlsx
        self.loader = UnstructuredExcelLoader(path_to_xlsx, mode="elements")
        self.docs = self.loader.load()
        self.docs = filter_complex_metadata(self.docs)
        # self.vectorstore = Chroma.from_documents(documents=self.docs, embedding=OpenAIEmbeddings())
        # self.retriever = self.vectorstore.as_retriever()
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
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        rag_chain = prompt | llm | StrOutputParser()

        resp = rag_chain.invoke({"query": f"{query}"})
        self.mem += f"\nUser: {query}"
        self.mem += f"\nAssistant: {resp}"

        print(self.mem)
        return resp
    
    
path_to_xlsx = "Database ChatGPT (1).xlsx"
df = pd.read_excel(path_to_xlsx)
pick_up_location = df.loc[df['Data'] == 'Pick Up Location', 'Example'].values[0]
drop_off_location = df.loc[df['Data'] == 'Drop Off Location', 'Example'].values[0]
pick_up_time = df.loc[df['Data'] == 'Pick up time', 'Example'].values[0]
drop_off_time = df.loc[df['Data'] == 'Drop off time', 'Example'].values[0]
job_id = df.loc[df['Data'] == 'Job ID', 'Example'].values[0]
tracking_id = df.loc[df['Data'] == 'Tracking ID', 'Example'].values[0]
last_job_date = df.loc[df['Data'] == 'Last Job Date', 'Example'].values[0]

assistant = ShippingAssistant(path_to_xlsx)

# Streamlit UI code
st.title("Shipping Assistant")

with st.sidebar:
    name = st.text_input("Your Name", key="name", type="default")
    company_name = st.text_input("Company Name", key="company", type="default")

initial_message = f"""
Hello {name},

We’re pleased to have awarded {company_name} Shipment {job_id} from {pick_up_location} to {drop_off_location}. As you are a verified carrier, welcome back. Please let us know if any of your carrier details have changed since your last job on {last_job_date}.

The shipment pick up and drop off details are below:
- **Pick-up Location:** {pick_up_location}
- **Pick-up Time:** {pick_up_time}
- **Drop-off Location:** {drop_off_location}
- **Drop-off Time:** {drop_off_time}

Please use the link provided to engage tracking. This is a mandatory requirement for all our carriers.
**Tracking ID:** {tracking_id}

Any questions, please don’t hesitate to ask.

Thanks,  
TILT
"""

if not (name and company_name):
    st.info("Please provide Your Name and Company Name to start chatting!")
else:
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": initial_message}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = assistant.ask_query(prompt, "Carrier 1", "123", "NYC", "LAX")
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
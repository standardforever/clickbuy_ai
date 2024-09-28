import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from dotenv import load_dotenv
import os
import io
import sys
import re

load_dotenv()

groq_api = os.getenv("GROQ_API_KEY")
llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key=groq_api)

def clean_output(text):
    actions = re.compile(r'Action:.*?\n', re.DOTALL)
    finished_chain = re.compile(r'> Finished chain.*?\n', re.DOTALL)
    response = re.compile(r"Response:.*\n", re.DOTALL)
    text = actions.sub('', text)
    text = finished_chain.sub('', text)
    text = response.sub('', text)
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

def query_data(agent, query):
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        response = agent.invoke(query)
        print("Response:", response)
    except Exception as e:
        print("Error:", str(e))
    sys.stdout = old_stdout
    return clean_output(buffer.getvalue())

if 'history' not in st.session_state:
    st.session_state['history'] = []

csv_paths = {
    "ASIN Total Profit": './asin_total_profit_count.csv',
    "ASIN Count": './asin_count.csv',
    "ASIN Flip count": './asin_flip_count.csv'
}
agents = {name: create_csv_agent(llm, path, verbose=True, allow_dangerous_code=True) for name, path in csv_paths.items()}

csv_selection = st.selectbox("Choose a CSV file:", list(csv_paths.keys()), key='navbar')

st.image("./Clickbuy.jpeg", use_column_width='always')

with st.sidebar:
    st.write("## Chat History")
    for chat in reversed(st.session_state['history']): 
        st.text(f"Q: {chat['query']}")
        st.text_area("A:", chat['response'], height=150)


st.title(f'Data Query Interface for {csv_selection}')
user_query = st.text_input("Enter your query:")

if st.button('Submit'):
    if user_query:
        captured_output = query_data(agents[csv_selection], user_query)
        st.session_state['history'].append({"query": user_query, "response": captured_output})
        if captured_output:
            st.text_area("Query Response", captured_output, height=300)
        else:
            st.error("No response received from the agent or response was empty.")
    else:
        st.error("Please enter a query to fetch data.")

footer_html = f"<div style='text-align: center;'><a href='https://www.clickbuy.ai/' target='_blank'>Visit ClickBuy.ai</a></div>"
st.markdown(footer_html, unsafe_allow_html=True)

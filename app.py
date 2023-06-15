import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import (
    HumanMessage,
)
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import openai
from typing import Any, Dict, List

st.header("Copy&Chat")
st.subheader("Copy > Paste > Chat")


# event setting consultation
                                
def get_state(): 
     if "state" not in st.session_state: 
         st.session_state.state = {"memory": ConversationBufferMemory(memory_key="chat_history")} 
     return st.session_state.state 
state = get_state()

PROMPT = PromptTemplate(
    input_variables=["chat_history","question"], 
    template='Based on the following chat_history, Please reply to the question in format of markdown. history: {chat_history}. question: {question}'
)

source_text = st.text_area("source",placeholder="enter your text you want to chat with")
user_input = st.text_input("You: ",placeholder = "Ask me anything ...")
ask = st.button('ask',type='primary')
    
st.markdown("----")


class SimpleStreamlitCallbackHandler(BaseCallbackHandler):
    """ Copied only streaming part from StreamlitCallbackHandler """
    
    def __init__(self) -> None:
        self.tokens_area = st.empty()
        self.tokens_stream = ""
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.tokens_stream += token
        self.tokens_area.markdown(self.tokens_stream)

handler = SimpleStreamlitCallbackHandler()


if ask:
    res_box = st.empty()
    with st.spinner('typing...'):
        report = []
        chat = ChatOpenAI(streaming=True, temperature=0.9)
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(        
            chunk_size = 500,
            chunk_overlap  = 0,
            length_function = len,
        )
        texts = text_splitter.split_text(source_text)
        
        from langchain.docstore.document import Document
        docs = [Document(page_content=t) for t in texts]
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(docs,embeddings)
        retriever = db.as_retriever()
        
        chain_type_kwargs = {"prompt": PROMPT}
        qa = ConversationalRetrievalChain.from_llm(
            llm=chat, 
            chain_type='stuff',
            retriever=retriever,
            callbacks=[handler]
        )
        chat_history = []
        res = qa({"question": user_input}) #, "chat_history": state['memory']['answer']})
        user_input = ''


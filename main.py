import logging 
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os

from huggingface_hub import hf_hub_download
from langchain import PromptTemplate, LLMChain
from langchain.llms import LlamaCpp

#MODEL ID FROM HUGGINGFACE 
model_id = "TheBloke/Llama-2-7B-Chat-GGML"
model_basename = "llama-2-7b-chat.ggmlv3.q2_K.bin" #IMPORTANT IF YOU USE A GGML MODEL TO RUN THE MODEL IN APPLE SILICON 

#DEVICE TYPE 
device_type = "mps"

def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    
    # test that the API key exists
    if os.getenv("HUGGINGFACEHUB_API_TOKEN") is None or os.getenv("HUGGINGFACEHUB_API_TOKEN") == "":
        print("HUGGINGFACEHUB_API_TOKEN is not set")
        exit(1)
    else:
        print("HUGGINGFACEHUB_API_TOKEN is set")

    # setup streamlit page
    st.set_page_config(
        page_title="Your own Local Llama2 chatbot",
        page_icon="🤖"
    )


def main():
    init()
    def load_model(device_type, model_id, model_basename):
        logging.info("Using Llamacpp for quantized models")
        model_path = hf_hub_download(repo_id= model_id, filename= model_basename)
        if device_type.lower() == "mps":
            return LlamaCpp(
                model_path= model_path,
                n_ctx= 2048,
                max_tokens=2048,
                temperature=0.8,
                n_gpu_layers=1000,
            )

    chat = load_model(device_type, model_id=model_id, model_basename=model_basename)

    #Prompt Template 

    template = """
    
    [INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>

    {question} [/INST]

    """
    prompt = PromptTemplate(template=template, input_variables=['question'])

    st.header("Your own Local Llama2 chatbot on your Macbook M1/M2 🤖")
    with st.chat_message("user"):
        user_input = st.text_input("Your message:", key= "user_input")
    if user_input:
        with st.spinner("Thinking..."):
            prompt = PromptTemplate(template=template, input_variables=['question'])
            llm_chain = LLMChain(llm = chat, prompt=prompt,verbose=True)
            response = llm_chain.run(user_input)
            with st.chat_message(name="assistant"):
                st.write(response)
  
if __name__ == '__main__':
    main()

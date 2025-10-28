import streamlit as st
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

def generate_response(
    uploaded_file,
    openai_api_key,
    query_text,
    response_text
):
    # format uploaded file
    documents = [uploaded_file.read().decode()]
    
    # break it in small chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    texts = text_splitter.create_documents(documents)
    embeddings = OpenAIEmbeddings(
        api_key=openai_api_key
    )
    
    # create a vectorstore and store there the texts
    db = FAISS.from_documents(texts, embeddings)
    
    # create a retriever interface
    retriever = db.as_retriever()
    
    # get relevant documents
    relevant_docs = retriever.invoke(query_text)
    
    # create LLM
    llm = OpenAI(api_key=openai_api_key)
    
    # build context from retrieved documents
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # create prompt
    prompt = f"""Answer the following question based on the provided context:

Context: {context}

Question: {query_text}

Answer:"""
    
    # get answer from LLM
    ai_answer = llm.invoke(prompt)
    
    # prepare response
    response = {
        "question": query_text,
        "expected_answer": response_text,
        "ai_answer": ai_answer
    }
    
    return response

st.set_page_config(
    page_title="Evalua una aplicación RAG",
)
st.title("Evalua una aplicación RAG")

with st.expander("Evalua la calidad de una aplicación RAG"):
    st.write("""
        
Para evaluar la calidad de una aplicación RAG, le haremos preguntas cuyas respuestas reales ya conocemos.
        
De esa manera podemos ver si la aplicación está produciendo las respuestas correctas o si está alucinando.
    """)

uploaded_file = st.file_uploader(
    "Sube un documento de texto para usar como base de conocimiento:",
    type="txt"
)

query_text = st.text_input(
    "Ingresa una pregunta que ya hayas verificado:",
    placeholder="Escribe tu pregunta aquí",
    disabled=not uploaded_file
)

response_text = st.text_input(
    "Introduzca la respuesta real a la pregunta anterior",
    placeholder="Escriba la respuesta verificada aquí",
    disabled=not uploaded_file
)

result = []
with st.form(
    "myform",
    clear_on_submit=True
):
    openai_api_key = st.text_input(
        "OpenAI API Key:",
        type="password",
        disabled=not (uploaded_file and query_text)
    )
    submitted = st.form_submit_button(
        "Enviar",
        disabled=not (uploaded_file and query_text)
    )
    if submitted and openai_api_key.startswith("sk-"):
        with st.spinner(
            "Espere, por favor. Estamos trabajando en ello..."
            ):
            response = generate_response(
                uploaded_file,
                openai_api_key,
                query_text,
                response_text
            )
            result.append(response)
            del openai_api_key
            
if len(result):
     st.write("Question")
    st.info(response["predictions"][0]["question"])
    st.write("Real answer")
    st.info(response["predictions"][0]["answer"])
    st.write("Answer provided by the AI App")
    st.info(response["predictions"][0]["result"])
    st.write("Therefore, the AI App answer was")
    st.info(response["graded_outputs"][0]["results"])

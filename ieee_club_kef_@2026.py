import streamlit as st
import os
import tempfile
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]


def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]


def display_chat_history(chain):
    # Conteneurs pour afficher le chat et le formulaire
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Posez votre question sur le document", key='input')
            submit_button = st.form_submit_button(label='Envoyer')

        if submit_button and user_input:
            with st.spinner('Génération de la réponse...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                st.chat_message("user").text(st.session_state["past"][i])
                st.chat_message("assistant").text(st.session_state["generated"][i])


def create_conversational_chain(vector_store, selected_model, temperature):
    # Dictionnaire de correspondance entre le nom affiché et l'identifiant du modèle
    model_mapping = {
        "deepseek": "deepseek-r1-distill-llama-70b",
        "llamma": "llama-3.1-8b-instant",  # Remplacer par l'identifiant correct pour Llamma
        "gemma": "gemma2-9b-it"  # Remplacer par l'identifiant correct pour Mistral
    }
    model_identifier = model_mapping.get(selected_model, "deepseek-r1-distill-llama-70b")

    llm_judge = ChatGroq(
        temperature=temperature,
        groq_api_key=os.environ.get("GROQ_API_KEY", ""),
        model=model_identifier
    )
    llm_judge.verbose = True
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_judge,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )
    return chain


def main():
    initialize_session_state()
    st.title("Assistant IA pour l'analyse de documents PDF via RAG")

    # Paramètres d'upload dans la sidebar
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Uploader un ou plusieurs fichiers", accept_multiple_files=True)

    # Paramètres du LLM dans la sidebar
    st.sidebar.title("LLM Settings")
    temperature = st.sidebar.slider("Sélectionnez la température", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    selected_model = st.sidebar.selectbox("Sélectionnez le modèle LLM", ["deepseek", "llamma", "gemma"])

    if not uploaded_files:
        st.info("Veuillez uploader un ou plusieurs fichiers pour démarrer la conversation.")
        return

    # Traitement des fichiers uploadés
    text = []
    for file in uploaded_files:
        file_extension = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        loader = None
        if file_extension.lower() == ".pdf":
            loader = PyPDFLoader(temp_file_path)

        if loader:
            text.extend(loader.load())
            os.remove(temp_file_path)

    # Vérification que le document a bien été chargé
    if not text:
        st.error("Impossible de charger les documents. Vérifiez le format ou le contenu des fichiers.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    text_chunks = text_splitter.split_documents(text)

    # Création des embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="distilbert-base-nli-mean-tokens",
        model_kwargs={'device': 'cpu'}
    )

    # Utilisation de ChromaDB pour stocker les documents
    vector_store = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory="./chroma_db")
    vector_store.persist()

    # Création du chain de conversation avec les paramètres spécifiés
    chain = create_conversational_chain(vector_store, selected_model, temperature)

    # Affichage du chat
    display_chat_history(chain)


if __name__ == "__main__":
    main()

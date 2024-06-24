import os
from langchain.chains import ConversationalRetrievalChain
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

api_key = "xxx"
os.environ["OPENAI_API_KEY"] = api_key

embeddings = OpenAIEmbeddings()

# slow
# dir with pdf files
# dir_path = "./data"

# fast
# dir with converted to txt files
dir_path = "./txtdata"



# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = True

# Load documents and create vectorstore
if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")  # if you want to
    vectorstore = Chroma(persist_directory="persist", embedding_function=embeddings)
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = DirectoryLoader(dir_path, show_progress=True)
    if PERSIST:
        print("persist data")
        index = VectorstoreIndexCreator(vectorstore_cls=Chroma, embedding=embeddings, vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])

# Create a Conversational Retrieval Chain
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

# Function to get the response from the LangChain Conversational Retrieval Chain
def get_langchain_response(query, chat_history):
    result = chain({"question": query, "chat_history": chat_history})
    return result['answer']

# Function to simulate the conversation
def simulate_conversation(num_turns=2):
    chat_history = []
    for i in range(num_turns):
        if i % 2 == 0:  # Philosopher's turn
            # query = "What do you think about the impact of technology on society?"
            query = "What is gyber?"  # it knows about it

            print(f"Turn {i + 1} (Philosopher):")
        else:  # Blockchain Expert's turn
            query = "Can you explain how blockchain can enhance data security?"
            print(f"Turn {i + 1} (Blockchain Expert):")

        response = get_langchain_response(query, chat_history)
        print(response)
        chat_history.append((query, response))

if __name__ == '__main__':
    simulate_conversation()

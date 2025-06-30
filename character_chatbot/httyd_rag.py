from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
load_dotenv("../.env")

loader = TextLoader("character_chatbot/HTTYD_knowledge_base.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
chunks = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

db.persist()
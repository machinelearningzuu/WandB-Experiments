import os, yaml, argparse
os.chdir("/Users/1zuu/Desktop/Desktop - Isuru’s Mac mini/ML Research/MLOps Projects/wandb practice/")

from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

with open('secrets.yaml') as f:
    secrets = yaml.safe_load(f)

os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

chat_llm = OpenAI(
                api_key=secrets.get("OPENAI_API_KEY"),
                max_tokens=500
                )

embed_llm = HuggingFaceBgeEmbeddings(
                                    model_name = "BAAI/bge-small-en", 
                                    model_kwargs = {"device": "mps"}, 
                                    encode_kwargs = {"normalize_embeddings": True}
                                    )

loader = PyPDFLoader("00-simple-rag-experiment/2022-annual-report.pdf")

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
                                                chunk_size=500,
                                                chunk_overlap=50
                                                )
texts = text_splitter.split_documents(documents)

docsearch = FAISS.from_documents(
                                texts, 
                                embed_llm
                                )

qa_chain = RetrievalQA.from_chain_type(
                                    llm=chat_llm,
                                    chain_type="stuff",
                                    retriever=docsearch.as_retriever(),
                                )

def ask_question(question):
    response = qa_chain.run(question)
    print_str = ""
    print_str = "### Question\n" + question + "\n"
    print_str += "### Answer\n" + response + "\n"
    print(print_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, help="Question to ask the model")
    args = parser.parse_args()
    ask_question(args.question)

"""
    python 00-simple-rag-experiment/app.py --question "What is the company's revenue?"
"""
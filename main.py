"""
Import required langchain packages
"""
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
import pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

"""
Initialize vector store. A vector store takes input for LLM embeddings in the form of vector array 
and creates a vector space using prompts and context.
"""
pinecone.init(
    api_key=os.getenv("picone_api_key"),
    environment=os.getenv("picone_environment"),
)

if __name__ == "__main__":
    """
    LLM text loader can load the content from HTML, markdown, JSON, PDF and CSV.
    We are loading **Options Trading blogs** from medium as a contextual data.
    """
    loader = TextLoader("resources/medium/options_trading.txt")
    document = loader.load()

    """
    CharacterTextSplitter is the simplest method. This splits based on characters (by default "\n\n") 
    and measure chunk length by number of characters.
    """
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    """
    The Embeddings class is a class designed for interfacing with text embedding models. 
    There are lots of embedding model providers (OpenAI, Cohere, Hugging Face, etc) - this class is designed 
    to provide a standard interface for all of them.
    
    Embeddings create a vector representation of a piece of text.
    It means we can think about text in the vector space, and do things like semantic search where we look for 
    pieces of text that are most similar in the vector space.
    """
    embeddings = OpenAIEmbeddings()

    """
    One of the most common ways to store and search over unstructured data is to embed it and 
    store the resulting embedding vectors, and then at query time to embed the unstructured query 
    and retrieve the embedding vectors that are 'most similar' to the embedded query.
    """
    docsearch = Pinecone.from_documents(texts, embeddings, index_name="my-wiki-index")

    """
    Question answering over an index. You can easily specify different chain types to load and use in the 
    RetrievalQA chain. There are two ways to load different chain types: **from_chain_type** & **map_reduce**.
    You can specify the chain type argument in the **from_chain_type** method. 
    This allows you to pass in the name of the chain type you want to use.
    """
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever()
    )
    query = "What are call and options options? Give me 2 points answer for a begginner"
    result = qa({"query": query})
    print(result)

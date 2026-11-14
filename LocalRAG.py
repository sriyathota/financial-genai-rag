import os
import faiss
import time
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

VECTORSTORE_LOAD_PATH = '/Users/sakethkoona/Documents/Finance VIP/VIP-GenAI/trading_books/db_store'

# Getting a path to our trading books
os.chdir(os.getcwd())
books_dir = "trading_books"
list_of_books = os.listdir(books_dir)

# Reading in our trading books using LangChain's PyPDF Loader
def read_trading_books(books_path, num_books = None):
    print("READING TRADING BOOKS")
    extracted_data = []

    # First, we want to switch to the directory where the books are
    os.chdir(books_path)

    pdfs = sorted(f for f in os.listdir(os.getcwd()) if f.lower().endswith(".pdf"))
    books_to_read = pdfs[:num_books] if num_books else pdfs

    for book in books_to_read:
        print(book)
        loader = PyPDFLoader(book)
        data = loader.load()

        extracted_data.extend(data)


    return [i.page_content for i in extracted_data] # Return type is a list of lists where each inner list is a list of documents from that specific book

def split_trading_books(book_texts, splitter = None):
    print("SPLITTING TRADING BOOKS")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 750,
        chunk_overlap = 100,
        length_function = len,
        is_separator_regex=False
    ) if not splitter else splitter

    chunks = []

    for text in book_texts:
        chunk = text_splitter.split_text(text)
        chunks.extend(chunk)

    return chunks


def get_embeddings(chunked_docs = None, embedder = None, use_gpu = False, only_return_embedder = False):
    # embeddings_model = GoogleGenerativeAIEmbeddings(
    #     model="models/embedding-001", 
    #     google_api_key=os.getenv("GOOGLE_API_KEY")
    #     ) if not embedder else embedder

    embeddings_model = (
        HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},  # Mac safe default
            encode_kwargs={"normalize_embeddings": True},
        )
        if not embedder else embedder
    )


    if not only_return_embedder:
        print("GETTING EMBEDDER AND EMBEDDINGS")
        embeddings = embeddings_model.embed_documents(chunked_docs)
        return embeddings_model, np.array(embeddings)
    else:
        print("GETTING EMBEDDER")
        return embeddings_model


# def get_vectorstore(chunked_data: str = None, save_path: str = None, load_path : str = None, embedder = None): # Here we either create a new vectorsore, or load one in from memory
#     if load_path:
#         print("GETTING PRELOADED VECTORSTORE")
#         db_store = FAISS.load_local(load_path, embedder, allow_dangerous_deserialization=True)
#         return db_store
#     else:
#         print("CREATING VECTORSTORE")
#         embeddings_model, embeddings = get_embeddings(chunked_data, embedder)
#         embeddings = np.array(embeddings)
#         dim = embeddings.shape[1]
#         db_index = faiss.IndexFlatL2(dim)

#         gpu_index = db_index
#         gpu_index.add(embeddings)

#         docs = [Document(page_content=text) for text in chunked_data]
#         docstore = InMemoryDocstore({
#             str(i): docs[i] for i in range(len(docs))
#         })
#         index_to_docstore_id = {i: str(i) for i in range(len(docs))}

#         vectorstore = FAISS(
#             embedding_function=embeddings_model,
#             index=db_index,
#             docstore=docstore,
#             index_to_docstore_id=index_to_docstore_id
#         )

#         vectorstore.save_local(save_path)
#         return vectorstore

# def get_vectorstore(chunked_data, embedder, save_path, force_rebuild=False):
#     index_path = os.path.join(save_path, "faiss_index")

#     if (not force_rebuild) and os.path.exists(index_path):
#         return FAISS.load_local(
#             index_path,
#             embedder,
#             allow_dangerous_deserialization=True
#         )

#     batch_size = 32
#     vectorstore = None

#     for i in range(0, len(chunked_data), batch_size):
#         batch = chunked_data[i:i + batch_size]

#         for attempt in range(3):
#             try:
#                 if vectorstore is None:
#                     vectorstore = FAISS.from_texts(batch, embedder)
#                 else:
#                     vectorstore.add_texts(batch)
#                 break
#             except Exception:
#                 if attempt == 2:
#                     raise
#                 time.sleep(2 * (attempt + 1))

#     vectorstore.save_local(index_path)
#     return vectorstore

def get_vectorstore(chunked_data=None, embedder=None, save_path=None, load_path=None):
    if embedder is None:
        raise ValueError("An embedder is required")

    if load_path is not None:
        print("Loading saved FAISS index...")
        return FAISS.load_local(
            load_path,
            embedder,
            allow_dangerous_deserialization=True,
        )

    if chunked_data is None or save_path is None:
        raise ValueError("chunked_data and save_path are required to build an index")

    print("Embedding all documents...")
    embeddings = embedder.embed_documents(chunked_data)

    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(chunked_data, embeddings)),
        embedding=embedder,
    )
    vectorstore.save_local(save_path)
    return vectorstore

def append_documents_to_vectorstore(new_documents, embedder, index_dir):
    if not new_documents:
        raise ValueError("Provide at least one new document")

    index_dir = os.path.abspath(os.fspath(index_dir))
    for filename in ("index.faiss", "index.pkl"):
        if not os.path.isfile(os.path.join(index_dir, filename)):
            raise FileNotFoundError(f"Missing {filename} in {index_dir}")

    vectorstore = get_vectorstore(load_path=index_dir, embedder=embedder)
    vectorstore.add_documents(new_documents)
    vectorstore.save_local(index_dir)
    return vectorstore


class CrossEncoderRerank(BaseDocumentCompressor):
    model: HuggingFaceCrossEncoder
    top_n: int = 5

    class Config:
        arbitrary_types_allowed = True

    def compress_documents(self, documents, query, callbacks=None):
        if not documents:
            return []
        pairs = [(query, d.page_content) for d in documents]
        scores = self.model.score(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[: self.top_n]]

    async def acompress_documents(self, documents, query, callbacks=None):
        return self.compress_documents(documents, query, callbacks)


def get_retreiver(db_store, k=5, initial_k=20, rerank_model_name="BAAI/bge-reranker-base"):
    if os.getenv("USE_RERANKER", "0") == "0":
        return db_store.as_retriever(search_kwargs={"k": k})
    base = db_store.as_retriever(search_kwargs={"k": initial_k})
    reranker = CrossEncoderRerank(
        model=HuggingFaceCrossEncoder(model_name=rerank_model_name),
        top_n=k,
    )
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base,
    )

USE_HIDDEN_COT = os.getenv("USE_HIDDEN_COT", "1") != "0"

HIDDEN_REASONING_INSTRUCTION = (
    "Use careful internal reasoning to answer using ONLY the provided context. "
    "Do NOT reveal your internal reasoning. Output ONLY the final concise answer. "
    "If the context is insufficient, say: \"I don't know based on the provided context.\""
)

def create_rag_chain(retriever, prompt, llm=None, llm_kwargs=None, use_hidden_cot: bool | None = None):
    model = llm(**(llm_kwargs or {})) if llm else ChatOllama(model="llama3.2", temperature=float(os.getenv("TEMPERATURE", "0")))

    cot_enabled = USE_HIDDEN_COT if use_hidden_cot is None else use_hidden_cot

    prompt_text = prompt
    if cot_enabled:
        prompt_text = prompt_text + "\n\n" + HIDDEN_REASONING_INSTRUCTION

    # system_prompt = ChatPromptTemplate.from_messages([
    #     ("system", prompt_text),
    #     ("human", "{input}")
    # ])

    # chain = create_stuff_documents_chain(model, system_prompt)

    # rag_chain = (
    #     RunnableMap({
    #         "input": lambda x: x["input"],
    #         "context": lambda x: retriever.get_relevant_documents(x["input"])
    #     })
    #     | RunnableLambda(lambda x: {
    #         "answer": chain.invoke({"input": x["input"], "context": x["context"]}),
    #         "retrieved_docs": x["context"]
    #     })
    # )
    template = prompt_text
    if "{context}" not in template:
        template = template + "\n\n{context}"
    template = template + "\n\nQuestion: {question}\nAnswer:"
    qa_prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    rag_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt},
    )

    return rag_chain


def create_whole_pipeline(
    system_prompt,
    vectorstore_load_path=None,
    documents_dir=None,
    vectorstore_save_path=None,
    use_hidden_cot=None,
):
    embedder = get_embeddings(only_return_embedder=True)

    index_dir = vectorstore_load_path or vectorstore_save_path
    if index_dir is None:
        raise ValueError("Provide a vector store load or save path")

    # Resolve the path before read_trading_books changes the working directory.
    index_dir = os.path.abspath(os.fspath(index_dir))
    index_exists = all(
        os.path.isfile(os.path.join(index_dir, filename))
        for filename in ("index.faiss", "index.pkl")
    )

    if index_exists:
        vectorstore = get_vectorstore(load_path=index_dir, embedder=embedder)
    elif vectorstore_load_path is not None:
        raise FileNotFoundError(f"No saved FAISS index found in {index_dir}")
    else:
        data = read_trading_books(documents_dir)
        chunked = split_trading_books(data)
        vectorstore = get_vectorstore(
            chunked_data=chunked,
            embedder=embedder,
            save_path=index_dir,
        )

    retriever = get_retreiver(vectorstore)
    prompt = system_prompt + "\n\n{context}"
    return create_rag_chain(
        retriever,
        prompt,
        use_hidden_cot=use_hidden_cot,
    )

if __name__ == "__main__":
    embedder = get_embeddings(only_return_embedder=True)

    vectorstore = get_vectorstore(load_path=VECTORSTORE_LOAD_PATH, embedder=embedder)
    retreiver = get_retreiver(vectorstore)

    prompt = """
    You are an expert on trading strategies and quantitative trading, and your job is to guide and assist traders, evaluating
    their strategies and determining, whether or not they are profitable, as well as answering their questions.

    Formulate your answers based on both the following context as well as your own knoweledge about trading and investments.

    \n\n{context}
    """

    rag_chain = create_rag_chain(retreiver, prompt)

    sample_prompt = "Give me some sample trading strategies, and tell me if you got that information from the context or your own knoweledge"

    ai_response = rag_chain.invoke({
        "input": sample_prompt
    })

    print(ai_response['answer'])
    

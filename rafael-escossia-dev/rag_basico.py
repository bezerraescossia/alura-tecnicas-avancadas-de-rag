from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import OpenSearchVectorSearch, VectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def embedding_pipeline():
    urls = [
        "https://canaldareceita.com.br/como-fazer/crepioca-simples-com-ovo/",
        "https://www.tuasaude.com/receitas-de-crepioca-para-emagrecer/"
    ]

    # Download dos documentos
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()

    # Criação dos chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # Embedding dos chunks
    embeddings_model = OpenAIEmbeddings(
        model='Qwen/Qwen3-Embedding-0.6B',
        base_url="http://localhost:8000/v1",
        check_embedding_ctx_length=False,
    )

    # indexando no vectorstore
    vectorstore = OpenSearchVectorSearch.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        opensearch_url="http://localhost:9200",
        index_name="index-qwen",
        engine='faiss',
        use_ssl=False,
        verify_certs=False
    )

    return vectorstore


def rag_pipeline(query: str, vectorstore: VectorStore):
    model = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0.5,
    )

    # Retrieve
    retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
    chunks = retriever.invoke(query)
    contexto = "\n\n".join(chunk.page_content for chunk in chunks)

    # Augment
    prompt = ChatPromptTemplate.from_messages(  # type: ignore
        [
            ("system",
             "Responda usando exclusivamente o conteúdo fornecido. \n\nContexto: \n{contexto}"),
            ("human", "{query}")
        ]

    )
    chain = prompt | model | StrOutputParser()  # type: ignore

    return chain.invoke({'query': query, 'contexto': contexto})  # type: ignore


if __name__ == '__main__':
    vectorstore = embedding_pipeline()
    answer = rag_pipeline('Como fazer crepioca?', vectorstore)
    print(answer)

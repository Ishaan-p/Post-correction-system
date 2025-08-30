import os, certifi

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

from langchain.chat_models.openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# from langchain.memory import ConversationSummaryBufferMemory
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA


class QuestionAnsweringSystem:
    def __init__(self):
        # Set up environment variables for API keys
        os.environ["OPENAI_API_KEY"] = "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"
        os.environ["GOOGLE_CSE_ID"] = "25fb733ebf76a4c44"
        os.environ["GOOGLE_API_KEY"] = "AIzaSyAblDI8fMv5J9-8HvZrW_-3n4UVeo6raMs"

        # Initialize components
        self.chat_model = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0,
            streaming=False,   # or True if you really want streams
        )

        self.vector_store = Chroma(
            embedding_function=OpenAIEmbeddings(),
            persist_directory="./chroma_db_oai"
        )

        self.google_search = GoogleSearchAPIWrapper()
        self.web_research_retriever = WebResearchRetriever.from_llm(
            vectorstore=self.vector_store,
            llm=self.chat_model,
            search=self.google_search,
            allow_dangerous_requests=True
        )

        # build a RetrievalQA chain that *stuff*s all retrieved docs into the prompt
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.chat_model,
            chain_type="stuff",                # simple concatenation
            retriever=self.web_research_retriever,
            return_source_documents=True
        )
        # if you want to see the actual prompt:
        self.qa_chain.verbose = True

    def answer_question(self, user_input_question: str):
        # invoke with "query" (default key for RetrievalQA)
        result = self.qa_chain.invoke({"query": user_input_question})

        # the answer text:
        answer = result["result"]

        # pull out whatever metadata you want for your “sources”
        source_docs = result["source_documents"]
        sources = [doc.metadata.get("source", "<no source>") for doc in source_docs]

        return answer, sources

# Example usage:
qa_system = QuestionAnsweringSystem()
user_input_question = input("Ask a question: ")
answer, sources = qa_system.answer_question(user_input_question)
print("Answer:", answer)
print("Sources:", sources)
from crewai import Agent
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_openai import ChatOpenAI

#load same Embedding model and vector store 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chromadb_store", # path where your DB is stored
                      embedding_function=embedding_model)


# LLM
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# Agents
isupport_agent = Agent(
    role="ISupport Agent",
    goal="Resolve software, hardware, and iSupport-related issues",
    backstory="Expert in IT support and troubleshooting.",
    verbose=True,
    llm=llm
)

ssc_agent = Agent(
    role="SSC Agent",
    goal="Handle payroll, tax, PF, and SSC portal-related queries",
    backstory="Specialist in HR and finance-related employee services.",
    verbose=True,
    llm=llm
)

knowledge_agent = Agent(
    role="Knowledge Agent",
    goal="Answer questions about leaves, IT declarations, and innovation policies",
    backstory="HR policy expert with deep knowledge of internal documentation.",
    verbose=True,
    llm=llm
)

supervisor_agent = Agent(
    role="Supervisor Agent",
    goal="Route queries to the appropriate subagent based on topic",
    backstory="Acts as a smart router to delegate tasks to the right expert.",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

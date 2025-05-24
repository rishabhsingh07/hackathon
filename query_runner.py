from crewai import Task, Crew
from agents_setup import isupport_agent, ssc_agent, knowledge_agent, supervisor_agent, vectorstore

def route_query(query):
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in ["isupport", "software", "hardware"]):
        return isupport_agent
    elif any(keyword in query_lower for keyword in ["payroll", "reimbursement", "income-tax", "pf", "salary", "ssc"]):
        return ssc_agent
    elif any(keyword in query_lower for keyword in ["leaves", "it declaration", "innovation"]):
        return knowledge_agent
    else:
        return None

def handle_query(query):
    agent = route_query(query)
    if agent is None:
        print("No suitable agent found.")
        return

    print(f"Routing to: {agent.role}")

    results = vectorstore.similarity_search(query, k=3)
    context = "\n".join([res.page_content for res in results])
    print("Retrieved Context:\n", context)

    task = Task(
        description=f"Answer the following query using the context:\n{query}\n\nContext:\n{context}",
        expected_output="A helpful and accurate answer to the user's query based on the provided context.",
        agent=agent
    )

    crew = Crew(agents=[supervisor_agent, agent], tasks=[task])
    result = crew.kickoff()
    print("Answer:\n", result)

# Example usage
if __name__ == "__main__":
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        handle_query(query)

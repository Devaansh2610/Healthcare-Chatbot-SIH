from src.BOT.workflow.graph import app

if __name__ == "__main__":
    query = "What are the symptoms of a heart attack?"
    inputs = {"messages": [("human", query)]}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Output from node '{key}': {value}")
        print("\n---\n")
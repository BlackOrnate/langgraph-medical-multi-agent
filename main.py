"""Command-line runner: python main.py        run the four test questions
                       python main.py -i     ask questions interactively"""
import asyncio
import sys

from graph import graph


TEST_CASES = [
    "Show me my previous medical visits",
    "I've been coughing for three days and have a slight fever. What should I do?",
    "Which chief physicians are in Orthopedics?",
    "Will the stock market go up today?",
]


async def ask(question: str, user_id: int = 1) -> dict:
    return await graph.ainvoke(
        {"messages": [{"role": "user", "content": question}], "user_id": user_id},
        # Run name, tags and metadata shown in LangSmith, for easier filtering
        config={"run_name": "Medical Consultation", "tags": ["cli"], "metadata": {"user_id": user_id}},
    )


async def run_tests():
    graph.get_graph().print_ascii()
    for question in TEST_CASES:
        print(f"\n{'=' * 20} Question: {question} {'=' * 20}")
        result = await ask(question)
        print(f"\n>>> Category: {result['category']}")
        print(f">>> Final answer:\n{result['final_answer']}")


async def interactive():
    user_id = int(input("Enter user ID (1 / 2 / 99): ") or 1)
    while True:
        question = input("\nEnter a question (press Enter to quit): ").strip()
        if not question:
            break
        result = await ask(question, user_id)
        print(f"\n>>> [{result['category']}]\n{result['final_answer']}")


if __name__ == "__main__":
    asyncio.run(interactive() if "-i" in sys.argv else run_tests())

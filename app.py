"""Gradio demo UI: run `python app.py`, then open http://127.0.0.1:7860"""
import gradio as gr

from graph import graph


CATEGORY_NAMES = {
    "history": "Medical History",
    "guide": "Symptom Consultation",
    "doctor": "Doctor Information",
    "other": "Other",
}


# Use astream to show progress in the UI while the graph is running
async def chat(message, history, user_id):
    category = ""
    answer = ""

    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": message}], "user_id": int(user_id)},
        stream_mode="updates",
        # Run name, tags and metadata shown in LangSmith, for easier filtering
        config={"run_name": "Medical Consultation", "tags": ["gradio"], "metadata": {"user_id": int(user_id)}},
    ):
        for update in chunk.values():
            if not update:
                continue
            if "category" in update:
                category = CATEGORY_NAMES.get(update["category"], update["category"])
                yield f"⏳ Question category: **{category}**, processing..."
            if "final_answer" in update:
                answer = update["final_answer"]

    yield f"**[{category}]**\n\n{answer}"


demo = gr.ChatInterface(
    fn=chat,
    title="🏥 Medical Consultation Multi-Agent",
    description="Look up your visit history, ask about symptoms, or find doctor information. Switch the user ID below to see a different user's visit history.",
    additional_inputs=[
        gr.Dropdown(choices=[1, 2, 99], value=1, label="Current user ID"),
    ],
    examples=[
        ["Show me my previous medical visits", 1],
        ["I've been coughing for three days and have a slight fever. What should I do?", 1],
        ["Which Cardiology doctors are available on Wednesday?", 1],
        ["Will the stock market go up today?", 1],
    ],
)


if __name__ == "__main__":
    demo.launch()

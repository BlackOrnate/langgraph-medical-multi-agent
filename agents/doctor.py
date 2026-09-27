"""Doctor information Agent: the model generates SQL to query a mock doctor database."""
import sqlite3

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain_core.tools import tool

from agents.common import agent_update, ask_agent, create_model
from state import State


# In-memory database with mock data; check_same_thread=False: the Agent may call tools from another thread
conn = sqlite3.connect(":memory:", check_same_thread=False)
conn.execute("""
    CREATE TABLE doctors (
        id INTEGER PRIMARY KEY,
        name TEXT,          -- Doctor's name
        department TEXT,    -- Department
        title TEXT,         -- Title: Chief Physician / Associate Chief Physician / Attending Physician
        schedule TEXT       -- Clinic hours, multiple slots separated by commas
    )
""")
conn.executemany(
    "INSERT INTO doctors (name, department, title, schedule) VALUES (?, ?, ?, ?)",
    [
        ("Zhang Wei", "Cardiology", "Chief Physician", "Monday AM,Wednesday PM"),
        ("Li Na", "Cardiology", "Attending Physician", "Wednesday AM,Friday AM"),
        ("Wang Qiang", "Orthopedics", "Chief Physician", "Tuesday AM,Thursday PM"),
        ("Liu Yang", "Orthopedics", "Associate Chief Physician", "Monday PM,Friday PM"),
        ("Chen Jing", "Respiratory Medicine", "Chief Physician", "Tuesday PM,Thursday AM"),
        ("Zhao Min", "Respiratory Medicine", "Attending Physician", "Wednesday AM,Saturday AM"),
        ("Sun Lei", "Neurology", "Associate Chief Physician", "Monday AM,Thursday AM"),
        ("Zhou Ting", "Pediatrics", "Attending Physician", "Tuesday AM,Wednesday PM,Friday AM"),
    ],
)
conn.commit()


@tool
def get_table_schema() -> str:
    """Get the schema of the doctors table. Call this before generating SQL."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='doctors'"
    ).fetchone()
    return row[0]


@tool
def run_sql(sql: str) -> str:
    """Run a single SELECT statement on the doctor database and return the results."""
    # Only allow queries, so model-generated SQL can't modify or delete data
    if not sql.strip().lower().startswith("select"):
        return "Error: only SELECT statements are allowed"
    try:
        cursor = conn.execute(sql)
    except sqlite3.Error as e:
        # Return the error to the model so it can fix the SQL and retry
        return f"SQL execution error: {e}"

    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()
    if not rows:
        return "The query returned no results"
    # Pair each row with column names so the model knows which field each value belongs to
    return "\n".join(str(dict(zip(columns, row))) for row in rows)


DOCTOR_FAIL_TEXT = "Sorry, doctor information is temporarily unavailable. Please try again later or ask the hospital information desk."

doctor_agent = create_agent(
    model=create_model(),
    tools=[get_table_schema, run_sql],
    system_prompt="""
        You are the hospital's doctor information assistant. You answer questions about doctors' departments, titles and clinic hours.

        Follow these steps:
        1. Call get_table_schema to see the table structure
        2. Generate a SQLite SELECT statement for the user's question and run it with run_sql
           - The schedule column holds multiple comma-separated slots; when filtering by weekday use LIKE, e.g. schedule LIKE '%Wednesday%'
        3. Answer the user in natural language based on the query results

        Answer only from the query results. If the results are empty, tell the user honestly that no matching doctor was found.
    """,
    middleware=[ModelRetryMiddleware(max_retries=2)],
)


async def doctor_node(state: State):
    print("~~~~~~This is Doctor Agent~~~~~~")
    answer = await ask_agent(doctor_agent, state["messages"][0].content, DOCTOR_FAIL_TEXT)
    return agent_update(answer)

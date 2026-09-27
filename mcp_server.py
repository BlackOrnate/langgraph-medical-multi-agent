"""Medical records MCP Server. Run: python mcp_server.py"""
from mcp.server import FastMCP

from config import MCP_HOST, MCP_PORT

mcp = FastMCP("Medical Records", host=MCP_HOST, port=MCP_PORT)


# Mock data: user ID -> list of visit records
MEDICAL_RECORDS = {
    1: [
        {"date": "2026-03-12", "department": "Respiratory Medicine", "diagnosis": "Acute bronchitis", "medicine": "Amoxicillin capsules, compound licorice tablets"},
        {"date": "2026-08-05", "department": "Gastroenterology", "diagnosis": "Chronic gastritis", "medicine": "Omeprazole enteric-coated capsules"},
    ],
    2: [
        {"date": "2025-11-20", "department": "Cardiology", "diagnosis": "Primary hypertension", "medicine": "Amlodipine tablets"},
        {"date": "2026-06-18", "department": "Cardiology", "diagnosis": "Hypertension follow-up, blood pressure well controlled", "medicine": "Amlodipine tablets (continue)"},
        {"date": "2026-09-01", "department": "Orthopedics", "diagnosis": "Lumbar muscle strain", "medicine": "Ibuprofen sustained-release capsules, topical patches"},
    ],
}


@mcp.tool()
def query_medical_records(user_id: int) -> str:
    """Query the given user's medical visit history, sorted from most recent to oldest."""
    records = MEDICAL_RECORDS.get(user_id)
    if not records:
        return "No medical records found for this user"

    records = sorted(records, key=lambda r: r["date"], reverse=True)
    return "\n".join(
        f"Date: {r['date']}, Department: {r['department']}, Diagnosis: {r['diagnosis']}, Medication: {r['medicine']}"
        for r in records
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

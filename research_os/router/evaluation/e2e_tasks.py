"""End-to-end scenarios. Success = correct agent AND correct tool call."""
from __future__ import annotations

# intent -> (agent, tool expected among the calls, or None meaning no tool calls at all)
EXPECT = {
    "BOOK_APPOINTMENT": ("booking_agent", "book_appointment"),
    "RESCHEDULE_APPOINTMENT": ("booking_agent", "reschedule_appointment"),
    "CANCEL_APPOINTMENT": ("booking_agent", "cancel_appointment"),
    "PRESCRIPTION_QUERY": ("prescription_agent", "get_prescription"),
    "FOLLOWUP_QUERY": ("prescription_agent", "get_followup"),
    "MEDICAL_RESEARCH": ("research_agent", "search_pubmed"),
    "GENERAL_QUERY": ("faq_agent", None),
    "EMERGENCY": ("emergency_agent", None),
}

_SCENARIOS = [
    # clear utterances
    ("I want to see Dr Patel on Friday", "BOOK_APPOINTMENT"),
    ("can you book me with a dermatologist tomorrow", "BOOK_APPOINTMENT"),
    ("Can I move my appointment to Friday?", "RESCHEDULE_APPOINTMENT"),
    ("i need to reschedule my visit to thursday", "RESCHEDULE_APPOINTMENT"),
    ("Cancel my appointment", "CANCEL_APPOINTMENT"),
    ("um please cancel my visit with Dr Sharma", "CANCEL_APPOINTMENT"),
    ("What medicines did the doctor give me?", "PRESCRIPTION_QUERY"),
    ("how many times a day should i take metformin", "PRESCRIPTION_QUERY"),
    ("When should I follow up?", "FOLLOWUP_QUERY"),
    ("when is my next checkup due", "FOLLOWUP_QUERY"),
    ("What are the latest treatments for migraine?", "MEDICAL_RESEARCH"),
    ("any new research on type 2 diabetes", "MEDICAL_RESEARCH"),
    ("What services do you provide?", "GENERAL_QUERY"),
    ("what are your opening hours", "GENERAL_QUERY"),
    ("I'm having severe chest pain", "EMERGENCY"),
    ("help someone is unconscious and not breathing", "EMERGENCY"),
    # hard_test.csv utterances (ambiguous / multi-intent), verbatim
    ("i booked thursday but i also have a meeting then is there any other slot", "RESCHEDULE_APPOINTMENT"),
    ("my mother needs to see a heart specialist as soon as possible this month", "BOOK_APPOINTMENT"),
    ("i don't think i need to come in on monday anymore the rash is gone", "CANCEL_APPOINTMENT"),
    ("i ran out of my pills what was the dose again", "PRESCRIPTION_QUERY"),
    ("did the doctor want me back after the scan or was that optional", "FOLLOWUP_QUERY"),
    ("i heard a new migraine injection came out how does it compare to what's used now", "MEDICAL_RESEARCH"),
    ("do you treat children too or only adults", "GENERAL_QUERY"),
    ("my chest has been feeling tight since this morning and it's spreading to my left arm", "EMERGENCY"),
    ("i have an appointment tomorrow but right now my chest hurts badly", "EMERGENCY"),
    ("i want to cancel my slot because i think i'm having a heart attack", "EMERGENCY"),
    ("i need to book a visit but first my vision just went completely black", "EMERGENCY"),
    ("what is the dosage for my tablets because i took six by mistake and feel strange", "EMERGENCY"),
    # more clear ones for balance
    ("schedule a consultation with Dr Rao for 2 pm", "BOOK_APPOINTMENT"),
    ("push my appointment to monday", "RESCHEDULE_APPOINTMENT"),
    ("i won't be able to come tomorrow, cancel it", "CANCEL_APPOINTMENT"),
    ("list my current medications", "PRESCRIPTION_QUERY"),
    ("do i have a follow up scheduled", "FOLLOWUP_QUERY"),
    ("summarize recent papers on asthma", "MEDICAL_RESEARCH"),
    ("where is the clinic located", "GENERAL_QUERY"),
    ("i can't breathe", "EMERGENCY"),
    ("my wife just collapsed", "EMERGENCY"),
    ("is there a cardiologist at your clinic", "GENERAL_QUERY"),
    ("tell me my prescribed medicines for hypertension", "PRESCRIPTION_QUERY"),
    ("find recent clinical trials for psoriasis", "MEDICAL_RESEARCH"),
]

E2E_TASKS = [
    {"text": t, "expected_intent": i, "expected_agent": EXPECT[i][0], "expected_tool": EXPECT[i][1]}
    for t, i in _SCENARIOS
]


def task_success(state: dict, task: dict) -> bool:
    """Correct agent and correct tool call (no tool calls at all when none is expected)."""
    if state.get("agent_name") != task["expected_agent"]:
        return False
    names = [c["name"] for c in state.get("tool_calls", [])]
    return not names if task["expected_tool"] is None else task["expected_tool"] in names

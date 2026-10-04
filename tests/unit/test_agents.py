import pytest

from research_os.legacy.agents.booking import BookingAgent
from research_os.legacy.agents.emergency import EmergencyAgent
from research_os.legacy.agents.faq import FAQAgent
from research_os.legacy.agents.prescription import PrescriptionAgent
from research_os.legacy.agents.research import ResearchAgent
from research_os.legacy.agents.tools import ToolBox
from research_os.llm.mock_client import MockLLMClient


@pytest.mark.parametrize("llm", [None, MockLLMClient()])
def test_booking_book(llm):
    r = BookingAgent().run("I want to see Dr Sharma on Friday", ToolBox(), llm, "BOOK_APPOINTMENT")
    assert r["agent"] == "booking_agent"
    assert [c["name"] for c in r["tool_calls"]][-1] == "book_appointment"
    assert "booked" in r["response"].lower() and "Dr Sharma" in r["response"]
    assert r["tool_calls"][-1]["name"] == "book_appointment"


def test_booking_reschedule():
    r = BookingAgent().run("Can I move my appointment to Friday?", ToolBox(), None,
                           "RESCHEDULE_APPOINTMENT")
    assert [c["name"] for c in r["tool_calls"]][-1] == "reschedule_appointment"
    assert "friday" in r["response"].lower()


def test_booking_cancel():
    r = BookingAgent().run("cancel my appointment", ToolBox(), None, "CANCEL_APPOINTMENT")
    assert r["tool_calls"][-1]["name"] == "cancel_appointment"


def test_booking_cancel_with_nothing_booked_is_graceful():
    tb = ToolBox()
    tb.cancel_appointment("P001")
    r = BookingAgent().run("cancel my appointment", tb, None, "CANCEL_APPOINTMENT")
    assert "no appointment" in r["response"].lower()


def test_booking_when_no_slot_available_is_graceful():
    r = BookingAgent().run("book Dr Nobody on friday", ToolBox(), None, "BOOK_APPOINTMENT")
    assert all(c["name"] != "book_appointment" for c in r["tool_calls"])
    assert "dr nobody" in r["response"].lower() and "roster" in r["response"].lower()


def test_prescription_agent_routes_by_intent():
    a = PrescriptionAgent()
    p = a.run("what medicines did the doctor give me", ToolBox(), None, "PRESCRIPTION_QUERY")
    f = a.run("when should I follow up", ToolBox(), None, "FOLLOWUP_QUERY")
    assert p["tool_calls"][0]["name"] == "get_prescription"
    assert f["tool_calls"][0]["name"] == "get_followup"


def test_research_agent():
    r = ResearchAgent().run("latest treatments for migraine", ToolBox(), MockLLMClient())
    assert r["tool_calls"][0]["name"] == "search_pubmed" and r["response"]


def test_emergency_agent_makes_no_tool_calls_or_diagnosis():
    r = EmergencyAgent().run("I'm having severe chest pain", ToolBox(), MockLLMClient())
    assert r["tool_calls"] == [] and r["agent"] == "emergency_agent"
    assert "emergency services" in r["response"].lower()


def test_faq_agent_answers_known_and_unknown():
    a = FAQAgent()
    assert "open" in a.run("what are your opening hours", ToolBox(), None)["response"].lower()
    assert a.run("blorp", ToolBox(), None)["response"]


# ---- review fixes: STT-style doctor names and unresolvable constraints ----

def test_doctor_spelled_out_is_matched():
    r = BookingAgent().run("i wanna see doctor sharma tomorrow", ToolBox(), None, "BOOK_APPOINTMENT")
    assert "Dr Sharma" in r["response"]


def test_unknown_doctor_spelled_out_is_flagged():
    r = BookingAgent().run("book doctor nobody on friday", ToolBox(), None, "BOOK_APPOINTMENT")
    assert "roster" in r["response"].lower()
    assert all(c["name"] != "book_appointment" for c in r["tool_calls"])


def test_unsupported_day_asks_instead_of_booking_something_else():
    tb = ToolBox()
    r = BookingAgent().run("move it to sunday", tb, None, "RESCHEDULE_APPOINTMENT")
    assert "sunday" in r["response"].lower() and "monday" in r["response"].lower()
    assert all(c["name"] != "reschedule_appointment" for c in r["tool_calls"])
    assert tb.get_appointment("P001")["day"] == "wednesday"  # unchanged


def test_requested_time_unavailable_is_not_silently_replaced():
    tb = ToolBox()
    r = BookingAgent().run("book dr sharma on wednesday at 10:30", tb, None, "BOOK_APPOINTMENT")
    assert "10:30" in r["response"] and "not available" in r["response"].lower()
    assert all(c["name"] != "book_appointment" for c in r["tool_calls"])
    assert len(tb.appointments["P001"]) == 1


def test_reschedule_with_unavailable_time_does_not_ignore_time():
    r = BookingAgent().run("move my appointment to friday at 3:15", ToolBox(), None,
                           "RESCHEDULE_APPOINTMENT")
    assert "3:15" in r["response"] or "15:15" in r["response"]
    assert "Done" not in r["response"]


def test_time_parsing_prefers_real_times_over_other_numbers():
    from research_os.legacy.agents.parsing import extract_time
    assert extract_time("book me for 2 people at 3 pm with dr rao") == "15:00"
    assert extract_time("at 2:00") == "14:00"      # bare 1-6 hours read as afternoon
    assert extract_time("10:30 please") == "10:30"
    assert extract_time("9 am") == "09:00"
    assert extract_time("friday") is None

from research_os.legacy.agents.tools import ToolBox


def test_check_slots_filters_by_doctor_and_day():
    tb = ToolBox()
    all_slots = tb.check_slots()["slots"]
    sharma = tb.check_slots(doctor="Dr Sharma")["slots"]
    assert sharma and all(s["doctor"] == "Dr Sharma" for s in sharma)
    friday = tb.check_slots(day="friday")["slots"]
    assert friday and all(s["day"] == "friday" for s in friday)
    assert len(all_slots) > len(sharma)


def test_book_then_slot_is_taken():
    tb = ToolBox()
    slot = tb.check_slots(doctor="Dr Rao", day="monday")["slots"][0]
    r = tb.book_appointment("P002", slot["doctor"], slot["day"], slot["time"])
    assert r["ok"] and r["appointment"]["doctor"] == "Dr Rao"
    again = tb.book_appointment("P003", slot["doctor"], slot["day"], slot["time"])
    assert "error" in again


def test_book_unknown_patient_and_bad_slot():
    tb = ToolBox()
    assert "error" in tb.book_appointment("NOPE", "Dr Rao", "monday", "09:00")
    assert "error" in tb.book_appointment("P002", "Dr Nobody", "monday", "09:00")


def test_reschedule_moves_appointment_and_frees_old_slot():
    tb = ToolBox()
    old = tb.get_appointment("P001")
    r = tb.reschedule_appointment("P001", day="friday")
    assert r["ok"] and r["appointment"]["day"] == "friday"
    freed = tb.check_slots(doctor=old["doctor"], day=old["day"])["slots"]
    assert any(s["time"] == old["time"] for s in freed)


def test_reschedule_without_appointment_is_error_not_exception():
    assert "error" in ToolBox().reschedule_appointment("P002", day="friday")


def test_cancel_and_double_cancel():
    tb = ToolBox()
    assert tb.cancel_appointment("P001")["ok"]
    assert "error" in tb.cancel_appointment("P001")


def test_prescription_and_followup():
    tb = ToolBox()
    assert tb.get_prescription("P001")["medications"]
    assert tb.get_followup("P001")["followup"]
    assert "error" in tb.get_prescription("NOPE")
    assert "error" in tb.get_followup("NOPE")


def test_research_tools_return_canned_results():
    tb = ToolBox()
    assert tb.search_pubmed("migraine treatment")["results"]
    assert tb.search_web("migraine")["results"]


def test_patient_can_hold_multiple_appointments():
    tb = ToolBox()
    slot = tb.check_slots(doctor="Dr Rao", day="monday")["slots"][0]
    assert tb.book_appointment("P001", slot["doctor"], slot["day"], slot["time"])["ok"]
    assert len(tb.appointments["P001"]) == 2

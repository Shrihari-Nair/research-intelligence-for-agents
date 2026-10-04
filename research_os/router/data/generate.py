"""Programmatic synthetic utterance generator (all data is synthetic)."""
from __future__ import annotations

import random
import re

import pandas as pd

from research_os.router.data.validate import normalize

DOCS = ["Dr Sharma", "Dr Patel", "Dr Rao", "Dr Iyer", "Dr Mehta", "Dr Khan", "Dr Nair",
        "Dr Gupta", "Dr Reddy", "Dr Singh", "Dr Fernandes", "Dr Bose"]
SPECS = ["cardiologist", "dermatologist", "dentist", "eye doctor", "orthopedic specialist",
         "pediatrician", "physiotherapist", "neurologist", "gynecologist", "ENT specialist",
         "general physician", "psychiatrist"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "tomorrow",
        "next week", "this evening", "the day after tomorrow", "next Monday", "this Friday"]
DATES = ["the 5th", "March 3rd", "the 12th of next month", "June 18", "the 21st", "April 9",
         "the 30th", "July 2nd"]
TIMES = ["9 am", "10:30", "2 pm", "after lunch", "early morning", "4 in the evening",
         "around noon", "6 pm", "first thing", "11 am"]
DRUGS = ["metformin", "amoxicillin", "ibuprofen", "atorvastatin", "paracetamol", "omeprazole",
         "amlodipine", "cetirizine", "azithromycin", "levothyroxine", "insulin", "losartan",
         "pantoprazole", "aspirin"]
CONDS = ["migraine", "type 2 diabetes", "hypertension", "asthma", "arthritis", "PCOS",
         "thyroid disorder", "depression", "eczema", "high cholesterol", "sleep apnea",
         "chronic back pain", "psoriasis", "anemia", "Alzheimer's disease", "IBS"]
SYMPTOMS = ["chest pain", "chest tightness", "trouble breathing", "heavy bleeding",
            "a seizure", "sudden numbness on one side", "severe stomach pain",
            "a really high fever", "sudden vision loss", "blacking out"]

PREFIX = ["um ", "so ", "uh ", "hi, ", "hello ", "hey ", "okay so ", "yeah ", "well ",
          "excuse me, ", "good morning, ", "hi there, "]
SUFFIX = [" please", " thanks", " if possible", " when you get a chance", " asap", " thank you",
          " if that's okay"]

T: dict[str, list[str]] = {
    "BOOK_APPOINTMENT": [
        "I want to see {doc} {day}", "can I book an appointment with {doc}",
        "book me a slot with {doc} on {day}", "I need to schedule a visit with a {spec}",
        "is {doc} available {day} at {time}", "set up an appointment for {day} at {time}",
        "I'd like to make an appointment with the {spec}", "can you fit me in {day}",
        "I need a checkup {day}", "get me in to see {doc} on {date}",
        "schedule a consultation with {doc} for {time}",
        "any openings with a {spec} {day}", "I want to book a visit for my {cond}",
        "need an appointment with {doc} {day} {time}", "can I get a new appointment {day}",
        "make me an appointment for {date} at {time}",
        "I'd like to see a {spec} about my {cond}", "put me down for {doc} on {day}",
        "I'm looking to book a {spec} {day}", "reserve a slot {day} {time}",
    ],
    "RESCHEDULE_APPOINTMENT": [
        "can I move my appointment to {day}", "I need to reschedule my visit with {doc}",
        "change my appointment from {day} to {date}", "can we push my appointment to {time}",
        "I can't make it {day} can we do {day2} instead", "move my {time} appointment to {day}",
        "reschedule me with {doc} to {day}", "is it possible to shift my appointment to {date}",
        "I want to change the time of my appointment to {time}",
        "can I take a different slot with {doc}", "need to postpone my appointment till {day}",
        "my {day} appointment doesn't work anymore can you change it to {day2}",
        "swap my appointment to {day} at {time}", "can I get my visit moved to {date}",
        "something came up, can we reschedule {doc} to {day}",
        "I'd like an earlier slot than {time}", "bring my appointment forward to {day}",
        "change my booking to {day2} {time}", "can you rebook me for {date} instead",
        "I have to shift my {spec} visit to {day}",
    ],
    "CANCEL_APPOINTMENT": [
        "cancel my appointment", "please cancel my visit with {doc}",
        "I want to cancel my {day} appointment", "I won't be able to come {day}, cancel it",
        "cancel the {time} slot", "remove my appointment on {date}",
        "I no longer need the appointment with {doc}", "call off my visit {day}",
        "delete my booking for {day}", "I need to cancel my appointment with the {spec}",
        "please drop my {time} appointment", "scrap my appointment, I'm fine now",
        "cancel my {date} booking", "I'd like to cancel {doc}'s appointment",
        "can you cancel my appointment for {day}", "forget the appointment {day} cancel it",
        "unbook my slot with {doc}", "I changed my mind, cancel my visit",
        "cancel everything I have booked with {doc}", "I need to cancel my {spec} appointment",
    ],
    "PRESCRIPTION_QUERY": [
        "what medicines did the doctor give me", "what's my prescription",
        "how many times a day should I take {drug}", "what is the dosage of {drug} I was given",
        "can you read out my latest prescription", "which tablets am I supposed to take for {cond}",
        "when should I take {drug}, before or after food", "what did {doc} prescribe for me",
        "how long do I have to continue {drug}", "I forgot my dosage for {drug}",
        "list my current medications", "is {drug} part of my prescription",
        "what are the instructions for {drug}", "tell me my prescribed medicines for {cond}",
        "how many {drug} tablets are left on my prescription",
        "did {doc} change my {drug} dose", "what was I prescribed last visit",
        "remind me what pills I take in the morning", "what's the strength of my {drug}",
        "I need to know the medicine {doc} wrote for my {cond}",
    ],
    "FOLLOWUP_QUERY": [
        "when should I follow up", "when is my next follow up visit",
        "do I need to come back for a checkup", "when do I need to see {doc} again",
        "how soon should I return after my {cond} treatment", "is a follow up needed for my {cond}",
        "what's the follow up date for my last visit", "when should I get my next review",
        "do I have a follow up scheduled", "how long until I need to revisit {doc}",
        "should I come back in two weeks", "when is my next checkup due",
        "what did {doc} say about my next visit", "when do I need to get my blood test repeated",
        "tell me my follow up plan for {cond}", "do I have to return for another review",
        "is there a recheck after my {spec} visit", "when's my next review with {doc}",
        "how often should I come for {cond} monitoring", "what's my next follow up appointment",
    ],
    "MEDICAL_RESEARCH": [
        "what are the latest treatments for {cond}", "any new research on {cond}",
        "what are the side effects of {drug} according to studies",
        "find recent clinical trials for {cond}", "what does the evidence say about {drug} for {cond}",
        "summarize recent papers on {cond}", "is there a new guideline for {cond}",
        "how effective is {drug} in treating {cond}", "what causes {cond} according to research",
        "search pubmed for {cond} treatment", "are there any breakthroughs in {cond}",
        "compare {drug} with other drugs for {cond}", "what is the current medical consensus on {cond}",
        "look up studies about {drug} and {cond}", "what's new in {cond} therapy",
        "what do the latest studies say about {cond} and diet", "is {drug} safe long term per research",
        "what are the risk factors for {cond}", "find me literature on {cond} in women",
        "has anything changed in how {cond} is treated recently",
    ],
    "GENERAL_QUERY": [
        "what services do you provide", "what are your opening hours", "where is the clinic located",
        "do you accept insurance", "how much does a consultation cost",
        "do you have a parking facility", "which specialists work here",
        "is the clinic open on Sundays", "what's your phone number", "do you offer lab tests",
        "how do I get my medical records", "what payment methods do you take",
        "do you have a pharmacy on site", "can I bring a family member with me",
        "is there a {spec} at your clinic", "how do I reach you by public transport",
        "do you do telemedicine", "what should I bring to my first visit",
        "do you have wheelchair access", "who runs this clinic", "tell me about your clinic",
        "are you open on public holidays", "what vaccines do you offer",
    ],
    "EMERGENCY": [
        "I'm having severe {sym}", "I have {sym} right now", "help I think I'm having a heart attack",
        "someone is unconscious and not breathing", "I took too many pills I think I overdosed",
        "my father is having {sym}", "it's an emergency I need help now",
        "I can't breathe properly", "my child swallowed something and is choking",
        "there's heavy bleeding and it won't stop", "I think I'm having a stroke",
        "I'm experiencing {sym} and I'm scared", "call an ambulance please",
        "my wife just collapsed", "I feel like I'm going to pass out and my chest hurts",
        "I have {sym} and my arm feels numb", "emergency, he is not responding",
        "she is having {sym} what do I do", "I'm having thoughts of ending my life",
        "severe allergic reaction, my throat is closing", "i have {sym} help me",
        "my baby is turning blue", "I just had {sym} out of nowhere",
    ],
}

# Hand-written colloquial / indirect lines (extra diversity, added verbatim).
EXTRA: dict[str, list[str]] = {
    "BOOK_APPOINTMENT": [
        "my knee's been acting up, need someone to look at it this week",
        "I should probably get this mole checked, any chance of a visit",
        "got a free afternoon thursday, can the doctor see me",
        "need to get in front of a doctor soon",
    ],
    "RESCHEDULE_APPOINTMENT": [
        "work has been crazy, can my visit slide to next week",
        "that slot clashes with my meeting, any other time",
        "can we do a different day for my appointment",
        "i'm stuck in traffic duty that day, push it a bit",
    ],
    "CANCEL_APPOINTMENT": [
        "never mind about my appointment",
        "i'm feeling better so no need to come in anymore",
        "yeah i'm not going to make it, you can free up my slot",
        "forget that booking",
    ],
    "PRESCRIPTION_QUERY": [
        "what was that tablet the doctor wrote for my sugar",
        "those white pills i'm on, how many do i take",
        "can you tell me what i'm supposed to be taking",
        "remind me about my meds",
    ],
    "FOLLOWUP_QUERY": [
        "do i need to come back or am i good",
        "am I done with treatment or is there another visit",
        "so when do they want to see me again",
        "is there a next visit coming up",
    ],
    "MEDICAL_RESEARCH": [
        "read me something recent about new diabetes drugs",
        "what's the science behind intermittent fasting for blood pressure",
        "is there any new evidence on vitamin d and depression",
        "how do doctors currently treat long covid",
    ],
    "GENERAL_QUERY": [
        "hi, what can you help me with",
        "what do you guys do",
        "is it a walk in clinic or do I need a booking system",
        "who do I talk to about billing",
    ],
    "EMERGENCY": [
        "i cant breathe",
        "chest feels like its being crushed",
        "oh god he's not waking up",
        "i think i'm dying",
        "my heart is racing and i feel faint call someone",
    ],
}


def _fill(template: str, rng: random.Random) -> str:
    day, day2 = rng.sample(DAYS, 2)
    return template.format(
        doc=rng.choice(DOCS), spec=rng.choice(SPECS), day=day, day2=day2,
        date=rng.choice(DATES), time=rng.choice(TIMES), drug=rng.choice(DRUGS),
        cond=rng.choice(CONDS), sym=rng.choice(SYMPTOMS),
    )


def _style(text: str, rng: random.Random) -> str:
    """Apply speech-style fillers and STT-style casing/punctuation."""
    if rng.random() < 0.25:
        text = rng.choice(PREFIX) + text
    if rng.random() < 0.15:
        text += rng.choice(SUFFIX)
    r = rng.random()
    if r < 0.35:  # STT-like: lowercase, no punctuation
        text = re.sub(r"[^\w\s']", "", text.lower())
    elif r < 0.85:
        text = text[0].upper() + text[1:]
        if rng.random() < 0.5 and not text.endswith("?"):
            text += "?" if text.split()[0].lower() in {"what", "when", "how", "can", "is", "do",
                                                         "where", "which", "are", "who"} else "."
    return text.strip()


def _wordcount_ok(t: str) -> bool:
    return 3 <= len(t.split()) <= 30


def generate(seed: int, per_intent: int = 110) -> pd.DataFrame:
    """Return a DataFrame(text, intent) with `per_intent` unique rows per intent."""
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    for intent, templates in T.items():
        seen: set[str] = set()
        out: list[str] = []
        for t in EXTRA[intent]:
            if normalize(t) not in seen and _wordcount_ok(t):
                seen.add(normalize(t))
                out.append(t)
        attempts = 0
        while len(out) < per_intent:
            attempts += 1
            if attempts > 200_000:
                raise RuntimeError(f"could not reach {per_intent} unique rows for {intent}")
            text = _style(_fill(rng.choice(templates), rng), rng)
            n = normalize(text)
            if n in seen or not _wordcount_ok(text):
                continue
            seen.add(n)
            out.append(text)
        rows += [(t, intent) for t in out]
    df = pd.DataFrame(rows, columns=["text", "intent"])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

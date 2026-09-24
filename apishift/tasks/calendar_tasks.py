"""Calendar task templates.

Lookup rule for every template: the target sits at index >= 3 in the listing
the oracle plan uses, so pagination changes are never inert.
"""

from __future__ import annotations

import random

from apishift.tasks.model import Call, Expect, Unordered, pick
from apishift.tasks.pools import MEETINGS, TASK_DAYS, hex_id, hhmm, insert_at, people

LOCATIONS = (None, "Room 4B", "Zoom", "HQ Atrium", "Cafe Lounge")


def _contacts(rng, n=7):
    return [{"id": hex_id(rng, "con"), "name": nm, "email": em} for nm, em in people(rng, n)]


def _event(rng, title, date, start_min, dur, attendees):
    return {
        "id": hex_id(rng, "evt"),
        "title": title,
        "start": f"{date} {hhmm(start_min)}",
        "end": f"{date} {hhmm(start_min + dur)}",
        "attendees": list(attendees),
        "location": rng.choice(LOCATIONS),
        "status": "confirmed",
    }


def _world(rng, *, contacts=None, exclude_email=None):
    """Contacts plus events: a target on `date` preceded by >= 3 same-day events."""
    contacts = contacts if contacts is not None else _contacts(rng)
    emails = [c["email"] for c in contacts if c["email"] != exclude_email]
    date = rng.choice(TASK_DAYS)
    other_days = [d for d in TASK_DAYS if d != date]
    titles = rng.sample(MEETINGS, 6)
    slots = rng.sample(range(8 * 60, 17 * 60, 30), 6)

    def ev(title, day, slot):
        return _event(rng, title, day, slot, rng.choice((30, 60)), rng.sample(emails, rng.randint(1, 3)))

    n_before = rng.randint(3, 4)
    before = [ev(titles[1 + i], date, slots[1 + i]) for i in range(n_before)]
    target = ev(titles[0], date, slots[0])
    after = [ev(titles[5], date, slots[5])] if rng.random() < 0.5 else []
    others = [ev(rng.choice(MEETINGS), rng.choice(other_days), rng.choice(slots)) for _ in range(3)]
    near_miss = ev(titles[0], rng.choice(other_days), slots[0])
    events = insert_at(rng, [*before, target, *after], [*others, near_miss])
    return contacts, events, target, date, slots


def _state(contacts, events):
    return {
        "contacts": {c["id"]: c for c in contacts},
        "events": {e["id"]: e for e in events},
        "reminders": {},
    }


def _target_contact(rng, contacts):
    return contacts[rng.randint(3, len(contacts) - 1)]


def create_meeting(rng: random.Random):
    contacts, events, _, date, slots = _world(rng)
    who = _target_contact(rng, contacts)
    title = rng.choice([m for m in MEETINGS if m not in {e["title"] for e in events}])
    start = rng.choice([s for s in range(9 * 60, 17 * 60, 30) if s not in slots])
    dur = rng.choice((30, 45, 60))
    s, e = f"{date} {hhmm(start)}", f"{date} {hhmm(start + dur)}"
    instruction = (f"Schedule a {dur}-minute meeting titled '{title}' with {who['name']} "
                   f"on {date} from {hhmm(start)} to {hhmm(start + dur)} UTC.")

    def plan():
        body = yield Call("list_contacts", {}, match=lambda c: c["name"] == who["name"])
        contact = pick(body, lambda c: c["name"] == who["name"])
        yield Call("create_event", {"title": title, "start": s, "end": e, "attendees": [contact["email"]]})

    expects = (Expect("events", "create", {"title": title, "start": s, "end": e,
                                           "attendees": Unordered((who["email"],)),
                                           "status": "confirmed"}),)
    return instruction, _state(contacts, events), expects, plan


def _find_target(date, title):
    match = lambda ev: ev["title"] == title  # noqa: E731
    return Call("list_events", {"date": date}, match=match), match


def cancel_event(rng: random.Random):
    contacts, events, target, date, _ = _world(rng)
    title = target["title"]
    instruction = f"Cancel the '{title}' event on {date}."

    def plan():
        call, match = _find_target(date, title)
        ev = pick((yield call), match)
        yield Call("cancel_event", {"event_id": ev["id"]})

    expects = (Expect("events", "update", {"status": "cancelled"}, record_id=target["id"]),)
    return instruction, _state(contacts, events), expects, plan


def reschedule_event(rng: random.Random):
    contacts, events, target, date, slots = _world(rng)
    title = target["title"]
    dur = rng.choice((30, 60))
    start = rng.choice([s for s in range(8 * 60, 18 * 60, 30) if s not in slots])
    s, e = f"{date} {hhmm(start)}", f"{date} {hhmm(start + dur)}"
    instruction = (f"Move the '{title}' event on {date} to {hhmm(start)}-{hhmm(start + dur)} UTC "
                   f"the same day.")

    def plan():
        call, match = _find_target(date, title)
        ev = pick((yield call), match)
        yield Call("update_event", {"event_id": ev["id"], "start": s, "end": e})

    expects = (Expect("events", "update", {"start": s, "end": e}, record_id=target["id"]),)
    return instruction, _state(contacts, events), expects, plan


def add_attendee(rng: random.Random):
    contacts = _contacts(rng)
    who = _target_contact(rng, contacts)
    _, events, target, date, _ = _world(rng, contacts=contacts, exclude_email=who["email"])
    title = target["title"]
    instruction = f"Add {who['name']} to the '{title}' meeting on {date}."

    def plan():
        body = yield Call("list_contacts", {}, match=lambda c: c["name"] == who["name"])
        contact = pick(body, lambda c: c["name"] == who["name"])
        call, match = _find_target(date, title)
        ev = pick((yield call), match)
        yield Call("add_attendee", {"event_id": ev["id"], "email": contact["email"]})

    new_attendees = Unordered((*target["attendees"], who["email"]))
    expects = (Expect("events", "update", {"attendees": new_attendees}, record_id=target["id"]),)
    return instruction, _state(contacts, events), expects, plan


def create_reminder(rng: random.Random):
    contacts, events, target, date, _ = _world(rng)
    title = target["title"]
    minutes = rng.choice((10, 15, 30, 45, 60, 90, 120))
    instruction = f"Set a reminder {minutes} minutes before the '{title}' event on {date}."

    def plan():
        call, match = _find_target(date, title)
        ev = pick((yield call), match)
        yield Call("create_reminder", {"event_id": ev["id"], "minutes_before": minutes})

    expects = (Expect("reminders", "create", {"event_id": target["id"], "minutes_before": minutes}),)
    return instruction, _state(contacts, events), expects, plan


TEMPLATES = {
    "create_meeting": create_meeting,
    "cancel_event": cancel_event,
    "reschedule_event": reschedule_event,
    "add_attendee": add_attendee,
    "create_reminder": create_reminder,
}

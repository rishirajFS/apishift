"""Calendar mock API: events, contacts, reminders."""

from __future__ import annotations

from apishift.envs.domain import Domain, fetch, put
from apishift.envs.errors import conflict, invalid, not_found
from apishift.envs.formats import validate_format
from apishift.envs.mutations import (
    Deprecation,
    FixedParam,
    FormatChangeSpec,
    NewRequiredField,
    PaginationChange,
    RenameParam,
)
from apishift.envs.schema import Endpoint, Param

EVENT_RETURNS = "Event object {id, title, start, end, attendees, location, status}."
TIMEZONES = ("UTC", "America/New_York", "America/Los_Angeles", "Europe/London", "Asia/Kolkata")


def _list_events(state, args, new_id):
    date = args.get("date")
    include_cancelled = args.get("include_cancelled", False)
    items = [
        dict(e)
        for e in state["events"].values()
        if (date is None or e["start"].startswith(date))
        and (include_cancelled or e["status"] != "cancelled")
    ]
    return state, 200, {"items": items}


def _get_event(state, args, new_id):
    return state, 200, dict(fetch(state, "events", args["event_id"], "event", "event_id"))


def _check_emails(emails, param):
    for email in emails:
        if not validate_format("email", email):
            raise invalid(f"Invalid email address in '{param}': {email!r}", param=param)


def _create_event(state, args, new_id):
    if args["end"] <= args["start"]:
        raise invalid("'end' must be after 'start'.", param="end")
    attendees = list(args.get("attendees") or [])
    _check_emails(attendees, "attendees")
    event = {
        "id": new_id("evt"),
        "title": args["title"],
        "start": args["start"],
        "end": args["end"],
        "attendees": attendees,
        "location": args.get("location"),
        "status": "confirmed",
    }
    return put(state, "events", event), 201, dict(event)


def _active_event(state, event_id, param="event_id"):
    event = fetch(state, "events", event_id, "event", param)
    if event["status"] == "cancelled":
        raise conflict(f"Event '{event_id}' is cancelled.", param=param)
    return event


def _update_event(state, args, new_id):
    event = _active_event(state, args["event_id"])
    changes = {k: args[k] for k in ("title", "start", "end", "location") if args.get(k) is not None}
    if not changes:
        raise invalid("No fields to update.", hint="Provide at least one of title, start, end, location.")
    updated = {**event, **changes}
    if updated["end"] <= updated["start"]:
        raise invalid("'end' must be after 'start'.", param="end")
    return put(state, "events", updated), 200, dict(updated)


def _cancel_event(state, args, new_id):
    event = _active_event(state, args["event_id"])
    updated = {**event, "status": "cancelled"}
    return put(state, "events", updated), 200, dict(updated)


def _add_attendee(state, args, new_id):
    event = _active_event(state, args["event_id"])
    email = args["email"]
    _check_emails([email], "email")
    if email in event["attendees"]:
        raise conflict(f"{email} is already an attendee.", param="email")
    updated = {**event, "attendees": [*event["attendees"], email]}
    return put(state, "events", updated), 200, dict(updated)


def _remove_attendee(state, args, new_id):
    event = _active_event(state, args["event_id"])
    if args["email"] not in event["attendees"]:
        raise not_found("attendee", args["email"], "email")
    updated = {**event, "attendees": [a for a in event["attendees"] if a != args["email"]]}
    return put(state, "events", updated), 200, dict(updated)


def _list_contacts(state, args, new_id):
    return state, 200, {"items": [dict(c) for c in state["contacts"].values()]}


def _create_reminder(state, args, new_id):
    _active_event(state, args["event_id"])
    reminder = {
        "id": new_id("rem"),
        "event_id": args["event_id"],
        "minutes_before": args["minutes_before"],
    }
    return put(state, "reminders", reminder), 201, dict(reminder)


EVENT_ID = Param("event_id", "string", "ID of the event, e.g. 'evt_8f3a2c91d0'. Find it with list_events.",
                 required=True)
DT = "datetime_local"

ENDPOINTS = (
    Endpoint("list_events", "List events, optionally filtered to one day.", (
        Param("date", "string", "Only events starting on this day.", fmt="date"),
        Param("include_cancelled", "boolean", "Include cancelled events (default false)."),
    ), "Object {items: [Event]}.", _list_events),
    Endpoint("get_event", "Get one event by ID.", (EVENT_ID,), EVENT_RETURNS, _get_event),
    Endpoint("create_event", "Create a calendar event.", (
        Param("title", "string", "Event title.", required=True),
        Param("start", "string", "Start time.", required=True, fmt=DT),
        Param("end", "string", "End time.", required=True, fmt=DT),
        Param("attendees", "array", "Attendee email addresses. Look up a contact's email with list_contacts.",
              items={"type": "string"}),
        Param("location", "string", "Location."),
    ), EVENT_RETURNS, _create_event),
    Endpoint("update_event", "Update fields of an event.", (
        EVENT_ID,
        Param("title", "string", "New title."),
        Param("start", "string", "New start time.", fmt=DT),
        Param("end", "string", "New end time.", fmt=DT),
        Param("location", "string", "New location."),
    ), EVENT_RETURNS, _update_event),
    Endpoint("cancel_event", "Cancel an event.", (EVENT_ID,), EVENT_RETURNS, _cancel_event),
    Endpoint("add_attendee", "Add an attendee to an event.", (
        EVENT_ID, Param("email", "string", "Attendee email address (see list_contacts).", required=True, fmt="email"),
    ), EVENT_RETURNS, _add_attendee),
    Endpoint("remove_attendee", "Remove an attendee from an event.", (
        EVENT_ID, Param("email", "string", "Attendee email address (see list_contacts).", required=True, fmt="email"),
    ), EVENT_RETURNS, _remove_attendee),
    Endpoint("list_contacts", "List the user's contacts.", (),
             "Object {items: [Contact {id, name, email}]}.", _list_contacts),
    Endpoint("create_reminder", "Create a reminder for an event.", (
        EVENT_ID,
        Param("minutes_before", "integer", "Minutes before the event start.", required=True,
              minimum=1, maximum=10080),
    ), "Reminder object {id, event_id, minutes_before}.", _create_reminder),
)

NOTIFY = Param("notify_attendees", "boolean", "Whether to email attendees about this change.")

SITES = {
    "rename_param": (
        RenameParam("create_event", "title", "summary"),
        RenameParam("create_event", "attendees", "invitees"),
        RenameParam("create_event", "start", "starts_at"),
        RenameParam("list_events", "date", "day"),
        RenameParam("update_event", "event_id", "id"),
        RenameParam("update_event", "start", "starts_at"),
        RenameParam("cancel_event", "event_id", "id"),
        RenameParam("add_attendee", "email", "attendee_email"),
        RenameParam("add_attendee", "event_id", "id"),
        RenameParam("create_reminder", "minutes_before", "offset_minutes"),
        RenameParam("create_reminder", "event_id", "event"),
    ),
    "format_change": (
        FormatChangeSpec(("start", "end"), "datetime_to_iso8601"),
        FormatChangeSpec(("minutes_before",), "minutes_to_iso8601_duration"),
        FormatChangeSpec(("event_id",), "ids_to_global_ids"),
        FormatChangeSpec(("attendees",), "attendees_to_objects"),
    ),
    "new_required_field": (
        NewRequiredField("create_event", Param("timezone", "string", "IANA timezone for start and end.",
                                               enum=TIMEZONES), "UTC"),
        NewRequiredField("update_event", NOTIFY, True),
        NewRequiredField("cancel_event", NOTIFY, True),
        NewRequiredField("add_attendee", Param("role", "string", "Attendee role.",
                                               enum=("required", "optional")), "required"),
        NewRequiredField("create_reminder", Param("method", "string", "Reminder delivery method.",
                                                  enum=("email", "popup")), "popup"),
        NewRequiredField("list_events", Param("calendar_id", "string",
                                              "Calendar to list. Use 'primary' for the user's calendar.",
                                              enum=("primary",)), "primary"),
    ),
    "deprecation_with_migration": (
        Deprecation("cancel_event", "set_event_status", "Set the status of an event.",
                    fixed=(FixedParam("status", "string", "cancelled", "New status."),)),
        Deprecation("create_reminder", "create_notification", "Schedule a notification before an event.",
                    param_map=(("minutes_before", "offset_minutes"),),
                    fixed=(FixedParam("channel", "string", "popup", "Notification channel."),)),
        Deprecation("add_attendee", "add_event_guest", "Add a guest to an event.",
                    param_map=(("email", "guest_email"),)),
        Deprecation("list_events", "search_events", "Search events.",
                    param_map=(("date", "on_date"),)),
        Deprecation("update_event", "patch_event", "Partially update an event.",
                    param_map=(("event_id", "id"),)),
        Deprecation("create_event", "insert_event", "Insert a calendar event.",
                    param_map=(("title", "summary"),)),
        Deprecation("list_contacts", "list_people", "List people in the user's directory."),
    ),
    "pagination_change": tuple(
        PaginationChange(ep, size) for ep in ("list_events", "list_contacts") for size in (2, 3)
    ),
}

CALENDAR = Domain(
    name="calendar",
    description="Calendar API for managing events, attendees, contacts and reminders. All times UTC.",
    endpoints=ENDPOINTS,
    sites=SITES,
)

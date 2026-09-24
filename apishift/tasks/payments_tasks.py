"""Payments task templates. Instructions name customers, never IDs.

The target customer sits at index >= 3 in list_customers. The target charge or
invoice sits at index >= 3 in that customer's listing.
"""

from __future__ import annotations

import random

from apishift.tasks.model import Call, Expect, pick
from apishift.tasks.pools import (
    CHARGE_DESCS,
    INVOICE_MEMOS,
    TASK_DAYS,
    fmt_money,
    hex_id,
    insert_at,
    money,
    people,
)


def _us_date(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{m}/{d}/{y}"


def _customers(rng):
    custs = [{"id": hex_id(rng, "cus"), "name": nm, "email": em} for nm, em in people(rng, 7)]
    return custs, custs[rng.randint(3, 6)]


def _charge(rng, customer_id, desc, status="succeeded"):
    amount = money(rng)
    refunded = amount if status == "refunded" else 0
    return {"id": hex_id(rng, "ch"), "customer_id": customer_id, "amount_cents": amount,
            "refunded_cents": refunded, "currency": "usd", "description": desc, "status": status}


def _invoice(rng, customer_id, memo, status="open"):
    return {"id": hex_id(rng, "in"), "customer_id": customer_id, "amount_cents": money(rng),
            "due_date": _us_date(rng.choice(TASK_DAYS)), "memo": memo, "status": status}


def _world(rng):
    """Customers, charges and invoices with >= 3 same-customer records before each target."""
    custs, who = _customers(rng)
    others = [c for c in custs if c["id"] != who["id"]]
    descs = rng.sample(CHARGE_DESCS, 6)
    memos = rng.sample(INVOICE_MEMOS, 6)
    n_before = rng.randint(3, 4)

    before_ch = [_charge(rng, who["id"], descs[1 + i], rng.choice(("succeeded", "refunded")))
                 for i in range(n_before)]
    target_ch = _charge(rng, who["id"], descs[0])
    after_ch = [_charge(rng, who["id"], descs[5])] if rng.random() < 0.5 else []
    other_ch = [_charge(rng, rng.choice(others)["id"], rng.choice(CHARGE_DESCS)) for _ in range(4)]
    charges = insert_at(rng, [*before_ch, target_ch, *after_ch], other_ch)

    before_in = [_invoice(rng, who["id"], memos[1 + i], rng.choice(("open", "paid", "void")))
                 for i in range(n_before)]
    target_in = _invoice(rng, who["id"], memos[0])
    other_in = [_invoice(rng, rng.choice(others)["id"], rng.choice(INVOICE_MEMOS)) for _ in range(3)]
    invoices = insert_at(rng, [*before_in, target_in], other_in)

    state = {
        "customers": {c["id"]: c for c in custs},
        "charges": {c["id"]: c for c in charges},
        "refunds": {},
        "invoices": {i["id"]: i for i in invoices},
    }
    return state, who, target_ch, target_in, descs, memos


def _find_customer(name):
    match = lambda c: c["name"] == name  # noqa: E731
    return Call("list_customers", {}, match=match), match


def charge_customer(rng: random.Random):
    state, who, _, _, descs, _ = _world(rng)
    desc = rng.choice([d for d in CHARGE_DESCS if d not in descs])
    amount = money(rng, 5, 400)
    instruction = f"Charge {who['name']} {fmt_money(amount)} for '{desc}'."

    def plan():
        call, match = _find_customer(who["name"])
        cust = pick((yield call), match)
        yield Call("create_charge", {"customer_id": cust["id"], "amount": amount / 100, "description": desc})

    expects = (Expect("charges", "create", {"customer_id": who["id"], "amount_cents": amount,
                                            "refunded_cents": 0, "description": desc,
                                            "status": "succeeded"}),)
    return instruction, state, expects, plan


def _charge_lookup(who, desc):
    match = lambda c: c["description"] == desc and c["status"] == "succeeded"  # noqa: E731
    return lambda cust_id: Call("list_charges", {"customer_id": cust_id}, match=match), match


def refund_full(rng: random.Random):
    state, who, target, _, _, _ = _world(rng)
    desc = target["description"]
    instruction = (f"Refund {who['name']}'s {fmt_money(target['amount_cents'])} charge for "
                   f"'{desc}' in full.")

    def plan():
        call, cmatch = _find_customer(who["name"])
        cust = pick((yield call), cmatch)
        make, match = _charge_lookup(who, desc)
        ch = pick((yield make(cust["id"])), match)
        yield Call("refund_charge", {"charge_id": ch["id"]})

    expects = (
        Expect("charges", "update", {"status": "refunded", "refunded_cents": target["amount_cents"]},
               record_id=target["id"]),
        Expect("refunds", "create", {"charge_id": target["id"], "amount_cents": target["amount_cents"]}),
    )
    return instruction, state, expects, plan


def refund_partial(rng: random.Random):
    state, who, target, _, _, _ = _world(rng)
    desc = target["description"]
    part = rng.randint(1, target["amount_cents"] // 100 - 1) * 100 + rng.choice((0, 50))
    instruction = f"Refund {fmt_money(part)} of {who['name']}'s '{desc}' charge."

    def plan():
        call, cmatch = _find_customer(who["name"])
        cust = pick((yield call), cmatch)
        make, match = _charge_lookup(who, desc)
        ch = pick((yield make(cust["id"])), match)
        yield Call("refund_charge", {"charge_id": ch["id"], "amount": part / 100})

    expects = (
        Expect("charges", "update", {"status": "partially_refunded", "refunded_cents": part},
               record_id=target["id"]),
        Expect("refunds", "create", {"charge_id": target["id"], "amount_cents": part}),
    )
    return instruction, state, expects, plan


def create_invoice(rng: random.Random):
    state, who, _, _, _, memos = _world(rng)
    memo = rng.choice([m for m in INVOICE_MEMOS if m not in memos])
    amount = money(rng, 100, 5000)
    due = rng.choice(TASK_DAYS)
    instruction = f"Invoice {who['name']} {fmt_money(amount)} for '{memo}', due {due}."

    def plan():
        call, match = _find_customer(who["name"])
        cust = pick((yield call), match)
        yield Call("create_invoice", {"customer_id": cust["id"], "amount": amount / 100,
                                      "due_date": _us_date(due), "memo": memo})

    expects = (Expect("invoices", "create", {"customer_id": who["id"], "amount_cents": amount,
                                             "due_date": _us_date(due), "memo": memo,
                                             "status": "open"}),)
    return instruction, state, expects, plan


def void_invoice(rng: random.Random):
    state, who, _, target, _, _ = _world(rng)
    memo = target["memo"]
    instruction = f"Void {who['name']}'s open invoice for '{memo}'."

    def plan():
        call, cmatch = _find_customer(who["name"])
        cust = pick((yield call), cmatch)
        match = lambda i: i["memo"] == memo and i["status"] == "open"  # noqa: E731
        inv = pick((yield Call("list_invoices", {"customer_id": cust["id"]}, match=match)), match)
        yield Call("void_invoice", {"invoice_id": inv["id"]})

    expects = (Expect("invoices", "update", {"status": "void"}, record_id=target["id"]),)
    return instruction, state, expects, plan


TEMPLATES = {
    "charge_customer": charge_customer,
    "refund_full": refund_full,
    "refund_partial": refund_partial,
    "create_invoice": create_invoice,
    "void_invoice": void_invoice,
}

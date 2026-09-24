"""Payments mock API: customers, charges, refunds, invoices.

Amounts are stored as integer cents in state. The canonical API speaks dollars.
"""

from __future__ import annotations

from apishift.envs.domain import Domain, fetch, put
from apishift.envs.errors import conflict, invalid
from apishift.envs.mutations import (
    Deprecation,
    FixedParam,
    FormatChangeSpec,
    NewRequiredField,
    PaginationChange,
    RenameParam,
)
from apishift.envs.schema import Endpoint, Param

MIN_CENTS = 50
CURRENCIES = ("usd", "eur", "gbp")


def _dollars(cents: int) -> float:
    return cents / 100


def _cents(dollars: float) -> int:
    return round(dollars * 100)


def _charge_view(c):
    return {
        "id": c["id"],
        "customer_id": c["customer_id"],
        "amount": _dollars(c["amount_cents"]),
        "amount_refunded": _dollars(c["refunded_cents"]),
        "currency": c["currency"],
        "description": c["description"],
        "status": c["status"],
    }


def _invoice_view(i):
    return {
        "id": i["id"],
        "customer_id": i["customer_id"],
        "amount": _dollars(i["amount_cents"]),
        "due_date": i["due_date"],
        "memo": i["memo"],
        "status": i["status"],
    }


def _refund_view(r):
    return {"id": r["id"], "charge_id": r["charge_id"], "amount": _dollars(r["amount_cents"]),
            "reason": r["reason"], "status": r["status"]}


def _customer(state, customer_id):
    return fetch(state, "customers", customer_id, "customer", "customer_id")


def _amount_cents(args, param="amount"):
    cents = _cents(args[param])
    if cents < MIN_CENTS:
        raise invalid("Amount must be at least $0.50.", param=param)
    return cents


def _list_customers(state, args, new_id):
    email = args.get("email")
    items = [dict(c) for c in state["customers"].values() if email is None or c["email"] == email]
    return state, 200, {"items": items}


def _get_customer(state, args, new_id):
    return state, 200, dict(_customer(state, args["customer_id"]))


def _create_customer(state, args, new_id):
    if any(c["email"] == args["email"] for c in state["customers"].values()):
        raise conflict(f"A customer with email {args['email']} already exists.", param="email")
    customer = {"id": new_id("cus"), "name": args["name"], "email": args["email"]}
    return put(state, "customers", customer), 201, dict(customer)


def _create_charge(state, args, new_id):
    _customer(state, args["customer_id"])
    charge = {
        "id": new_id("ch"),
        "customer_id": args["customer_id"],
        "amount_cents": _amount_cents(args),
        "refunded_cents": 0,
        "currency": "usd",
        "description": args.get("description", ""),
        "status": "succeeded",
    }
    return put(state, "charges", charge), 201, _charge_view(charge)


def _list_charges(state, args, new_id):
    _customer(state, args["customer_id"])
    items = [_charge_view(c) for c in state["charges"].values() if c["customer_id"] == args["customer_id"]]
    return state, 200, {"items": items}


def _refund_charge(state, args, new_id):
    charge = fetch(state, "charges", args["charge_id"], "charge", "charge_id")
    remaining = charge["amount_cents"] - charge["refunded_cents"]
    if remaining <= 0:
        raise conflict(f"Charge '{charge['id']}' has already been fully refunded.", param="charge_id")
    cents = _cents(args["amount"]) if args.get("amount") is not None else remaining
    if cents <= 0:
        raise invalid("Refund amount must be positive.", param="amount")
    if cents > remaining:
        raise invalid(f"Refund amount exceeds the remaining ${_dollars(remaining):.2f}.", param="amount")
    refunded = charge["refunded_cents"] + cents
    status = "refunded" if refunded == charge["amount_cents"] else "partially_refunded"
    updated = {**charge, "refunded_cents": refunded, "status": status}
    refund = {"id": new_id("re"), "charge_id": charge["id"], "amount_cents": cents,
              "reason": args.get("reason"), "status": "succeeded"}
    new_state = put(put(state, "charges", updated), "refunds", refund)
    return new_state, 201, _refund_view(refund)


def _create_invoice(state, args, new_id):
    _customer(state, args["customer_id"])
    invoice = {
        "id": new_id("in"),
        "customer_id": args["customer_id"],
        "amount_cents": _amount_cents(args),
        "due_date": args["due_date"],
        "memo": args.get("memo", ""),
        "status": "open",
    }
    return put(state, "invoices", invoice), 201, _invoice_view(invoice)


def _list_invoices(state, args, new_id):
    _customer(state, args["customer_id"])
    status = args.get("status")
    items = [
        _invoice_view(i)
        for i in state["invoices"].values()
        if i["customer_id"] == args["customer_id"] and (status is None or i["status"] == status)
    ]
    return state, 200, {"items": items}


def _void_invoice(state, args, new_id):
    invoice = fetch(state, "invoices", args["invoice_id"], "invoice", "invoice_id")
    if invoice["status"] != "open":
        raise conflict(f"Only open invoices can be voided (status: {invoice['status']}).",
                       param="invoice_id")
    updated = {**invoice, "status": "void"}
    return put(state, "invoices", updated), 200, _invoice_view(updated)


CUSTOMER_ID = Param("customer_id", "string", "ID of the customer, e.g. 'cus_4d2a9e71b3'. Find it with list_customers.",
                    required=True)
AMOUNT = Param("amount", "number", "Amount.", required=True, fmt="money_dollars")
CHARGE_RETURNS = "Charge object {id, customer_id, amount, amount_refunded, currency, description, status}."
INVOICE_RETURNS = "Invoice object {id, customer_id, amount, due_date, memo, status}."

ENDPOINTS = (
    Endpoint("list_customers", "List customers, optionally filtered by exact email.", (
        Param("email", "string", "Exact email to match.", fmt="email"),
    ), "Object {items: [Customer {id, name, email}]}.", _list_customers),
    Endpoint("get_customer", "Get one customer.", (CUSTOMER_ID,), "Customer object.", _get_customer),
    Endpoint("create_customer", "Create a customer.", (
        Param("name", "string", "Full name.", required=True),
        Param("email", "string", "Email.", required=True, fmt="email"),
    ), "Customer object.", _create_customer),
    Endpoint("create_charge", "Charge a customer's card on file.", (
        CUSTOMER_ID, AMOUNT, Param("description", "string", "Description shown on the receipt."),
    ), CHARGE_RETURNS, _create_charge),
    Endpoint("list_charges", "List a customer's charges.", (CUSTOMER_ID,),
             "Object {items: [Charge]}.", _list_charges),
    Endpoint("refund_charge", "Refund a charge, fully or partially.", (
        Param("charge_id", "string", "ID of the charge, e.g. 'ch_7e3b1a90c4'. Find it with list_charges.",
              required=True),
        Param("amount", "number", "Amount to refund; defaults to the full remaining amount.",
              fmt="money_dollars"),
        Param("reason", "string", "Refund reason.",
              enum=("requested_by_customer", "duplicate", "fraudulent")),
    ), "Refund object {id, charge_id, amount, reason, status}.", _refund_charge),
    Endpoint("create_invoice", "Create an invoice for a customer.", (
        CUSTOMER_ID, AMOUNT,
        Param("due_date", "string", "Due date.", required=True, fmt="date_us"),
        Param("memo", "string", "Memo shown on the invoice."),
    ), INVOICE_RETURNS, _create_invoice),
    Endpoint("list_invoices", "List a customer's invoices.", (
        CUSTOMER_ID, Param("status", "string", "Filter by status.", enum=("open", "paid", "void")),
    ), "Object {items: [Invoice]}.", _list_invoices),
    Endpoint("void_invoice", "Void an open invoice.", (
        Param("invoice_id", "string", "ID of the invoice, e.g. 'in_2c8f5d1a6e'. Find it with list_invoices.",
              required=True),
    ), INVOICE_RETURNS, _void_invoice),
)

CURRENCY = Param("currency", "string", "Three-letter ISO currency code, lowercase.", enum=CURRENCIES)

SITES = {
    "rename_param": (
        RenameParam("create_charge", "customer_id", "customer"),
        RenameParam("create_charge", "description", "statement_descriptor"),
        RenameParam("list_charges", "customer_id", "customer"),
        RenameParam("refund_charge", "charge_id", "charge"),
        RenameParam("refund_charge", "amount", "refund_amount"),
        RenameParam("create_invoice", "customer_id", "customer"),
        RenameParam("create_invoice", "due_date", "due_on"),
        RenameParam("create_invoice", "memo", "description"),
        RenameParam("list_invoices", "customer_id", "customer"),
        RenameParam("void_invoice", "invoice_id", "id"),
    ),
    "format_change": (
        FormatChangeSpec(("amount", "amount_refunded"), "dollars_to_cents"),
        FormatChangeSpec(("due_date",), "us_date_to_iso8601"),
        FormatChangeSpec(("customer_id", "charge_id", "invoice_id"), "ids_to_global_ids"),
    ),
    "new_required_field": (
        NewRequiredField("create_charge", CURRENCY, "usd"),
        NewRequiredField("create_invoice", CURRENCY, "usd"),
        NewRequiredField("refund_charge", Param("idempotency_key", "string",
                                                "Unique key that makes the request safe to retry."),
                         "refund-0001"),
        NewRequiredField("void_invoice", Param("void_reason", "string", "Why the invoice is voided.",
                                               enum=("duplicate", "customer_request", "other")),
                         "customer_request"),
        NewRequiredField("list_customers", Param("account_id", "string",
                                                 "Account to scope the request to. Use 'acct_main'.",
                                                 enum=("acct_main",)), "acct_main"),
        NewRequiredField("list_charges", Param("account_id", "string",
                                               "Account to scope the request to. Use 'acct_main'.",
                                               enum=("acct_main",)), "acct_main"),
    ),
    "deprecation_with_migration": (
        Deprecation("create_charge", "create_payment_intent", "Create and confirm a payment intent.",
                    param_map=(("customer_id", "customer"),),
                    fixed=(FixedParam("confirm", "boolean", True, "Confirm immediately."),)),
        Deprecation("refund_charge", "create_refund", "Create a refund for a charge.",
                    param_map=(("charge_id", "charge"),)),
        Deprecation("void_invoice", "update_invoice_status", "Change the status of an invoice.",
                    fixed=(FixedParam("status", "string", "void", "New status."),)),
        Deprecation("list_charges", "search_charges", "Search a customer's charges.",
                    param_map=(("customer_id", "customer"),)),
        Deprecation("create_invoice", "issue_invoice", "Issue an invoice to a customer.",
                    param_map=(("due_date", "due_on"),)),
        Deprecation("list_customers", "search_customers", "Search customers."),
    ),
    "pagination_change": tuple(
        PaginationChange(ep, size)
        for ep in ("list_customers", "list_charges", "list_invoices")
        for size in (2, 3)
    ),
}

PAYMENTS = Domain(
    name="payments",
    description="Payments API for customers, card charges, refunds and invoices. Currency: USD.",
    endpoints=ENDPOINTS,
    sites=SITES,
)

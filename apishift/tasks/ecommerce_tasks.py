"""E-commerce task templates.

The target order sits at index >= 3 in the customer's list_orders result.
Its key product appears in no other order of that customer.
"""

from __future__ import annotations

import random

from apishift.tasks.model import Call, Expect, pick
from apishift.tasks.pools import COUPONS, PRODUCTS, address, hex_id, insert_at, people

RETURN_REASONS = {"damaged": "it arrived damaged", "wrong_item": "the wrong item was sent",
                  "no_longer_needed": "it is no longer needed"}


def _products(rng):
    chosen = rng.sample(PRODUCTS, 10)
    return [{"id": hex_id(rng, "prod"), "name": n, "category": c,
             "price_cents": rng.randint(8, 250) * 100 + rng.choice((0, 99, 50)),
             "stock": rng.randint(8, 40)} for n, c in chosen]


def _order(rng, email, lines, status):
    return {"id": hex_id(rng, "ord"), "customer_email": email,
            "items": [{"product_id": p["id"], "quantity": q} for p, q in lines],
            "status": status, "shipping_address": address(rng), "coupon": None, "cancel_reason": None}


def _world(rng, target_status):
    products = _products(rng)
    (_, email), *others = people(rng, 4)
    key, rest = products[0], products[1:]
    n_before = rng.randint(3, 4)
    statuses = ("pending", "shipped", "delivered", "cancelled")

    def distractor(owner):
        lines = [(p, rng.randint(1, 3)) for p in rng.sample(rest, rng.randint(1, 2))]
        return _order(rng, owner, lines, rng.choice(statuses))

    qty = rng.randint(2, 4)
    extra = [(rng.choice(rest), 1)] if rng.random() < 0.4 else []
    target = _order(rng, email, [(key, qty), *extra], target_status)
    mine = [distractor(email) for _ in range(n_before)] + [target]
    mine += [distractor(email)] if rng.random() < 0.5 else []
    theirs = [distractor(o[1]) for o in others for _ in range(rng.randint(1, 2))]
    orders = insert_at(rng, mine, theirs)
    state = {
        "products": {p["id"]: p for p in products},
        "orders": {o["id"]: o for o in orders},
        "coupons": {c: {"id": c, "code": c, "percent_off": pct} for c, pct in COUPONS},
        "returns": {},
    }
    return state, email, key, target, qty


def _find_order(email, pname, status):
    def match(o):
        return o["status"] == status and any(it["name"] == pname for it in o["items"])

    return Call("list_orders", {"customer_email": email}, match=match), match


def place_order(rng: random.Random):
    state, _, _, _, _ = _world(rng, "delivered")
    (_, email), = people(rng, 1)
    product = rng.choice(list(state["products"].values()))
    qty = rng.randint(1, 3)
    ship = address(rng)
    instruction = f"Order {qty} x '{product['name']}' for {email}, shipping to {ship}."
    pname = product["name"]

    def plan():
        match = lambda p: p["name"] == pname  # noqa: E731
        prod = pick((yield Call("search_products", {"query": pname}, match=match)), match)
        yield Call("create_order", {"customer_email": email,
                                    "items": [{"product_id": prod["id"], "quantity": qty}],
                                    "shipping_address": ship})

    expects = (
        Expect("orders", "create", {"customer_email": email,
                                    "items": [{"product_id": product["id"], "quantity": qty}],
                                    "shipping_address": ship, "status": "pending"}),
        Expect("products", "update", {"stock": product["stock"] - qty}, record_id=product["id"]),
    )
    return instruction, state, expects, plan


def cancel_order(rng: random.Random):
    state, email, key, target, _ = _world(rng, "pending")
    instruction = f"Cancel {email}'s pending order of '{key['name']}'."

    def plan():
        call, match = _find_order(email, key["name"], "pending")
        order = pick((yield call), match)
        yield Call("cancel_order", {"order_id": order["id"]})

    restock = tuple(
        Expect("products", "update",
               {"stock": state["products"][it["product_id"]]["stock"] + it["quantity"]},
               record_id=it["product_id"])
        for it in target["items"]
    )
    expects = (
        Expect("orders", "update", {"status": "cancelled"}, record_id=target["id"],
               free_fields=("cancel_reason",)),
        *restock,
    )
    return instruction, state, expects, plan


def update_address(rng: random.Random):
    state, email, key, target, _ = _world(rng, "pending")
    ship = address(rng)
    while ship == target["shipping_address"]:
        ship = address(rng)
    instruction = f"Change the shipping address of {email}'s pending '{key['name']}' order to {ship}."

    def plan():
        call, match = _find_order(email, key["name"], "pending")
        order = pick((yield call), match)
        yield Call("update_shipping_address", {"order_id": order["id"], "shipping_address": ship})

    expects = (Expect("orders", "update", {"shipping_address": ship}, record_id=target["id"]),)
    return instruction, state, expects, plan


def apply_coupon(rng: random.Random):
    state, email, key, target, _ = _world(rng, "pending")
    code = rng.choice(COUPONS)[0]
    instruction = f"Apply coupon {code} to {email}'s pending '{key['name']}' order."

    def plan():
        call, match = _find_order(email, key["name"], "pending")
        order = pick((yield call), match)
        yield Call("apply_coupon", {"order_id": order["id"], "code": code})

    expects = (Expect("orders", "update", {"coupon": code}, record_id=target["id"]),)
    return instruction, state, expects, plan


def create_return(rng: random.Random):
    state, email, key, target, qty = _world(rng, "delivered")
    n = rng.randint(1, qty)
    reason = rng.choice(sorted(RETURN_REASONS))
    instruction = (f"Start a return of {n} x '{key['name']}' from {email}'s delivered order "
                   f"because {RETURN_REASONS[reason]}.")

    def plan():
        call, match = _find_order(email, key["name"], "delivered")
        order = pick((yield call), match)
        line = next(it for it in order["items"] if it["name"] == key["name"])
        yield Call("create_return", {"order_id": order["id"], "product_id": line["product_id"],
                                     "quantity": n, "reason": reason})

    expects = (Expect("returns", "create", {"order_id": target["id"], "product_id": key["id"],
                                            "quantity": n, "reason": reason,
                                            "status": "requested"}),)
    return instruction, state, expects, plan


TEMPLATES = {
    "place_order": place_order,
    "cancel_order": cancel_order,
    "update_address": update_address,
    "apply_coupon": apply_coupon,
    "create_return": create_return,
}

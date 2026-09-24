"""E-commerce mock API: products, orders, coupons, returns."""

from __future__ import annotations

from apishift.envs.domain import Domain, fetch, put
from apishift.envs.errors import conflict, invalid, not_found
from apishift.envs.mutations import (
    Deprecation,
    FixedParam,
    FormatChangeSpec,
    NewRequiredField,
    PaginationChange,
    RenameParam,
)
from apishift.envs.schema import Endpoint, Param

RETURN_REASONS = ("damaged", "wrong_item", "no_longer_needed")
ORDER_STATUSES = ("pending", "shipped", "delivered", "cancelled")


def _product_view(p):
    return {"id": p["id"], "name": p["name"], "category": p["category"],
            "price": p["price_cents"] / 100, "stock": p["stock"]}


def _order_view(state, o):
    return {
        "id": o["id"],
        "customer_email": o["customer_email"],
        "items": [
            {"product_id": it["product_id"], "name": state["products"][it["product_id"]]["name"],
             "quantity": it["quantity"]}
            for it in o["items"]
        ],
        "status": o["status"],
        "shipping_address": o["shipping_address"],
        "coupon": o["coupon"],
    }


def _order(state, order_id):
    return fetch(state, "orders", order_id, "order", "order_id")


def _pending(state, order_id, action):
    order = _order(state, order_id)
    if order["status"] != "pending":
        raise conflict(f"Only pending orders can be {action} (status: {order['status']}).",
                       param="order_id")
    return order


def _adjust_stock(state, items, sign):
    for it in items:
        p = state["products"][it["product_id"]]
        state = put(state, "products", {**p, "stock": p["stock"] + sign * it["quantity"]})
    return state


def _search_products(state, args, new_id):
    q = args["query"].lower()
    category = args.get("category")
    items = [
        _product_view(p)
        for p in state["products"].values()
        if q in p["name"].lower() and (category is None or p["category"] == category)
    ]
    return state, 200, {"items": items}


def _get_product(state, args, new_id):
    return state, 200, _product_view(fetch(state, "products", args["product_id"], "product", "product_id"))


def _list_orders(state, args, new_id):
    status = args.get("status")
    items = [
        _order_view(state, o)
        for o in state["orders"].values()
        if o["customer_email"] == args["customer_email"] and (status is None or o["status"] == status)
    ]
    return state, 200, {"items": items}


def _get_order(state, args, new_id):
    return state, 200, _order_view(state, _order(state, args["order_id"]))


def _create_order(state, args, new_id):
    items = args["items"]
    if not items:
        raise invalid("'items' must not be empty.", param="items")
    for it in items:
        p = state["products"].get(it["product_id"])
        if p is None:
            raise not_found("product", it["product_id"], "items")
        if it["quantity"] < 1:
            raise invalid("Item quantity must be at least 1.", param="items")
        if p["stock"] < it["quantity"]:
            raise conflict(f"Insufficient stock for '{p['name']}' ({p['stock']} left).", param="items")
    order = {
        "id": new_id("ord"),
        "customer_email": args["customer_email"],
        "items": [{"product_id": it["product_id"], "quantity": it["quantity"]} for it in items],
        "status": "pending",
        "shipping_address": args["shipping_address"],
        "coupon": None,
        "cancel_reason": None,
    }
    new_state = _adjust_stock(put(state, "orders", order), order["items"], -1)
    return new_state, 201, _order_view(new_state, order)


def _cancel_order(state, args, new_id):
    order = _pending(state, args["order_id"], "cancelled")
    updated = {**order, "status": "cancelled", "cancel_reason": args.get("reason")}
    new_state = _adjust_stock(put(state, "orders", updated), order["items"], +1)
    return new_state, 200, _order_view(new_state, updated)


def _update_shipping_address(state, args, new_id):
    order = _pending(state, args["order_id"], "updated")
    updated = {**order, "shipping_address": args["shipping_address"]}
    return put(state, "orders", updated), 200, _order_view(state, updated)


def _apply_coupon(state, args, new_id):
    order = _pending(state, args["order_id"], "discounted")
    code = args["code"]
    if code not in state["coupons"]:
        raise not_found("coupon", code, "code")
    if order["coupon"] is not None:
        raise conflict(f"Order already has coupon {order['coupon']}.", param="code")
    updated = {**order, "coupon": code}
    return put(state, "orders", updated), 200, _order_view(state, updated)


def _create_return(state, args, new_id):
    order = _order(state, args["order_id"])
    if order["status"] != "delivered":
        raise conflict(f"Only delivered orders can be returned (status: {order['status']}).",
                       param="order_id")
    line = next((it for it in order["items"] if it["product_id"] == args["product_id"]), None)
    if line is None:
        raise not_found("order item", args["product_id"], "product_id")
    already = sum(r["quantity"] for r in state["returns"].values()
                  if r["order_id"] == order["id"] and r["product_id"] == args["product_id"])
    if args["quantity"] > line["quantity"] - already:
        raise invalid(f"Cannot return more than {line['quantity'] - already} unit(s).", param="quantity")
    ret = {"id": new_id("ret"), "order_id": order["id"], "product_id": args["product_id"],
           "quantity": args["quantity"], "reason": args["reason"], "status": "requested"}
    return put(state, "returns", ret), 201, dict(ret)


def _restock_product(state, args, new_id):
    p = fetch(state, "products", args["product_id"], "product", "product_id")
    updated = {**p, "stock": p["stock"] + args["quantity"]}
    return put(state, "products", updated), 200, _product_view(updated)


ORDER_ID = Param("order_id", "string", "ID of the order, e.g. 'ord_5b1e07c4a2'. Find it with list_orders.",
                 required=True)
PRODUCT_ID = Param("product_id", "string", "ID of the product, e.g. 'prod_91c2d4e6f8'. Find it with search_products.",
                   required=True)
ORDER_RETURNS = "Order object {id, customer_email, items: [{product_id, name, quantity}], status, " \
                "shipping_address, coupon}."
ADDRESS = Param("shipping_address", "string", "Shipping address.", required=True, fmt="address_string")
LINE_ITEMS = {
    "type": "object",
    "properties": {"product_id": {"type": "string"}, "quantity": {"type": "integer"}},
    "required": ["product_id", "quantity"],
}

ENDPOINTS = (
    Endpoint("search_products", "Search products by name.", (
        Param("query", "string", "Case-insensitive substring of the product name.", required=True),
        Param("category", "string", "Filter by category."),
    ), "Object {items: [Product {id, name, category, price, stock}]}.", _search_products),
    Endpoint("get_product", "Get one product.", (PRODUCT_ID,),
             "Product object.", _get_product),
    Endpoint("list_orders", "List a customer's orders.", (
        Param("customer_email", "string", "Customer email.", required=True, fmt="email"),
        Param("status", "string", "Filter by status.", enum=ORDER_STATUSES),
    ), "Object {items: [Order]}.", _list_orders),
    Endpoint("get_order", "Get one order.", (ORDER_ID,), ORDER_RETURNS, _get_order),
    Endpoint("create_order", "Place an order.", (
        Param("customer_email", "string", "Customer email.", required=True, fmt="email"),
        Param("items", "array", "Line items. Find product IDs with search_products.", required=True,
              items=LINE_ITEMS),
        ADDRESS,
    ), ORDER_RETURNS, _create_order),
    Endpoint("cancel_order", "Cancel a pending order and restock its items.", (
        ORDER_ID, Param("reason", "string", "Cancellation reason."),
    ), ORDER_RETURNS, _cancel_order),
    Endpoint("update_shipping_address", "Change the shipping address of a pending order.",
             (ORDER_ID, ADDRESS), ORDER_RETURNS, _update_shipping_address),
    Endpoint("apply_coupon", "Apply a coupon code to a pending order.", (
        ORDER_ID, Param("code", "string", "Coupon code.", required=True),
    ), ORDER_RETURNS, _apply_coupon),
    Endpoint("create_return", "Start a return for items of a delivered order.", (
        ORDER_ID,
        Param("product_id", "string", "ID of the product to return (see the order's items).", required=True),
        Param("quantity", "integer", "Units to return.", required=True, minimum=1),
        Param("reason", "string", "Return reason.", required=True, enum=RETURN_REASONS),
    ), "Return object {id, order_id, product_id, quantity, reason, status}.", _create_return),
    Endpoint("restock_product", "Add units to a product's stock.", (
        PRODUCT_ID,
        Param("quantity", "integer", "Units to add.", required=True, minimum=1),
    ), "Product object.", _restock_product),
)

NOTIFY = Param("notify_customer", "boolean", "Whether to email the customer about this change.")

SITES = {
    "rename_param": (
        RenameParam("search_products", "query", "q"),
        RenameParam("list_orders", "customer_email", "email"),
        RenameParam("create_order", "customer_email", "email"),
        RenameParam("create_order", "items", "line_items"),
        RenameParam("create_order", "shipping_address", "ship_to"),
        RenameParam("cancel_order", "order_id", "id"),
        RenameParam("update_shipping_address", "shipping_address", "address"),
        RenameParam("update_shipping_address", "order_id", "id"),
        RenameParam("apply_coupon", "code", "coupon_code"),
        RenameParam("apply_coupon", "order_id", "order"),
        RenameParam("create_return", "quantity", "qty"),
        RenameParam("create_return", "order_id", "order"),
    ),
    "format_change": (
        FormatChangeSpec(("shipping_address",), "address_string_to_object"),
        FormatChangeSpec(("order_id", "product_id"), "ids_to_global_ids"),
        FormatChangeSpec(("reason",), "enum_to_upper"),
    ),
    "new_required_field": (
        NewRequiredField("create_order", Param("payment_method", "string", "How the order is paid.",
                                               enum=("card_on_file", "store_credit")), "card_on_file"),
        NewRequiredField("cancel_order", NOTIFY, True),
        NewRequiredField("update_shipping_address", NOTIFY, True),
        NewRequiredField("create_return", Param("refund_method", "string", "Where the refund goes.",
                                                enum=("original_payment", "store_credit")),
                         "original_payment"),
        NewRequiredField("apply_coupon", Param("channel", "string", "Sales channel for redemption.",
                                               enum=("web", "pos")), "web"),
        NewRequiredField("list_orders", Param("store_id", "string", "Store to query. Use 'main'.",
                                              enum=("main",)), "main"),
    ),
    "deprecation_with_migration": (
        Deprecation("cancel_order", "update_order_status", "Change the status of an order.",
                    param_map=(("reason", "note"),),
                    fixed=(FixedParam("status", "string", "cancelled", "New status."),)),
        Deprecation("update_shipping_address", "patch_order", "Partially update a pending order.",
                    param_map=(("shipping_address", "ship_to"),)),
        Deprecation("apply_coupon", "add_discount", "Add a discount code to a pending order.",
                    param_map=(("code", "discount_code"),)),
        Deprecation("create_return", "create_rma", "Open a return merchandise authorization.",
                    param_map=(("product_id", "item_id"),)),
        Deprecation("search_products", "catalog_search", "Search the product catalog.",
                    param_map=(("query", "q"),)),
        Deprecation("list_orders", "search_orders", "Search a customer's orders.",
                    param_map=(("customer_email", "email"),)),
        Deprecation("create_order", "checkout", "Check out a new order.",
                    param_map=(("items", "line_items"),)),
    ),
    "pagination_change": tuple(PaginationChange("list_orders", size) for size in (2, 3)),
}

ECOMMERCE = Domain(
    name="ecommerce",
    description="E-commerce API for products, orders, coupons and returns.",
    endpoints=ENDPOINTS,
    sites=SITES,
)

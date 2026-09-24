"""Value pools and seeded helpers for task generation."""

from __future__ import annotations

import random

FIRST = ("Ana", "Ben", "Chen", "Divya", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jamal",
         "Kira", "Luis", "Maya", "Noah", "Omar", "Priya", "Quinn", "Rosa", "Sam", "Tara")
LAST = ("Alvarez", "Brooks", "Castro", "Dubois", "Eriksen", "Fischer", "Gupta", "Haddad",
        "Ito", "Jensen", "Kowalski", "Larsen", "Mensah", "Novak", "Okafor", "Patel")
DOMAINS = ("acme.io", "globex.com", "initech.dev", "umbrella.org", "hooli.com")

MEETINGS = ("Design Review", "Weekly Sync", "Budget Planning", "Sprint Retro", "Hiring Panel",
            "Roadmap Review", "Customer Call", "Security Audit", "Launch Prep", "Offsite Planning",
            "Vendor Demo", "Board Prep", "Onboarding", "Incident Review", "Quarterly Kickoff")

CHARGE_DESCS = ("Annual plan", "Monthly subscription", "Setup fee", "Consulting hours",
                "Hardware kit", "Support add-on", "Training session", "Extra seats",
                "Data export", "Priority onboarding", "Storage upgrade", "API overage")
INVOICE_MEMOS = ("Q3 retainer", "October services", "Workshop", "Design sprint", "Audit",
                 "Migration project", "License renewal", "Custom integration", "Phase 2 milestone")

PRODUCTS = (
    ("Walnut Desk Lamp", "home"), ("Ceramic Pour-Over Set", "kitchen"), ("Merino Travel Scarf", "apparel"),
    ("Trail Running Vest", "sports"), ("Noise Cancelling Headphones", "electronics"),
    ("Cast Iron Skillet", "kitchen"), ("Bamboo Cutting Board", "kitchen"), ("Wool Throw Blanket", "home"),
    ("Mechanical Keyboard", "electronics"), ("Yoga Mat", "sports"), ("Linen Shirt", "apparel"),
    ("Smart Plug", "electronics"), ("Glass Water Bottle", "sports"), ("Leather Notebook", "office"),
    ("Standing Desk Mat", "office"), ("Espresso Grinder", "kitchen"),
)
COUPONS = (("WELCOME10", 10), ("FALL15", 15), ("VIP20", 20), ("SHIPFREE", 5))
STREETS = ("Oak St", "Maple Ave", "Cedar Ln", "Pine Rd", "Elm Ct", "Birch Way", "Harbor Blvd")
CITIES = (("Springfield", "62704"), ("Riverton", "84065"), ("Fairview", "37062"),
          ("Lakewood", "80226"), ("Georgetown", "78626"), ("Salem", "97301"))

TASK_DAYS = tuple(f"2026-10-{d:02d}" for d in range(5, 31))


def hex_id(rng: random.Random, prefix: str) -> str:
    return f"{prefix}_{rng.getrandbits(40):010x}"


def people(rng: random.Random, n: int) -> list[tuple[str, str]]:
    """n distinct (name, email) pairs."""
    firsts = rng.sample(FIRST, n)
    lasts = rng.sample(LAST, n)
    out = []
    for f, last in zip(firsts, lasts, strict=True):
        out.append((f"{f} {last}", f"{f.lower()}.{last.lower()}@{rng.choice(DOMAINS)}"))
    return out


def address(rng: random.Random) -> str:
    city, postal = rng.choice(CITIES)
    return f"{rng.randint(10, 999)} {rng.choice(STREETS)}, {city}, {postal}"


def hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def money(rng: random.Random, lo: int = 20, hi: int = 900) -> int:
    """Random amount in cents with a realistic cents part."""
    return rng.randint(lo, hi) * 100 + rng.choice((0, 0, 50, 99, 25, 75))


def fmt_money(cents: int) -> str:
    return f"${cents // 100:,}.{cents % 100:02d}"


def insert_at(rng: random.Random, seq: list, items: list) -> list:
    """Insert `items` at random positions of `seq`, keeping seq's order. Never mutates."""
    out = list(seq)
    for it in items:
        out.insert(rng.randint(0, len(out)), it)
    return out

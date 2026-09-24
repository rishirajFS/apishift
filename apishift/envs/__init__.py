"""Mock API domains and the live (mutated) environment."""

from apishift.envs.calendar import CALENDAR
from apishift.envs.domain import Domain
from apishift.envs.ecommerce import ECOMMERCE
from apishift.envs.live import LiveEnv
from apishift.envs.payments import PAYMENTS

DOMAINS: dict[str, Domain] = {d.name: d for d in (CALENDAR, PAYMENTS, ECOMMERCE)}

__all__ = ["DOMAINS", "Domain", "LiveEnv"]

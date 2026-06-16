"""Custom exceptions for the ride-pooling simulation environment.

The design philosophy mandates *strict* action validation: any business-rule
violation raises an error immediately rather than being silently handled. This
guarantees reproducibility and forces upstream policies to coordinate.
"""


class EnvironmentError(Exception):
    """Base class for all simulation environment errors."""


class InvalidActionError(EnvironmentError):
    """Raised when a single driver's action violates a business rule.

    Examples
    --------
    * An action contains both an order-bidding set and a relocation target
      (the two are strictly mutually exclusive).
    * A relocation is requested while the driver is not eligible.
    * A relocation coordinate falls outside the global service area.
    * A relocation point index is out of range.
    """


class ConflictError(EnvironmentError):
    """Raised when two or more drivers bid for the same pending order.

    The environment NEVER auto-arbitrates conflicting bids. It terminates the
    step and reports the offending order id together with the competing driver
    ids, forcing the upstream multi-agent policy to resolve coordination.
    """

    def __init__(self, order_id, driver_ids):
        self.order_id = order_id
        self.driver_ids = list(driver_ids)
        super().__init__(
            f"Order {order_id!r} was simultaneously bid on by drivers "
            f"{self.driver_ids}. The environment does not arbitrate conflicts; "
            f"the upstream policy must coordinate to avoid this."
        )
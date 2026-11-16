from dataclasses import asdict, dataclass

PRODUCTION_MONTHS = ("2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06")
SHOCK_SEVERITY_BY_MONTH = dict(zip(PRODUCTION_MONTHS, (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), strict=True))


@dataclass(frozen=True)
class EconomicShock:
    interest_rate_hike_pp: float = 4.0
    nominal_inflation: float = 0.4
    new_profile_share: float = 0.4
    new_profile_income_factor: tuple[float, float] = (0.5, 0.8)
    new_profile_home_ownership: str = "RENT"
    stress_default_probability: float = 0.9
    stress_burden_floor: float = 0.05
    stress_burden_ceiling: float = 0.35
    stress_exposure_by_home_ownership: tuple[tuple[str, float], ...] = (
        ("RENT", 1.0),
        ("OTHER", 1.0),
        ("MORTGAGE", 0.5),
        ("OWN", 0.25),
    )

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_SHOCK = EconomicShock()

from typing import Tuple
from datetime import date
import calendar


def minus_months(year: int, month: int, diff_months: int) -> Tuple[int, int, date]:
    """Compute (year, month, last_day_date) shifted backwards by diff_months."""
    target_month = month - diff_months
    target_year = year

    while target_month <= 0:
        target_month += 12
        target_year -= 1

    last_day = calendar.monthrange(target_year, target_month)[1]
    return target_year, target_month, date(target_year, target_month, last_day)
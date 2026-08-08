"""
Currency conversion tool, exposed to the LLM via LangChain tool-calling
(see rag/answer.py, which binds this to the Groq model with .bind_tools).

This does NOT touch retrieval or RBAC - it's a pure calculator/lookup tool
for turning a dollar (or any currency) figure pulled from the filings into
another currency, e.g. "what was that revenue number in EUR?". The model
decides when to call it; we never call it ourselves.

Uses the free Frankfurter API (https://frankfurter.dev), which wraps
European Central Bank reference rates and requires no API key.
"""

import requests
from langchain_core.tools import tool

import config


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert a monetary amount from one currency to another using current
    (or latest available) foreign exchange reference rates.

    Use this whenever the user asks for a figure from the financial data to
    be expressed in a different currency, e.g. converting a USD revenue
    figure to EUR, GBP, JPY, etc.

    Args:
        amount: The numeric amount to convert (in the source currency).
        from_currency: Source currency as an ISO 4217 code, e.g. "USD".
        to_currency: Target currency as an ISO 4217 code, e.g. "EUR".
    """
    from_currency = (from_currency or "").strip().upper()
    to_currency = (to_currency or "").strip().upper()

    if not from_currency or not to_currency:
        return "Error: from_currency and to_currency must both be provided as currency codes (e.g. USD, EUR)."

    if from_currency == to_currency:
        return f"{amount:,.2f} {from_currency} = {amount:,.2f} {to_currency} (same currency, rate is 1.0)"

    try:
        # Use the updated v1 endpoint
        url = "https://api.frankfurter.dev/v1/latest"
        
        response = requests.get(
            url,
            # The API expects 'base' and 'symbols' now, not 'amount', 'from', 'to'
            params={"base": from_currency, "symbols": to_currency},
            timeout=config.CURRENCY_API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        
        rates = data.get("rates", {})
        if to_currency not in rates:
            return (
                f"Error: could not get a rate for {to_currency}. It may not be a "
                f"supported/valid ISO currency code."
            )
            
        # The API returns the exchange rate for 1 unit; we calculate the total amount here
        exchange_rate = rates[to_currency]
        converted = amount * exchange_rate
        
        return (
            f"{amount:,.2f} {from_currency} = {converted:,.2f} {to_currency} "
            f"(rate date: {data.get('date', 'latest')}, 1 {from_currency} = {exchange_rate:.4f} {to_currency})"
        )
    except requests.exceptions.RequestException as e:
        return f"Error: currency conversion service unavailable ({e})."
    except (ValueError, KeyError) as e:
        return f"Error: could not parse currency conversion response ({e})."


CURRENCY_TOOLS = [convert_currency]
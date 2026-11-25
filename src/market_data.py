
import logging
import os
import tomllib
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient


# --------------------------------------------------
# 1. Paths and logging
# --------------------------------------------------

# Project root is the parent directory of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "countries.toml"

logger = logging.getLogger("report")


# --------------------------------------------------
# 2. Create the ENTSO-E API client
# --------------------------------------------------

def get_entsoe_client() -> EntsoePandasClient:
    """
    Create and return an ENTSO-E API client
    """

    load_dotenv(PROJECT_ROOT / ".env")

    api_token = os.getenv("ENTSOE_API_TOKEN")

    if not api_token:
        raise RuntimeError(
            "ENTSOE_API_TOKEN is missing. "
            "Check that your .env file contains the API token."
        )

    return EntsoePandasClient(api_key=api_token)


# --------------------------------------------------
# 3. Load country and bidding-zone configuration
# --------------------------------------------------

def load_country_config() -> dict:
    """
    Load countries and bidding zones from countries.toml
    """

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("rb") as file:
        return tomllib.load(file)


# --------------------------------------------------
# 4. Retrieve day-ahead electricity prices
# --------------------------------------------------

def fetch_da_prices(
    client: EntsoePandasClient,
    country_code: str,
    timezone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """
    Fetch day-ahead electricity prices for a bidding zone.
    """

    logger.info(
        "Fetching day-ahead prices for %s from %s to %s",
        country_code,
        start,
        end,
    )

    return client.query_day_ahead_prices(
        country_code=country_code,
        start=start,
        end=end,
    )


# --------------------------------------------------
# 5. Retrieve prices for all configured bidding zones
# --------------------------------------------------

def load_day_ahead_data(
    days: int = 7,
) -> pd.DataFrame:
    """
    Retrieve day-ahead prices for all configured bidding zones.

    Returns a long-format DataFrame with:
    timestamp, bidding_zone, price
    """

    if days < 1:
        raise ValueError("'days' must be at least 1.")

    client = get_entsoe_client()
    config = load_country_config()

    results = {}

    for country, country_config in config["countries"].items():
        bidding_zones = country_config.get("bidding_zones", [])
        timezone = country_config["timezone"]

        if not bidding_zones:
            logger.warning(
                "No bidding zones configured for %s. Skipping.",
                country,
            )
            continue

        # Use the current local day's start as the end boundary.
        end = pd.Timestamp.now(tz=timezone).normalize()
        start = end - pd.Timedelta(days=days)

        for bidding_zone in bidding_zones:
            zone_name = bidding_zone["name"]

            try:
                prices = fetch_da_prices(
                    client=client,
                    country_code=zone_name,
                    timezone=timezone,
                    start=start,
                    end=end,
                )

                if prices is None or prices.empty:
                    logger.warning(
                        "No price data for %s (%s).",
                        country,
                        zone_name,
                    )
                    continue

                results[zone_name] = prices

            except Exception:
                logger.exception(
                    "Failed to fetch prices for %s (%s).",
                    country,
                    zone_name,
                )

    if not results:
        raise RuntimeError(
            "No day-ahead price data could be retrieved."
        )

    # Combine bidding zones into one DataFrame.
    prices_df = pd.DataFrame(results)

    # Convert to long format for analysis and plotting.
    prices_long = (
        prices_df
        .rename_axis("timestamp")
        .reset_index()
        .melt(
            id_vars="timestamp",
            var_name="bidding_zone",
            value_name="price",
        )
        .dropna(subset=["price"])
    )

    logger.info(
        "Retrieved %s observations across %s bidding zones.",
        len(prices_long),
        prices_long["bidding_zone"].nunique(),
    )

    return prices_long

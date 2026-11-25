import logging
import pandas as pd


logger = logging.getLogger("report")


# --------------------------------------------------
# 1. Validate input data
# --------------------------------------------------

def validate_price_data(prices: pd.DataFrame) -> None:
    """
    Validate the structure of the day-ahead price dataset.
    """

    required_columns = {"timestamp", "bidding_zone", "price"}
    missing_columns = required_columns - set(prices.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if prices.empty:
        raise ValueError("The price dataset is empty.")


# --------------------------------------------------
# 2. Calculate market statistics
# --------------------------------------------------

def calculate_market_statistics(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate descriptive statistics for each bidding zone.

    Expected input columns:
        timestamp, bidding_zone, price

    Returns one row per bidding zone.
    """

    validate_price_data(prices)

    data = prices.copy()
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.dropna(subset=["price"])

    if data.empty:
        raise ValueError("No valid price observations are available.")

    statistics = (
        data.groupby("bidding_zone")["price"]
        .agg(
            observations="count",
            mean_price="mean",
            median_price="median",
            std_price="std",
            min_price="min",
            max_price="max",
        )
        .reset_index()
    )

    # Additional indicators
    negative_prices = (
        data.assign(is_negative=data["price"] < 0)
        .groupby("bidding_zone")["is_negative"]
        .sum()
        .rename("negative_price_observations")
    )

    statistics = statistics.merge(
        negative_prices,
        on="bidding_zone",
        how="left",
    )

    statistics["negative_price_pct"] = (
        statistics["negative_price_observations"]
        / statistics["observations"]
        * 100
    )

    # Round numerical results for presentation
    numeric_columns = [
        "mean_price",
        "median_price",
        "std_price",
        "min_price",
        "max_price",
        "negative_price_pct",
    ]

    statistics[numeric_columns] = statistics[numeric_columns].round(2)

    statistics = statistics.sort_values(
        "mean_price",
        ascending=False,
    ).reset_index(drop=True)

    logger.info(
        "Calculated statistics for %s bidding zones.",
        len(statistics),
    )

    return statistics


# --------------------------------------------------
# 3. Calculate hourly price patterns
# --------------------------------------------------

def calculate_hourly_patterns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate average day-ahead prices by hour and bidding zone.

    Returns:
        bidding_zone, hour, mean_price
    """

    validate_price_data(prices)

    data = prices.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.dropna(subset=["timestamp", "price"])

    data["hour"] = data["timestamp"].dt.hour

    hourly_patterns = (
        data.groupby(["bidding_zone", "hour"])["price"]
        .mean()
        .reset_index(name="mean_price")
    )

    hourly_patterns["mean_price"] = (
        hourly_patterns["mean_price"].round(2)
    )

    return hourly_patterns


# --------------------------------------------------
# 4. Calculate daily average prices
# --------------------------------------------------

def calculate_daily_prices(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate daily average prices for each bidding zone.

    The date is derived from each observation's timestamp.
    """

    validate_price_data(prices)

    data = prices.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.dropna(subset=["timestamp", "price"])

    # Remove timezone information after retaining local date/time.
    # This allows daily grouping by the date represented by each timestamp.
    timestamps = data["timestamp"]

    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_localize(None)

    data["date"] = timestamps.dt.date

    daily_prices = (
        data.groupby(["date", "bidding_zone"])["price"]
        .mean()
        .reset_index(name="mean_price")
    )

    daily_prices["mean_price"] = daily_prices["mean_price"].round(2)

    return daily_prices


# --------------------------------------------------
# 5. Identify price extremes
# --------------------------------------------------

def identify_price_extremes(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the minimum and maximum observed prices for each zone.

    The timestamps identify when the extreme prices occurred.
    """

    validate_price_data(prices)

    data = prices.copy()
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.dropna(subset=["price", "timestamp"])

    if data.empty:
        raise ValueError("No valid price observations are available.")

    min_indices = data.groupby("bidding_zone")["price"].idxmin()
    max_indices = data.groupby("bidding_zone")["price"].idxmax()

    minimums = data.loc[
        min_indices, ["bidding_zone", "timestamp", "price"]
    ].rename(
        columns={
            "timestamp": "min_timestamp",
            "price": "min_price",
        }
    )

    maximums = data.loc[
        max_indices, ["bidding_zone", "timestamp", "price"]
    ].rename(
        columns={
            "timestamp": "max_timestamp",
            "price": "max_price",
        }
    )

    extremes = minimums.merge(
        maximums,
        on="bidding_zone",
        how="inner",
    )

    return extremes.reset_index(drop=True)


# --------------------------------------------------
# 6. Generate an executive summary
# --------------------------------------------------

def generate_executive_summary(
    statistics: pd.DataFrame,
) -> str:
    """
    Generate a concise Markdown executive summary
    from the calculated market statistics.
    """

    if statistics.empty:
        return "No market statistics are available."

    required_columns = {
        "bidding_zone",
        "mean_price",
        "min_price",
        "max_price",
        "std_price",
        "negative_price_pct",
    }

    missing_columns = required_columns - set(statistics.columns)

    if missing_columns:
        raise ValueError(
            "Statistics are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    data = statistics.dropna(subset=["mean_price"])

    if data.empty:
        return "No valid average prices are available."

    highest = data.loc[data["mean_price"].idxmax()]
    lowest = data.loc[data["mean_price"].idxmin()]

    overall_mean = data["mean_price"].mean()

    highest_volatility = data.dropna(subset=["std_price"])

    if not highest_volatility.empty:
        most_volatile = highest_volatility.loc[
            highest_volatility["std_price"].idxmax()
        ]
        volatility_text = (
            f"- **Highest price volatility:** "
            f"{most_volatile['bidding_zone']} "
            f"(standard deviation: {most_volatile['std_price']:.2f} "
            f"€/MWh).\n"
        )
    else:
        volatility_text = (
            "- **Price volatility:** Not available.\n"
        )

    negative_zones = data[data["negative_price_pct"] > 0]

    if not negative_zones.empty:
        negative_text = (
            f"- **Negative prices:** Observed in "
            f"{len(negative_zones)} bidding zone(s). "
            "See the detailed statistics for their frequency.\n"
        )
    else:
        negative_text = (
            "- **Negative prices:** No negative-price observations "
            "were recorded in the available data.\n"
        )

    summary = (
        f"- **Highest average price:** "
        f"{highest['bidding_zone']} "
        f"({highest['mean_price']:.2f} €/MWh).\n"
        f"- **Lowest average price:** "
        f"{lowest['bidding_zone']} "
        f"({lowest['mean_price']:.2f} €/MWh).\n"
        f"- **Average of bidding-zone averages:** "
        f"{overall_mean:.2f} €/MWh.\n"
        f"{volatility_text}"
        f"{negative_text}"
    )

    return summary

# %%
import tomllib
from pprint import pprint

with open("countries.toml", "rb") as f:
    config = tomllib.load(f)

pprint(config)


# %%
bidding_zones = [
    zone["name"]
    for zone in config["countries"]["Norway"]["bidding_zones"]
]
pprint(bidding_zones)


# %%
for country, data in config["countries"].items():

    print(f"\n{data['name']} ({data['country_code']})")

    for zone in data.get("bidding_zones", []):
        assert zone["eic"], f"Missing EIC: {country}/{zone['name']}"

        print(f"  {zone['name']}: {zone['eic']}")

print("\nConfiguration validated!")

# %%

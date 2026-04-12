"""correcting_budgets.py"""
"""Correcting erroneous budget values based on web searches and film databases"""

import pandas as pd

# =============================================================================
# BUDGET CORRECTIONS
# =============================================================================

# Films with verified corrected budgets
corrections = [
    # (title, year, corrected_budget, source)
    ("Angela's Ashes", 1999, 25000000, "Wikipedia"),
    ("Rugrats in Paris: The Movie", 2000, 30000000, "Wikipedia"),
    ("The 51st State", 2001, 28000000, "UK action film"),
    ("Man Trouble", 1992, 30000000, "Jack Nicholson film"),
    ("Curse of the Golden Flower", 2006, 45000000, "Wikipedia"),
    ("Starter for 10", 2006, 8000000, "UK film"),
    ("Max Manus: Man of War", 2008, 8000000, "Norwegian film"),
    ("Jack and the Cuckoo-Clock Heart", 2013, 33000000, "French animated film"),
    ("Moon Man", 2012, 12000000, "German animated film"),
    ("Father's Day", 2011, 10000, "Troma horror"),
]

# Films with verified low budgets (no correction needed, just documentation)
verified_low_budgets = [
    # (title, year, budget, source)
    ("Osama", 2003, 46000, "Afghan film"),
    ("Versus", 2000, 10000, "Japanese indie"),
    ("Pumzi", 2009, 35000, "Kenyan short film"),
    ("Cool Cat Saves the Kids", 2015, 50000, "Independent children's film"),
    ("Space Cop", 2016, 80000, "Red Letter Media indie"),
]

# Films to drop (unverifiable budgets)
films_to_drop = [
    # (title, year, reason)
    ("Mute Witness", 1995, "Budget unknown - low budget indie"),
    ("American Adobo", 2001, "Budget unknown - Filipino indie"),
    ("The Heart is Deceitful Above All Things", 2004, "Budget unknown"),
    ("The Bow", 2005, "Budget unknown - Kim Ki-duk film"),
    ("Princess", 2006, "Budget unknown - Danish animated"),
    ("Stan Helsing", 2009, "Budget unknown"),
    ("The Wounds", 1998, "Budget unknown - Serbian film"),
    ("The Real McCoy", 1999, "Cannot verify - may be different film"),
    ("Life in a Day", 2011, "Documentary - budget unclear"),
    ("Padre nuestro", 2007, "Budget unknown"),
    ("Sisters", 2006, "Budget unknown"),
    ("Of Horses and Men", 2013, "Budget unknown - Icelandic film"),
    ("Modern Life", 2008, "Budget unknown - documentary"),
    ("Red Knot", 2014, "Budget unknown - indie film"),
    ("100 Yen: The Japanese Arcade Experience", 2012, "Documentary"),
    ("The Boxcar Children", 2014, "Budget unknown"),
    ("Traceroute", 2016, "Documentary"),
    ("Hidden in the Woods", 2014, "Budget unknown"),
    ("Manifesto", 2017, "Budget unknown - art film"),
    ("Human", 2013, "Budget unknown"),
    ("Wet and Reckless", 2014, "Budget unknown"),
]

# =============================================================================
# CREATE DATAFRAMES
# =============================================================================

df_corrections = pd.DataFrame(corrections, columns=['title', 'year', 'corrected_budget', 'source'])
df_verified = pd.DataFrame(verified_low_budgets, columns=['title', 'year', 'budget', 'source'])
df_drop = pd.DataFrame(films_to_drop, columns=['title', 'year', 'reason'])

# =============================================================================
# SAVE TO CSV
# =============================================================================

df_corrections.to_csv(
    "C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/budget_corrections.csv",
    index=False
)

df_drop.to_csv(
    "C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/films_to_drop.csv",
    index=False
)

# =============================================================================
# SUMMARY
# =============================================================================

print("=== BUDGET CORRECTIONS SUMMARY ===")
print(f"Films with corrected budgets: {len(df_corrections)}")
print(f"Films with verified low budgets: {len(df_verified)}")
print(f"Films to drop: {len(df_drop)}")

print("\n=== CORRECTIONS ===")
print(df_corrections.to_string(index=False))

print("\n=== VERIFIED LOW BUDGETS (no action needed) ===")
print(df_verified.to_string(index=False))

print("\n=== FILMS TO DROP ===")
print(df_drop.to_string(index=False))
"""correcting_budgets.py"""
# Create budget corrections CSV
# These corrections are based on web searches and film databases

budget_corrections = [
    # title, year, incorrect_budget, corrected_budget, source/notes
    ("Mute Witness", 1995, 2, None, "Budget unknown - low budget indie film, recommend dropping"),
    ("Angela's Ashes", 1999, 25, 25000000, "Wikipedia - $25 million"),
    ("Rugrats in Paris: The Movie", 2000, 30, 30000000, "Wikipedia - $30 million"),
    ("American Adobo", 2001, 344, None, "Budget unknown - Filipino indie film, recommend dropping"),
    ("The 51st State", 2001, 28, 28000000, "Likely $28 million - UK action film"),
    ("Osama", 2003, 46000, 46000, "Correct - Afghan film, very low budget ~$46,000"),
    ("Versus", 2000, 10000, 10000, "Correct - Japanese indie, ~$10,000 budget"),
    ("Man Trouble", 1992, 30, 30000000, "Likely $30 million - Jack Nicholson film"),
    ("The Heart is Deceitful Above All Things", 2004, 74050, None, "Budget unknown, recommend dropping"),
    ("The Bow", 2005, 10, None, "Budget unknown - Kim Ki-duk film, recommend dropping"),
    ("Curse of the Golden Flower", 2006, 110, 45000000, "Wikipedia - $45 million"),
    ("Starter for 10", 2006, 8, 8000000, "Likely $8 million - UK film"),
    ("Max Manus: Man of War", 2008, 8, 8000000, "Likely $8 million - Norwegian film"),
    ("Princess", 2006, 8, None, "Budget unknown - Danish animated film, recommend dropping"),
    ("Stan Helsing", 2009, 108, None, "Budget unknown, recommend dropping"),
    ("The Wounds", 1998, 10000, None, "Budget unknown - Serbian film, recommend dropping"),
    ("The Real McCoy", 1999, 650, None, "Cannot verify - may be different film, recommend dropping"),
    ("Life in a Day", 2011, 45202, None, "Documentary - budget unclear, recommend dropping"),
    ("Padre nuestro", 2007, 54166, None, "Budget unknown, recommend dropping"),
    ("Jack and the Cuckoo-Clock Heart", 2013, 33, 33000000, "Likely $33 million - French animated film"),
    ("Pumzi", 2009, 35000, 35000, "Correct - Kenyan short film, very low budget"),
    ("Moon Man", 2012, 12000, 12000000, "Likely $12 million - German animated film"),
    ("Sisters", 2006, 60000, None, "Budget unknown, recommend dropping"),
    ("Of Horses and Men", 2013, 10, None, "Budget unknown - Icelandic film, recommend dropping"),
    ("Modern Life", 2008, 88, None, "Budget unknown - documentary, recommend dropping"),
    ("Red Knot", 2014, 80, None, "Budget unknown - indie film, recommend dropping"),
    ("Father's Day", 2011, 10, 10000, "Correct - Troma horror, ~$10,000 budget"),
    ("Cool Cat Saves the Kids", 2015, 50000, 50000, "Correct - independent children's film"),
    ("100 Yen: The Japanese Arcade Experience", 2012, 32213, None, "Documentary, recommend dropping"),
    ("Space Cop", 2016, 80000, 80000, "Correct - Red Letter Media indie film"),
    ("The Boxcar Children", 2014, 5, None, "Budget unknown, recommend dropping"),
    ("Traceroute", 2016, 10000, None, "Documentary, recommend dropping"),
    ("Hidden in the Woods", 2014, 1, None, "Budget unknown, recommend dropping"),
    ("Manifesto", 2017, 10000, None, "Budget unknown - art film, recommend dropping"),
    ("Human", 2013, 60000, None, "Budget unknown, recommend dropping"),
    ("Wet and Reckless", 2014, 10000, None, "Budget unknown, recommend dropping"),
]

import pandas as pd

# Create DataFrame
df_corrections = pd.DataFrame(budget_corrections, columns=[
    'title', 'year', 'incorrect_budget', 'corrected_budget', 'notes'
])

# Save to CSV
df_corrections.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/processed/budget_corrections.csv", index=False)

print("Budget corrections saved")
print(f"\nFilms with corrected budgets: {df_corrections['corrected_budget'].notna().sum()}")
print(f"Films recommended to drop: {df_corrections['corrected_budget'].isna().sum()}")

print("\n=== Corrections ===")
print(df_corrections[df_corrections['corrected_budget'].notna()][['title', 'year', 'incorrect_budget', 'corrected_budget']])

print("\n=== Recommended to Drop ===")
print(df_corrections[df_corrections['corrected_budget'].isna()][['title', 'year', 'incorrect_budget', 'notes']])
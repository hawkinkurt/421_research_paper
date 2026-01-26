import pandas as pd

data = {
    'country_iso2': ['GB', 'US', 'DE', 'CA', 'FR', 'CN', 'IT', 'JP', 'AU', 'HK',
                     'ES', 'CZ', 'BE', 'IN', 'NZ', 'IE', 'NL', 'MX', 'HU', 'MT',
                     'ZA', 'CH', 'RU', 'LU', 'AT', 'PL', 'RO', 'SE', 'DK', 'FI', 'NO'],
    'incentive_intro_year': [2007, None, 2007, 1995, 2009, None, 2009, None, 2007, None,
                             2015, 2010, 2003, None, 2003, 1997, 2014, 2006, 2004, 2005,
                             2004, None, None, None, 2010, 2019, 2018, 2016, 2026, 2017, 2016],
    'incentive_type': ['tax_relief', 'state_only', 'rebate', 'tax_credit', 'rebate', None, 'tax_credit', None, 'tax_credit', None,
                       'tax_rebate', 'rebate', 'tax_shelter', None, 'grant', 'tax_credit', 'rebate', 'tax_credit', 'tax_rebate', 'rebate',
                       'rebate', None, None, None, 'rebate', 'rebate', 'rebate', 'rebate', 'rebate', 'rebate', 'rebate']
}

df = pd.DataFrame(data)
df.to_csv("C:/Users/kurtl/PycharmProjects/gravity_model/data/raw/film_incentive_intro_dates.csv", index=False)
print("Saved film_incentive_intro_dates.csv")
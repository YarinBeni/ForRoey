import pandas as pd
from scipy import stats
import numpy as np


class AdDataAnalyzer:
    def __init__(self, data_file):
        self.df = pd.read_excel(data_file)
        print(f"Number of rows before dropna: {len(self.df.index)}")
        self.df.dropna(inplace=True)
        print(f"Number of rows after dropna: {len(self.df.index)}")
        self.clean_data()
        self.hypothesis_test()
        self.calculate_proportions()
        self.calculate_statistics()

    def clean_data(self):
        self.df['Impressions'] = self.df['Impressions'].replace(',', '').astype(int)
        self.df['Clicks'] = self.df['Clicks'].replace(',', '').astype(int)
        self.df['Actions'] = self.df['Actions'].replace(',', '').astype(int)
        self.df['Conversion Rate'] = np.where(self.df['Clicks'] != 0, self.df['Actions'] / self.df['Clicks'], 0)
        self.df['Clicks per Impressions'] = np.where(self.df['Impressions'] != 0,
                                                     self.df['Clicks'] / self.df['Impressions'], 0)
        self.df['Spent per Actions'] = np.where(self.df['Actions'] != 0, self.df['Spent'] / self.df['Actions'], 0)

    def calculate_proportions(self):
        total_impressions = self.df['Impressions'].sum()
        total_actions = self.df['Actions'].sum()
        total_clicks = self.df['Clicks'].sum()
        self.df['Impressions Proportion'] = self.df['Impressions'] / total_impressions
        self.df['Actions Proportion'] = self.df['Actions'] / total_actions
        self.df['Clicks Proportion'] = self.df['Clicks'] / total_clicks

    def calculate_statistics(self):
        for publisher in ['Bruiser.com', 'Honey.com']:
            self.calculate_publisher_statistics(publisher)

    def calculate_publisher_statistics(self, publisher):
        percentile_clicks = stats.percentileofscore(self.df['Clicks per Impressions'],
                                                    self.df.loc[self.df[
                                                                    'Publisher'] == publisher, 'Clicks per Impressions'].mean())
        percentile_spent = stats.percentileofscore(self.df['Spent per Actions'],
                                                   self.df.loc[
                                                       self.df['Publisher'] == publisher, 'Spent per Actions'].mean())
        percentile_CR = stats.percentileofscore(self.df['Conversion Rate'],
                                                self.df.loc[
                                                    self.df['Publisher'] == publisher, 'Conversion Rate'].mean())
        percentile_impressions = stats.percentileofscore(self.df['Impressions'],
                                                         self.df.loc[
                                                             self.df['Publisher'] == publisher, 'Impressions'].mean())
        impressions_proportion = self.df.loc[self.df['Publisher'] == publisher, 'Impressions Proportion'].sum()
        actions_proportion = self.df.loc[self.df['Publisher'] == publisher, 'Actions Proportion'].sum()
        clicks_proportion = self.df.loc[self.df['Publisher'] == publisher, 'Clicks Proportion'].sum()

        print(
            f"For {publisher}:\nImpressions Proportion: {impressions_proportion}\nClicks Proportion: {clicks_proportion}"
            f"\nActions Proportion: {actions_proportion}\npercentile in Impressions: {percentile_impressions}"
            f"\npercentile in Conversion Rate: {percentile_CR}\npercentile in Spent per Action: {percentile_spent}"
            f"\npercentile in Clicks per Impressions: {percentile_clicks}\n")

    def hypothesis_test(self):
        # Calculate the average conversion rate for all rows that aren't Bruiser.com or Honey.com
        average_conversion_rate = self.df.loc[
            ~self.df['Publisher'].isin(['Bruiser.com', 'Honey.com']), 'Conversion Rate'].mean()
        print(f'Average Conversion Rate (excluding Bruiser.com and Honey.com): {average_conversion_rate}')
        # Calculate Bruiser.com and Honey.com conversion rates
        bruiser_conversion_rate = self.df.loc[self.df['Publisher'] == 'Bruiser.com', 'Conversion Rate'].mean()
        honey_conversion_rate = self.df.loc[self.df['Publisher'] == 'Honey.com', 'Conversion Rate'].mean()
        print(f'Bruiser.com Conversion Rate: {bruiser_conversion_rate}')
        print(f'Honey.com Conversion Rate: {honey_conversion_rate}')
        # Perform t-tests to compare the conversion rates of Bruiser.com and Honey.com with the average
        t_stat_bruiser, p_value_bruiser = stats.ttest_1samp(
            self.df.loc[self.df['Publisher'] == 'Bruiser.com', 'Conversion Rate'],
            average_conversion_rate)
        t_stat_honey, p_value_honey = stats.ttest_1samp(
            self.df.loc[self.df['Publisher'] == 'Honey.com', 'Conversion Rate'],
            average_conversion_rate, alternative='less')
        # Print the p-values
        print(f'p-value for Bruiser.com: {p_value_bruiser}')
        print(f'p-value for Honey.com: {p_value_honey}')
        # Check if the advertiser's complaint is valid at 90% and 95% confidence levels
        for confidence in [0.90, 0.95]:
            print(f'\nAt {confidence * 100}% confidence level:')
            if p_value_bruiser < 1 - confidence:
                print("Bruiser.com's conversion rate is significantly different from the average.")
            else:
                print("Bruiser.com's conversion rate is not significantly different from the average.")

            if p_value_honey < 1 - confidence:
                print("Honey.com's conversion rate is significantly different from the average.")
            else:
                print("Honey.com's conversion rate is not significantly different from the average.")
        print()


AdDataAnalyzer('Data.xlsx')

#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Prediction application.

"""


# Import standard packages.
import bs4
import collections
import inspect
import logging
import os
import requests
import shelve
import sys
import textwrap
import time
# Import installed packages.
import geopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Import local packages.
from .. import utils


# Define module exports:
__all__ = ['etl']


# Define state settings and globals.
# Note: For non-root-level loggers, use `getLogger(__name__)`
#     http://stackoverflow.com/questions/17336680/python-logging-with-multiple-modules-does-not-work
logger = logging.getLogger(__name__)
# Set the matplotlib backend to the Anti-Grain Geometry C++ library.
# Note: Use plt.switch_backend since matplotlib.use('agg') before importing pyplot fails.
plt.switch_backend('agg')
# Set matplotlib styles with seaborn
sns.set()


# Define globals
# Return rates over 10% are considered excessive.
buyer_retrate_max = 0.1
# Return rate is ratio of Returned=1 to Returned not null.
buyer_retrate = 'BuyerID_fracReturned1DivReturnedNotNull'


def etl(
    df:pd.DataFrame
    ) -> pd.DataFrame:
    r"""Extract-transform-load.

    Args:
        df (pandas.DataFrame): Dataframe of raw data.

    Returns:
        df (pandas.DataFrame): Dataframe of formatted, cleaned data.

    TODO:
        * Modularize script into separate helper functions.
        * Modify dataframe in place

    """
    # Check input.
    # Copy dataframe to avoid in place modification.
    df = df.copy()
    ########################################
    # DSEligible, Returned: Fix DSEligible == 0 but Returned not null
    # Some vehicles have DSEligible=0 but have Returned!=nan due to errors or extenuating circumstances.
    # To correct: If Returned!=nan, then DSEligible=1
    logger.info(textwrap.dedent("""\
        DSEligible, Returned: Fix DSEligible == 0 but Returned not null.
        To correct: If Returned not null, then DSEligible = 1."""))
    logger.info("Before:\n{pt}".format(
        pt=pd.pivot_table(
            df[['DSEligible', 'Returned']].astype(str),
            index='DSEligible', columns='Returned',
            aggfunc=len, margins=True, dropna=False)))
    df.loc[df['Returned'].notnull(), 'DSEligible'] = 1
    logger.info("After:\n{pt}".format(
        pt=pd.pivot_table(
            df[['DSEligible', 'Returned']].astype(str),
            index='DSEligible', columns='Returned',
            aggfunc=len, margins=True, dropna=False)))
    ########################################
    # Returned: fill nulls
    # NOTE: THIS TRANSFORMATION MUST BE BEFORE FEATURE CREATION SINCE Returned.isnull() -> -1
    # Fill null values with -1 and cast to int.
    logger.info('Returned: Fill nulls with -1 and cast to int.')
    logger.info("Before:\n{pt}".format(
        pt=pd.pivot_table(
            df[['DSEligible', 'Returned']].astype(str),
            index='DSEligible', columns='Returned',
            aggfunc=len, margins=True, dropna=False)))
    df['Returned'] = df[['Returned']].fillna(value=-1).astype(int)
    logger.info("After:\n{pt}".format(
        pt=pd.pivot_table(
            df[['DSEligible', 'Returned']].astype(str),
            index='DSEligible', columns='Returned',
            aggfunc=len, margins=True, dropna=False)))
    ########################################
    # BuyerID, SellerID, VIN, SellingLocation, CarMake, JDPowersCat:
    # Cast to strings as categorical features.
    logger.info(textwrap.dedent("""\
        BuyerID, SellerID, VIN, SellingLocation, CarMake, JDPowersCat:
        Cast to strings as categorical features."""))
    for col in ['BuyerID', 'SellerID', 'VIN', 'SellingLocation', 'CarMake', 'JDPowersCat']:
        df[col] = df[col].astype(str)
    ########################################
    # CarMake: Deduplicate
    # TODO: Find/scrape hierarchical relationships between car brands
    #     (e.g. https://en.wikipedia.org/wiki/Category:Mid-size_cars). To business people: would that be helpful?
    # TODO: Deduplicate with spelling corrector.
    logger.info("CarMake: Deduplicate.")
    carmake_dedup = {
        '1SUZU': 'ISUZU',
        'CHEVY': 'CHEVROLET',
        'XHEVY': 'CHEVROLET',
        'DAMON': 'DEMON',
        'FORESTR':'FORESTRIVER',
        'FORESTRIVE': 'FORESTRIVER',
        'FREIGHTLI': 'FREIGHTLINER',
        'FREIGHTLIN': 'FREIGHTLINER',
        'FRIGHTLIE': 'FREIGHTLINER',
        'FRTLNRL': 'FREIGHTLINER',
        'XFREIGHTLN': 'FREIGHTLINER',
        'XREIGHTL': 'FREIGHTLINER',
        'HARLEY': 'HARLEYDAVIDSON',
        'HARLEY-DAV': 'HARLEYDAVIDSON',
        'INTERNATIO': 'INTERNATIONAL',
        'INTERNATL': 'INTERNATIONAL',
        'XINTERNATI': 'INTERNATIONAL',
        'MERCEDES': 'MERCEDES-BENZ',
        'nan': 'UNKNOWN',
        'XHINO': 'HINO',
        'XOSHKOSH': 'OSHKOSH',
        'XSMART': 'SMART'}
    df['CarMake'] = df['CarMake'].str.replace(' ', '').apply(
        lambda car: carmake_dedup[car] if car in carmake_dedup else car)
    # # TODO: Experiment with one-hot encoding (problem is that it doesn't scale)
    # df = pd.merge(
    #     left=df,
    #     right=pd.get_dummies(df['CarMake'], prefix='CarMake'),
    #     left_index=True,
    #     right_index=True)
    ########################################
    # LIGHTG, LIGHTY, LIGHTR
    # Retain light with highest warning
    logger.info("LIGHT*: Only retain light with highest warning.")
    pt = pd.DataFrame([
        df.loc[df['LIGHTG']==1, ['LIGHTG', 'LIGHTY', 'LIGHTR']].sum(),
        df.loc[df['LIGHTY']==1, ['LIGHTG', 'LIGHTY', 'LIGHTR']].sum(),
        df.loc[df['LIGHTR']==1, ['LIGHTG', 'LIGHTY', 'LIGHTR']].sum()],
        index=['LIGHTG=1', 'LIGHTY=1', 'LIGHTR=1'])
    pt.columns = ['LIGHTG=1', 'LIGHTY=1', 'LIGHTR=1']
    logger.info("Before:\n{pt}".format(pt=pt))
    df.loc[df['LIGHTR']==1, ['LIGHTG', 'LIGHTY']] = 0
    df.loc[df['LIGHTY']==1, ['LIGHTG']] = 0
    pt = pd.DataFrame([
        df.loc[df['LIGHTG']==1, ['LIGHTG', 'LIGHTY', 'LIGHTR']].sum(),
        df.loc[df['LIGHTY']==1, ['LIGHTG', 'LIGHTY', 'LIGHTR']].sum(),
        df.loc[df['LIGHTR']==1, ['LIGHTG', 'LIGHTY', 'LIGHTR']].sum()],
        index=['LIGHTG=1', 'LIGHTY=1', 'LIGHTR=1'])
    pt.columns = ['LIGHTG=1', 'LIGHTY=1', 'LIGHTR=1']
    logger.info("After:\n{pt}".format(pt=pt))
    ########################################
    # SaleDate: Cast to datetime.
    logger.info("SaleDate: Cast to datetime.")
    if df['SaleDate'].dtype == 'O':
        df['SaleDate'] = pd.to_datetime(df['SaleDate'], format=r'%y-%m-%d')
    ########################################
    # Autocheck_score: Fill null values with mode (1)
    # TODO: Use nearest neighbors to infer probable fill value.
    logger.info("Autocheck_score: Fill null values with mode (1).")
    df['Autocheck_score'] = df['Autocheck_score'].fillna(value=1)
    ########################################
    # ConditionReport
    # Map character codes to numerical values, invalid codes are "average".
    logger.info("ConditionReport: Map character codes to numerical values. Invalid codes are 'average'.")
    conrpt_value = {
        'EC': 50,
        'CL': 40,
        'AV': 30,
        'RG': 20,
        'PR': 10,
        'SL': 0,
        'A': 30,
        'A3': 30,
        'Y6': 30,
        'nan': 30}
    df['ConditionReport'] = df['ConditionReport'].astype(str).apply(
        lambda con: conrpt_value[con] if con in conrpt_value else con)
    df['ConditionReport'] = df['ConditionReport'].astype(int)
    return df


def create_features(
    df:pd.DataFrame,
    path_data_dir:str
    ) -> pd.DataFrame:
    r"""Create features for post-ETL data.

    Args:
        df (pandas.DataFrame): Dataframe of raw data.
        path_data_dir (str): Path to data directory for caching geocode shelf file.

    Returns:
        df (pandas.DataFrame): Dataframe of extracted data.

    See Also:
        etl

    Notes:
        * BuyerID_fracReturned1DivReturnedNotNull is the return rate for a buyer.

    TODO:
        * Modularize script into separate helper functions.
        * Modify dataframe in place

    """
    # Check input.
    # Copy dataframe to avoid in place modification.
    df = df.copy()
    # Check file path.
    if not os.path.exists(path_data_dir):
        raise IOError(textwrap.dedent("""\
            Path does not exist:
            path_data_dir = {path}""".format(
                path=path_data_dir)))
    ########################################
    # Returned_asm
    # Interpretation of assumptions:
    # If DSEligible=0, then the vehicle is not eligible for a guarantee.
    # * And Returned=-1 (null) since we don't know whether or not it would have been returned,
    #   but given that it wasn't eligible, it may have been likely to have Returned=1.
    # If DSEligible=1, then the vehicle is eligible for a guarantee.
    # * And if Returned=0 then the guarantee was purchased and the vehicle was not returned.
    # * And if Returned=1 then the guarantee was purchased and the vehicle was returned.
    # * And if Returned=-1 (null) then the guarantee was not purchased.
    #   We don't know whether or not it would have been returned,
    #   but given that the dealer did not purchase, it may have been likely to have Returned=0.
    # Assume:
    # If Returned=-1 and DSEligible=0, then Returned_asm=1
    # If Returned=-1 and DSEligible=1, then Returned_asm=0
    logger.info(textwrap.dedent("""\
        Returned_asm: Assume returned status to fill nulls as new feature.
        If Returned=-1 and DSEligible=0, then Returned_asm=1 (assumes low P(resale|buyer, car))
        If Returned=-1 and DSEligible=1, then Returned_asm=0 (assumes high P(resale|buyer, car))"""))
    df['Returned_asm'] = df['Returned']
    df.loc[
        np.logical_and(df['Returned'] == -1, df['DSEligible'] == 0),
        'Returned_asm'] = 1
    df.loc[
        np.logical_and(df['Returned'] == -1, df['DSEligible'] == 1),
        'Returned_asm'] = 0
    logger.info("Relationship between DSEligible and Returned:\n{pt}".format(
        pt=pd.pivot_table(
            df[['DSEligible', 'Returned']].astype(str),
            index='DSEligible', columns='Returned',
            aggfunc=len, margins=True, dropna=False)))
    logger.info("Relationship between DSEligible and Returned_asm:\n{pt}".format(
        pt=pd.pivot_table(
            df[['DSEligible', 'Returned_asm']].astype(str),
            index='DSEligible', columns='Returned_asm',
            aggfunc=len, margins=True, dropna=False)))
    logger.info("Relationship between Returned and Returned_asm:\n{pt}".format(
        pt=pd.pivot_table(
            df[['Returned', 'Returned_asm']].astype(str),
            index='Returned', columns='Returned_asm',
            aggfunc=len, margins=True, dropna=False)))
    ########################################
    # SellingLocation_lat, SellingLocation_lon
    # Cell takes ~1 min to execute if shelf does not exist.
    # Google API limit: https://developers.google.com/maps/documentation/geocoding/usage-limits
    logger.info(textwrap.dedent("""\
        SellingLocation: Geocode.
        Scraping webpages for addresses and looking up latitude, longitude coordinates."""))
    path_shelf = os.path.join(path_data_dir, 'sellloc_geoloc.shelf')
    seconds_per_query = 1.0/50.0 # Google API limit
    sellloc_geoloc = dict()
    with shelve.open(filename=path_shelf, flag='c') as shelf:
        for loc in df['SellingLocation'].unique():
            if loc in shelf:
                raw = shelf[loc]
                if raw is None:
                    location = raw
                else:
                    address = raw['formatted_address']
                    latitude = raw['geometry']['location']['lat']
                    longitude = raw['geometry']['location']['lng']
                    location = geopy.location.Location(
                        address=address, point=(latitude, longitude), raw=raw)
            else:        
                url = r'https://www.manheim.com/locations/{loc}/events'.format(loc=loc)
                page = requests.get(url)
                tree = bs4.BeautifulSoup(page.text, 'lxml')
                address = tree.find(name='p', class_='loc_address').get_text().strip()
                try:
                    components = {
                        'country': 'United States',
                        'postal_code': address.split()[-1]}
                    location = geopy.geocoders.GoogleV3().geocode(
                        query=address,
                        exactly_one=True,
                        components=components)
                except:
                    logger.warning(textwrap.dedent("""\
                        Exception raised. Setting {loc} geo location to `None`
                        sys.exc_info() =
                        {exc}""".format(loc=loc, exc=sys.exc_info())))
                    location = None
                finally:
                    time.sleep(seconds_per_query)
                    if location is None:
                        shelf[loc] = location
                    else:
                        shelf[loc] = location.raw
            sellloc_geoloc[loc] = location
    logger.info("Mapping SellingLocation to latitude, longitude coordinates.")
    sellloc_lat = {
        sellloc: (geoloc.latitude if geoloc is not None else 0.0)
        for (sellloc, geoloc) in sellloc_geoloc.items()}
    sellloc_lon = {
        sellloc: (geoloc.longitude if geoloc is not None else 0.0)
        for (sellloc, geoloc) in sellloc_geoloc.items()}
    df['SellingLocation_lat'] = df['SellingLocation'].map(sellloc_lat)
    df['SellingLocation_lon'] = df['SellingLocation'].map(sellloc_lon)
    # # TODO: experiment with one-hot encoding (problems is that it doesn't scale)
    # df = pd.merge(
    #     left=df,
    #     right=pd.get_dummies(df['SellingLocation'], prefix='SellingLocation'),
    #     how='inner',
    #     left_index=True,
    #     right_index=True)
    ########################################
    # JDPowersCat: One-hot encoding
    # TODO: Estimate sizes from Wikipedia, e.g. https://en.wikipedia.org/wiki/Vehicle_size_class.
    logger.info("JDPowersCat: One-hot encoding.")
    # Cast to string, replacing 'nan' with 'UNKNOWN'.
    df['JDPowersCat'] = (df['JDPowersCat'].astype(str)).str.replace(' ', '').apply(
        lambda cat: 'UNKNOWN' if cat == 'nan' else cat)
    # One-hot encoding.
    df = pd.merge(
        left=df,
        right=pd.get_dummies(df['JDPowersCat'], prefix='JDPowersCat'),
        left_index=True,
        right_index=True)
    ########################################
    # LIGHT_N0G1Y2R3
    # Rank lights by warning level.
    logger.info("LIGHT_N0G1Y2R3: Rank lights by warning level (null=0, green=1, yellow=2, red=3).")
    df['LIGHT_N0G1Y2R3'] = df['LIGHTG']*1 + df['LIGHTY']*2 + df['LIGHTR']*3
    ########################################
    # SaleDate_*: Extract timeseries features.
    logger.info("SaleDate: Extract timeseries features.")
    df['SaleDate_dow'] = df['SaleDate'].dt.dayofweek
    df['SaleDate_doy'] = df['SaleDate'].dt.dayofyear
    df['SaleDate_day'] = df['SaleDate'].dt.day
    df['SaleDate_decyear'] = df['SaleDate'].dt.year + (df['SaleDate'].dt.dayofyear-1)/366
    ########################################
    # BuyerID, SellerID, VIN, SellingLocation, CarMake, JDPowersCat:
    # Make cumulative informative priors (*_num*, *_frac*) for string features.
    logger.info(textwrap.dedent("""\
        BuyerID, SellerID, VIN, SellingLocation, CarMake, JDPowersCat:
        Make cumulative informative priors (*_num*, *_frac*) for string features."""))
    # Cumulative features require sorting by time.
    df.sort_values(by=['SaleDate'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    for col in ['BuyerID', 'SellerID', 'VIN', 'SellingLocation', 'CarMake', 'JDPowersCat']:
        logger.info("Processing {col}".format(col=col))
        ####################
        # Cumulative count of transactions and DSEligible:
        # Cumulative count of transactions (yes including current).
        df[col+'_numTransactions'] = df[[col]].groupby(by=col).cumcount().astype(int) + 1
        df[col+'_numTransactions'].fillna(value=1, inplace=True)
        # Cumulative count of transations that were DealShield-eligible (yes including current).
        df[col+'_numDSEligible1'] = df[[col, 'DSEligible']].groupby(by=col)['DSEligible'].cumsum().astype(int)
        df[col+'_numDSEligible1'].fillna(value=0, inplace=True)
        # Cumulative ratio of transactions that were DealShield-eligible (0=bad, 1=good).
        df[col+'_fracDSEligible1DivTransactions'] = (df[col+'_numDSEligible1']/df[col+'_numTransactions'])
        df[col+'_fracDSEligible1DivTransactions'].fillna(value=1, inplace=True)
        ####################
        # DSEligible and Returned
        # Note:
        # * DealShield-purchased ==> Returned != -1 (not null)
        # * below requires
        #     DSEligible == 0 ==> Returned == -1 (is null)
        #     Returned != -1 (not null) ==> DSEligible == 1
        assert (df.loc[df['DSEligible']==0, 'Returned'] == -1).all()
        assert (df.loc[df['Returned']!=-1, 'DSEligible'] == 1).all()
        # Cumulative count of transactions that were DealShield-eligible and DealShield-purchased.
        df_tmp = df[[col, 'Returned']].copy()
        df_tmp['ReturnedNotNull'] = df_tmp['Returned'] != -1
        df[col+'_numReturnedNotNull'] = df_tmp[[col, 'ReturnedNotNull']].groupby(by=col)['ReturnedNotNull'].cumsum().astype(int)
        df[col+'_numReturnedNotNull'].fillna(value=0, inplace=True)
        del df_tmp
        # Cumulative ratio of DealShield-eligible transactions that were DealShield-purchased (0=mode).
        df[col+'_fracReturnedNotNullDivDSEligible1'] = df[col+'_numReturnedNotNull']/df[col+'_numDSEligible1']
        df[col+'_fracReturnedNotNullDivDSEligible1'].fillna(value=0, inplace=True)
        # Cumulative count of transactions that were DealShield-elegible and DealShield-purchased and DealShield-returned.
        df_tmp = df[[col, 'Returned']].copy()
        df_tmp['Returned1'] = df_tmp['Returned'] == 1
        df[col+'_numReturned1'] = df_tmp[[col, 'Returned1']].groupby(by=col)['Returned1'].cumsum().astype(int)
        df[col+'_numReturned1'].fillna(value=0, inplace=True)
        del df_tmp
        # Cumulative ratio of DealShield-eligible, DealShield-purchased transactions that were DealShield-returned (0=good, 1=bad).
        # Note: BuyerID_fracReturned1DivReturnedNotNull is the cumulative return rate for a buyer.
        df[col+'_fracReturned1DivReturnedNotNull'] = df[col+'_numReturned1']/df[col+'_numReturnedNotNull']
        df[col+'_fracReturned1DivReturnedNotNull'].fillna(value=0, inplace=True)
        # Check that weighted average of return rate equals overall return rate.
        # Note: Requires groups sorted by date, ascending.
        assert np.isclose(
            (df[[col, col+'_fracReturned1DivReturnedNotNull', col+'_numReturnedNotNull']].groupby(by=col).last().product(axis=1).sum()/\
             df[[col, col+'_numReturnedNotNull']].groupby(by=col).last().sum()).values[0],
            sum(df['Returned']==1)/sum(df['Returned'] != -1),
            equal_nan=True)
        ####################
        # DSEligible and Returned_asm
        # NOTE:
        # * Below requires
        #     DSEligible == 0 ==> Returned_asm == 1
        #     Returned_asm == 0 ==> DSEligible == 1
        assert (df.loc[df['DSEligible']==0, 'Returned_asm'] == 1).all()
        assert (df.loc[df['Returned_asm']==0, 'DSEligible'] == 1).all()
        # Cumulative number of transactions that were assumed to be returned.
        df_tmp = df[[col, 'Returned_asm']].copy()
        df_tmp['Returnedasm1'] = df_tmp['Returned_asm'] == 1
        df[col+'_numReturnedasm1'] = df_tmp[[col, 'Returnedasm1']].groupby(by=col)['Returnedasm1'].cumsum().astype(int)
        df[col+'_numReturnedasm1'].fillna(value=0, inplace=True)
        del df_tmp
        # Cumulative ratio of transactions that were assumed to be returned (0=mode).
        df[col+'_fracReturnedasm1DivTransactions'] = df[col+'_numReturnedasm1']/df[col+'_numTransactions']
        df[col+'_fracReturnedasm1DivTransactions'].fillna(value=0, inplace=True)
        # Check that weighted average of assumed return rate equals overall assumed return rate.
        assert np.isclose(
            (df[[col, col+'_fracReturnedasm1DivTransactions', col+'_numTransactions']].groupby(by=col).last().product(axis=1).sum()/\
             df[[col, col+'_numTransactions']].groupby(by=col).last().sum()).values[0],
            sum(df['Returned_asm']==1)/sum(df['Returned_asm'] != -1),
            equal_nan=True)
        # Note:
        #   * Number of transactions that were DealShield-eligible and assumed to be returned ==
        #     number of transactions that were DealShield-elegible and DealShield-purchased and DealShield-returned
        #     (numReturned1)
    return df


def plot_eda(
    df:pd.DataFrame,
    columns:list,
    path_plot_dir:str=None
    ) -> None:
    r"""Make plots for exploratory data analysis (EDA).

    Args:
        df (pandas.DataFrame): Dataframe of formatted data.
        columns (list): List of strings of columns in `df` to plot.
        path_plot_dir (str, optional, None): Path to directory in which to save plots.

    Returns:
        None

    """
    # Check inputs.
    if not os.path.exists(path_plot_dir):
        raise IOError(textwrap.dedent("""\
            Path does not exist: path_plot_dir =
            {path}""".format(path=path_plot_dir)))
    ################################################################################
    # Plot frequency distributions.
    print('#'*80)
    print('Plot frequency distributions (histograms) of columns.')
    for col in columns:
        print('#'*40)
        print('Feature: {col}'.format(col=col))
        print('Timestamp:', time.strftime(r'%Y-%m-%dT%H:%M:%S%Z', time.gmtime()))
        # Plot frequency distributions by transaction.
        df_plot = df[['BuyerID', col, buyer_retrate]].copy()
        buyer_retrate_omax = buyer_retrate+'_omax'
        df_plot[buyer_retrate_omax] = df_plot[buyer_retrate] > buyer_retrate_max
        itemized_counts = {
            is_omax: grp[col].values
            for (is_omax, grp) in df_plot.groupby(by=buyer_retrate_omax)}
        itemized_counts = collections.OrderedDict(
            sorted(itemized_counts.items(), key=lambda tup: tup[0], reverse=False))
        keys = itemized_counts.keys()
        bins = 50
        colors = sns.light_palette(sns.color_palette()[2], n_colors=len(keys))
        plt.hist(
            [itemized_counts[key] for key in itemized_counts.keys()],
            bins=bins, stacked=True, rwidth=1.0, label=keys, color=colors)
        plt.title('{col}\nfrequency distribution'.format(col=col))
        plt.xlabel(col)
        plt.ylabel('Number of transactions with\n{col} = X'.format(col=col))
        plt.legend(
            title='Buyer return\nrate > {retrate:.0%}'.format(retrate=buyer_retrate_max),
            loc='upper left', bbox_to_anchor=(1.0, 1.0))
        rect = (0, 0, 0.85, 1)
        plt.tight_layout(rect=rect)
        if path_plot_dir is not None:
            plt.savefig(
                os.path.join(path_plot_dir, 'freq-dist-transaction_'+col+'.png'),
                dpi=300)
        plt.show()
        # Plot frequency distributions by buyer.
        itemized_counts = {
            is_omax: grp[['BuyerID', col]].groupby(by='BuyerID').mean().values.flatten()
            for (is_omax, grp) in df_plot.groupby(by=buyer_retrate_omax)}
        itemized_counts = collections.OrderedDict(
            sorted(itemized_counts.items(), key=lambda tup: tup[0], reverse=False))
        keys = itemized_counts.keys()
        plt.hist(
            [itemized_counts[key] for key in itemized_counts.keys()],
            bins=bins, stacked=True, rwidth=1.0, label=keys, color=colors)
        plt.title('Mean {col} per buyer\nfrequency distribution'.format(col=col))
        plt.xlabel('Mean '+col)
        plt.ylabel('Number of buyers with\nmean {col} = X'.format(col=col))
        plt.legend(
            title='Buyer return\nrate > {retrate:.0%}'.format(retrate=buyer_retrate_max),
            loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.tight_layout(rect=rect)
        if path_plot_dir is not None:
            plt.savefig(
                os.path.join(path_plot_dir, 'freq-dist-buyer_'+col+'.png'),
                dpi=300)
        plt.show()

    ################################################################################
    # Plot (timeseries) traces for fractional quantities vs fraction of completed transactions.
    # Columns to plot: catgory (cat), <category>_numTransactions (trans), <category>_frac* (col)
    print('#'*80)
    print('Plot traces (timeseries) for fractional quantities vs fraction of completed transactions.')
    plot_cols = list()
    for col in df.columns:
        if '_frac' in col:
            cat = col.split('_frac')[0]
            trans = cat+'_numTransactions'
            plot_cols.append([cat, trans, col])
    for (col_cat, col_trans, col_frac) in plot_cols:
        print('#'*40)
        print('Category column:    {col}'.format(col=col_cat))
        print('Transaction column: {col}'.format(col=col_trans))
        print('Fraction column:    {col}'.format(col=col_frac))
        print('Timestamp:', time.strftime(r'%Y-%m-%dT%H:%M:%S%Z', time.gmtime()))
        # Weight categorical values by number of transactions.       
        assert (df[[col_cat, col_trans]].groupby(by=col_cat).last().sum() == len(df)).all()
        cat_wts = df[[col_cat, col_trans]].groupby(by=col_cat).last()/len(df)
        cat_wts.columns = [col_cat+'_wts']
        cats = cat_wts.sample(n=30, replace=True, weights=col_cat+'_wts').index.values
        # Make plot.
        for idx in range(len(cats)):
            cat = cats[idx]
            tfmask = df[col_cat] == cat
            xvals = (df.loc[tfmask, col_trans]/sum(tfmask)).values
            yvals = df.loc[tfmask, col_frac].values
            xvals_omax = (df.loc[np.logical_and(tfmask, df[buyer_retrate] > buyer_retrate_max), col_trans]/sum(tfmask)).values
            yvals_omax = df.loc[np.logical_and(tfmask, df[buyer_retrate] > buyer_retrate_max), col_frac].values
            if len(xvals) > 51: # downsample for speed
                step = 1/50
                xvals_resampled = np.arange(start=0, stop=1+step, step=step)
                yvals_resampled = np.interp(x=xvals_resampled, xp=xvals, fp=yvals)
                (xvals, yvals) = (xvals_resampled, yvals_resampled)
            if len(xvals_omax) > 51: # downsample for speed
                idxs_omax = np.random.choice(range(len(xvals_omax)), size=51, replace=False)
                xvals_omax_resampled = xvals_omax[idxs_omax]
                yvals_omax_resampled = yvals_omax[idxs_omax]
                (xvals_omax, yvals_omax) = (xvals_omax_resampled, yvals_omax_resampled)
            plt.plot(
                xvals, yvals,
                marker='.', alpha=0.1, color=sns.color_palette()[0])
            if idx == 0:
                label = 'Buyer return\nrate > {retrate:.0%}'.format(retrate=buyer_retrate_max)
            else:
                label = None
            plt.plot(
                xvals_omax, yvals_omax,
                marker='o', alpha=0.2, linestyle='',
                color=sns.color_palette()[2], label=label)
        plt.title('{col_frac} vs\nfraction of transactions completed'.format(col_frac=col_frac))
        plt.xlabel("Fraction of transactions completed")
        plt.ylabel(col_frac)
        plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        rect = (0, 0, 0.80, 1)
        plt.tight_layout(rect=rect)
        if path_plot_dir is not None:
            plt.savefig(
                os.path.join(path_plot_dir, 'trace_'+col_frac+'.png'),
                dpi=300)
        plt.show()
    return None


def plot_heuristic(
    df:pd.DataFrame,
    path_plot_dir:str=None
    ) -> None:
    r"""Plot heuristic to predict bad dealers.

    Args:
        df (pandas.DataFrame): DataFrame of formatted data.
        path_plot_dir (str, optional, None): Path to directory in which to save plots.

    Returns:
        None

    TODO:
        * Use stacked area chart instead of histogram, but requires aggregating by date.
        * Format xaxis with dates.
          2013.0 = 2013-01-01
          2013.2 = 2013-03-14
          2013.4 = 2013-05-26
          2013.6 = 2013-08-07
          2013.8 = 2013-10-19

    """
    # Check inputs.
    if not os.path.exists(path_plot_dir):
        raise IOError(textwrap.dedent("""\
            Path does not exist: path_plot_dir =
            {path}""".format(path=path_plot_dir)))

    # Plot timeseries histogram of Returned vs SalesDate.
    # Bins represent weeks.
    df_plot = df[['SaleDate_decyear', 'Returned']].copy()
    itemized_counts = {
        ret: collections.Counter(grp['SaleDate_decyear'])
        for (ret, grp) in df_plot.groupby(by='Returned')}
    itemized_counts = collections.OrderedDict(
        sorted(itemized_counts.items(), key=lambda tup: tup[0], reverse=True))
    keys = itemized_counts.keys()
    bins = int(np.ceil((df_plot['SaleDate_decyear'].max() - df_plot['SaleDate_decyear'].min())*52))
    colors = sns.color_palette(n_colors=len(keys))[::-1]
    plt.hist(
        [list(itemized_counts[key].elements()) for key in itemized_counts.keys()],
        bins=bins, stacked=True, rwidth=1.0, label=keys, color=colors)
    plt.xlim(xmin=int(df_plot['SaleDate_decyear'].min()))
    xlim = plt.xlim()
    plt.title('Returned vs SaleDate\nby Returned status')
    plt.xlabel('SaleDate (decimal year)')
    plt.ylabel('Number of transactions with\nReturned = <status>')
    plt.legend(title='Returned\nstatus', loc='upper left', bbox_to_anchor=(1.0, 1.0))
    rect = (0, 0, 0.85, 1)
    plt.tight_layout(rect=rect)
    if path_plot_dir is not None:
        plt.savefig(
            os.path.join(path_plot_dir, 'heuristic0_returned101_vs_saledate_by_status.png'),
            dpi=300)
    plt.show()

    # Plot timeseries histogram of Returned (0,1) vs SalesDate.
    # Bins represent weeks.
    df_plot = df.loc[df['Returned']!=-1, ['SaleDate_decyear', 'Returned']].copy()
    itemized_counts = {
        ret: collections.Counter(grp['SaleDate_decyear'])
        for (ret, grp) in df_plot.groupby(by='Returned')}
    itemized_counts = collections.OrderedDict(
        sorted(itemized_counts.items(), key=lambda tup: tup[0], reverse=True))
    keys = itemized_counts.keys()
    bins = int(np.ceil((df_plot['SaleDate_decyear'].max() - df_plot['SaleDate_decyear'].min())*52))
    plt.hist(
        [list(itemized_counts[key].elements()) for key in itemized_counts.keys()],
        bins=bins, stacked=True, rwidth=1.0, label=keys, color=colors[:2])
    plt.xlim(xlim)
    plt.title('Returned vs SaleDate\nby Returned status')
    plt.xlabel('SaleDate (decimal year)')
    plt.ylabel('Number of transactions with\nReturned = <status>')
    plt.legend(title='Returned\nstatus', loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout(rect=rect)
    if path_plot_dir is not None:
        plt.savefig(
            os.path.join(path_plot_dir, 'heuristic1_returned01_vs_saledate_by_status.png'),
            dpi=300)
    plt.show()

    # # ARCHIVED: Use return rate as heuristic rather than return count.
    # # Plot timeseries histogram of Returned (1) vs SalesDate by BuyerID.
    # df_plot = df.loc[df['Returned']==1, ['SaleDate_decyear', 'BuyerID']].copy()
    # top = [tup[0] for tup in collections.Counter(df_plot['BuyerID']).most_common(n=20)]
    # itemized_counts_all = {
    #     buy: collections.Counter(grp['SaleDate_decyear'])
    #     for (buy, grp) in df_plot.groupby(by='BuyerID')}
    # itemized_counts_top = {'other': collections.Counter()}
    # for (buyerid, counts) in itemized_counts_all.items():
    #     if buyerid in top:
    #         itemized_counts_top[buyerid] = counts
    #     else:
    #         itemized_counts_top['other'].update(counts)
    # itemized_counts = collections.OrderedDict(
    #     sorted(itemized_counts_top.items(), key=lambda tup: sum(tup[1].values()), reverse=True))
    # itemized_counts.move_to_end('other')
    # keys = itemized_counts.keys()
    # bins = int(np.ceil((df_plot['SaleDate_decyear'].max() - df_plot['SaleDate_decyear'].min())*52))
    # colors = sns.light_palette(sns.color_palette()[2], n_colors=len(keys))
    # plt.hist(
    #     [list(itemized_counts[key].elements()) for key in itemized_counts.keys()],
    #     bins=bins, stacked=True, rwidth=1.0, label=keys, color=colors)
    # plt.title('Returned vs SaleDate by BuyerID')
    # plt.xlabel('SaleDate (decimal year)')
    # plt.ylabel('Returned (status=1)')
    # plt.legend(title='BuyerID', loc='upper left', bbox_to_anchor=(1.0, 1.0))
    # plt.show()

    # Plot timeseries histogram of Returned (1) vs SalesDate
    # by BuyerID for BuyerIDs with return rate > buyer_retrate_max (buyer_retrate_max=0.1).
    # buyer_retrate = 'BuyerID_fracReturned1DivReturnedNotNull'
    # Bins represent weeks.
    df_plot = df.loc[df['Returned']==1, ['SaleDate_decyear', 'BuyerID', buyer_retrate]].copy()
    buyer_retrate_omax = buyer_retrate+'_omax'
    df_plot[buyer_retrate_omax] = df_plot[buyer_retrate] > buyer_retrate_max
    itemized_counts = {
        is_omax: collections.Counter(grp['SaleDate_decyear'])
        for (is_omax, grp) in df_plot.groupby(by=buyer_retrate_omax)}
    itemized_counts = collections.OrderedDict(
        sorted(itemized_counts.items(), key=lambda tup: tup[0], reverse=False))
    keys = itemized_counts.keys()
    bins = int(np.ceil((df_plot['SaleDate_decyear'].max() - df_plot['SaleDate_decyear'].min())*52))
    colors = sns.light_palette(sns.color_palette()[2], n_colors=len(keys))
    plt.hist(
        [list(itemized_counts[key].elements()) for key in itemized_counts.keys()],
        bins=bins, stacked=True, rwidth=1.0, label=keys, color=colors)
    plt.xlim(xlim)
    plt.title('Returned vs SaleDate\nby buyer return rate')
    plt.xlabel('SaleDate (decimal year)')
    plt.ylabel('Number of transactions with Returned = 1\nand buyer return rate = <rate>')
    plt.legend(
        title='Buyer return\nrate > {retrate:.0%}'.format(retrate=buyer_retrate_max),
        loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout(rect=rect)
    if path_plot_dir is not None:
        plt.savefig(
            os.path.join(path_plot_dir, 'heuristic2_returned1_vs_saledate_by_returnrate.png'),
            dpi=300)
    plt.show()

    # Plot frequency distribution of return rates per BuyerID
    df_plot = df[['BuyerID', buyer_retrate]].copy()
    df_plot[buyer_retrate_omax] = df_plot[buyer_retrate] > buyer_retrate_max
    itemized_counts = {
        is_omax: grp[buyer_retrate].values
        for (is_omax, grp) in df_plot.groupby(by=buyer_retrate_omax)}
    itemized_counts = collections.OrderedDict(
        sorted(itemized_counts.items(), key=lambda tup: tup[0], reverse=False))
    keys = itemized_counts.keys()
    bins = 20
    colors = sns.light_palette(sns.color_palette()[2], n_colors=len(keys))
    plt.hist(
        [itemized_counts[key] for key in itemized_counts.keys()],
        bins=bins, stacked=True, rwidth=1.0, label=keys, color=colors)
    plt.title('Return rate per transaction\nfrequency distribution')
    plt.xlabel('Return rate')
    plt.ylabel('Number of transactions with\nbuyer return rate = X')
    plt.legend(
        title='Buyer return\nrate > {retrate:.0%}'.format(retrate=buyer_retrate_max),
        loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout(rect=rect)
    if path_plot_dir is not None:
        plt.savefig(
            os.path.join(path_plot_dir, 'heuristic3_returnrate_freq-dist-transaction_by_returnrate.png'),
            dpi=300)
    plt.show()

    # Plot frequency distribution of return rates per BuyerID
    # Note: Buyers can be counted twice in the histogram if they cross the
    #   buyer_retrate_max = 0.1 threshold.
    df_plot = df[['BuyerID', buyer_retrate]].copy()
    df_plot[buyer_retrate_omax] = df_plot[buyer_retrate] > buyer_retrate_max
    itemized_counts = {
        is_omax: grp[['BuyerID', buyer_retrate]].groupby(by='BuyerID').mean().values.flatten()
        for (is_omax, grp) in df_plot.groupby(by=buyer_retrate_omax)}
    itemized_counts = collections.OrderedDict(
        sorted(itemized_counts.items(), key=lambda tup: tup[0], reverse=False))
    keys = itemized_counts.keys()
    bins = 20
    colors = sns.light_palette(sns.color_palette()[2], n_colors=len(keys))
    plt.hist(
        [itemized_counts[key] for key in itemized_counts.keys()],
        bins=bins, stacked=True, rwidth=1.0, label=keys, color=colors)
    plt.title('Return rates per buyer\nfrequency distribution')
    plt.xlabel('Return rate')
    plt.ylabel('Number of buyers with\nreturn rate = X')
    plt.legend(
        title='Buyer return\nrate > {retrate:.0%}'.format(retrate=buyer_retrate_max),
        loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout(rect=rect)
    if path_plot_dir is not None:
        plt.savefig(
            os.path.join(path_plot_dir, 'heuristic4_returnrate_freq-dist-buyer_by_returnrate.png'),
            dpi=300)
    plt.show()
    return None

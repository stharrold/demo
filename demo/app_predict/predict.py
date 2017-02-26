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
__all__ = ['eta']


# Define state settings and globals.
# Note: For non-root-level loggers, use `getLogger(__name__)`
#     http://stackoverflow.com/questions/17336680/python-logging-with-multiple-modules-does-not-work
logger = logging.getLogger(__name__)
# Set the matplotlib backend to the Anti-Grain Geometry C++ library.
# Note: Use plt.switch_backend since matplotlib.use('agg') before importing pyplot fails.
plt.switch_backend('agg')
# Set matplotlib styles with seaborn
sns.set()


def eta(
    df:pd.DataFrame,
    path_data_dir:str
    ) -> pd.DataFrame:
    r"""Extract-transform-load.

    Args:
        df (pandas.DataFrame): Dataframe of raw data.
        path_data_dir (str): Path to data directory for caching geocode shelf file.

    Returns:
        df (pandas.DataFrame): Dataframe of extracted data.

    Notes:
        * BuyerID_fracReturned1DivReturnedNotNull is the return rate for a buyer.
    TODO:
        * Modularize script into separate helper functions.
        # Replace `print` with `logger.info`
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
    # Fix DSEligible == 0 but Returned not null
    # Some vehicles have DSEligible=0 but have Returned!=nan due to errors or extenuating circumstances.
    # To correct: If Returned!=nan, then DSEligible=1
    print('#'*80)
    print("DSEligible, Returned: Fix DSEligible == 0 but Returned not null.")
    print("To correct: If Returned not null, then DSEligible = 1.")
    print()
    print("Before:\n{pt}".format(
        pt=pd.pivot_table(
            df[['DSEligible', 'Returned']].astype(str),
            index='DSEligible', columns='Returned',
            aggfunc=len, margins=True, dropna=False)))
    print()
    df.loc[df['Returned'].notnull(), 'DSEligible'] = 1
    print("After:\n{pt}".format(
        pt=pd.pivot_table(
            df[['DSEligible', 'Returned']].astype(str),
            index='DSEligible', columns='Returned',
            aggfunc=len, margins=True, dropna=False)))
    print()

    ########################################
    # Returned
    # NOTE: THIS TRANSFORMATION MUST BE BEFORE INFORMATIVE PRIORS SINCE Returned.isnull() -> -1
    # Fill null values with -1 and cast to int.
    print('#'*80)
    print('Returned: Fill nulls with -1 and cast to int.')
    print()
    print("Before:\n{pt}".format(
        pt=pd.pivot_table(
            df[['DSEligible', 'Returned']].astype(str),
            index='DSEligible', columns='Returned',
            aggfunc=len, margins=True, dropna=False)))
    print()
    df['Returned'] = df[['Returned']].fillna(value=-1).astype(int)
    print("After:\n{pt}".format(
        pt=pd.pivot_table(
            df[['DSEligible', 'Returned']].astype(str),
            index='DSEligible', columns='Returned',
            aggfunc=len, margins=True, dropna=False)))
    print()

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
    print('#'*80)
    print(textwrap.dedent("""\
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
    print()
    print("Relationship between DSEligible and Returned:\n{pt}".format(
        pt=pd.pivot_table(
            df[['DSEligible', 'Returned']].astype(str),
            index='DSEligible', columns='Returned',
            aggfunc=len, margins=True, dropna=False)))
    print()
    print("Relationship between DSEligible and Returned_asm:\n{pt}".format(
        pt=pd.pivot_table(
            df[['DSEligible', 'Returned_asm']].astype(str),
            index='DSEligible', columns='Returned_asm',
            aggfunc=len, margins=True, dropna=False)))
    print()
    print("Relationship between Returned and Returned_asm:\n{pt}".format(
        pt=pd.pivot_table(
            df[['Returned', 'Returned_asm']].astype(str),
            index='Returned', columns='Returned_asm',
            aggfunc=len, margins=True, dropna=False)))
    print()

    ########################################
    # Geocode SellingLocation
    # Cell takes ~1 min to execute if shelf does not exist.
    # Google API limit: https://developers.google.com/maps/documentation/geocoding/usage-limits
    print('#'*80)
    print("SellingLocation: Geocode.")
    print("Scraping webpages for addresses and looking up latitude, longitude coordinates.")
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
    print("Mapping SellingLocation to latitude, longitude coordinates.")
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
    print()

    ########################################
    # Deduplicate CarMake
    # TODO: Find/scrape hierarchical relationships between car brands
    #     (e.g. https://en.wikipedia.org/wiki/Category:Mid-size_cars). To business people: would that be helpful?
    # TODO: Deduplicate with spelling corrector.
    print('#'*80)
    print("CarMake: Deduplicate.")
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
    df['CarMake'] = (df['CarMake'].astype(str)).str.replace(' ', '').apply(
        lambda car: carmake_dedup[car] if car in carmake_dedup else car)
    # # TODO: Experiment with one-hot encoding (problem is that it doesn't scale)
    # df = pd.merge(
    #     left=df,
    #     right=pd.get_dummies(df['CarMake'], prefix='CarMake'),
    #     left_index=True,
    #     right_index=True)
    print()

    ########################################
    # JDPowersCat
    # TODO: Estimate sizes from Wikipedia, e.g. https://en.wikipedia.org/wiki/Vehicle_size_class.
    print('#'*80)
    print("JDPowersCat: One-hot encoding.")
    # Cast to string, replacing 'nan' with 'UNKNOWN'.
    df['JDPowersCat'] = (df['JDPowersCat'].astype(str)).str.replace(' ', '').apply(
        lambda cat: 'UNKNOWN' if cat == 'nan' else cat)
    # One-hot encoding.
    df = pd.merge(
        left=df,
        right=pd.get_dummies(df['JDPowersCat'], prefix='JDPowersCat'),
        left_index=True,
        right_index=True)
    print()

    ########################################
    # LIGHTG, LIGHTY, LIGHTR, LIGHT_G1Y2R3
    # Fix all transactions so that only light with highest warning is retained.
    print('#'*80)
    print("LIGHT*: Only retain light with highest warning.")
    print()
    pt = pd.DataFrame([
        df.loc[df['LIGHTG']==1, ['LIGHTG', 'LIGHTY', 'LIGHTR']].sum(),
        df.loc[df['LIGHTY']==1, ['LIGHTG', 'LIGHTY', 'LIGHTR']].sum(),
        df.loc[df['LIGHTR']==1, ['LIGHTG', 'LIGHTY', 'LIGHTR']].sum()],
        index=['LIGHTG=1', 'LIGHTY=1', 'LIGHTR=1'])
    pt.columns = ['LIGHTG=1', 'LIGHTY=1', 'LIGHTR=1']
    print("Before:\n{pt}".format(pt=pt))
    print()
    df.loc[df['LIGHTR']==1, ['LIGHTG', 'LIGHTY']] = 0
    df.loc[df['LIGHTY']==1, ['LIGHTG']] = 0
    pt = pd.DataFrame([
        df.loc[df['LIGHTG']==1, ['LIGHTG', 'LIGHTY', 'LIGHTR']].sum(),
        df.loc[df['LIGHTY']==1, ['LIGHTG', 'LIGHTY', 'LIGHTR']].sum(),
        df.loc[df['LIGHTR']==1, ['LIGHTG', 'LIGHTY', 'LIGHTR']].sum()],
        index=['LIGHTG=1', 'LIGHTY=1', 'LIGHTR=1'])
    pt.columns = ['LIGHTG=1', 'LIGHTY=1', 'LIGHTR=1']
    print("After:\n{pt}".format(pt=pt))
    print()
    # Rank lights by warning level: null=0, green=1, yellow=2, red=3
    print("LIGHT_N0G1Y2R3: Rank lights by warning level (null=0, green=1, yellow=2, red=3).")
    df['LIGHT_N0G1Y2R3'] = df['LIGHTG']*1 + df['LIGHTY']*2 + df['LIGHTR']*3
    print()

    ########################################
    # SaleDate
    # Extract timeseries features for SaleDate
    print('#'*80)
    print("SaleDate: Extract timeseries features.")
    df['SaleDate'] = pd.to_datetime(df['SaleDate'], format=r'%y-%m-%d')
    df['SaleDate_dow'] = df['SaleDate'].dt.dayofweek
    df['SaleDate_doy'] = df['SaleDate'].dt.dayofyear
    df['SaleDate_day'] = df['SaleDate'].dt.day
    df['SaleDate_decyear'] = df['SaleDate'].dt.year + (df['SaleDate'].dt.dayofyear-1)/366
    print()

    ########################################
    # Autocheck_score
    # TODO: Use nearest neighbors to infer probable fill value.
    # Fill null values with mode (1.0).
    print('#'*80)
    print("Autocheck_score: Fill null values with mode (1).")
    df['Autocheck_score'] = df['Autocheck_score'].fillna(value=1)
    print()

    ########################################
    # ConditionReport
    # Map character codes to numerical values, invalid codes are "average".
    print('#'*80)
    print("ConditionReport: Map character codes to numerical values. Invalid codes are 'average'.")
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
    print()

    ########################################
    # BuyerID, SellerID, VIN, SellingLocation, CarMake, JDPowersCat:
    # Make informative priors (*_num*, *_frac*) for string features.
    # TODO: Also make priors using Returned_asm
    print('#'*80)
    print(textwrap.dedent("""\
        BuyerID, SellerID, VIN, SellingLocation, CarMake, JDPowersCat:
        Make informative priors (*_num*, *_frac*) for string features."""))
    for col in ['BuyerID', 'SellerID', 'VIN', 'SellingLocation', 'CarMake', 'JDPowersCat']:
        print("Processing {col}".format(col=col))
        ####################
        # Number of transactions and DSEligible:
        # Cast to string.
        df[col] = df[col].astype(str)
        # Number of transactions.
        df[col+'_numTransactions'] = df[col].map(
            collections.Counter(df[col].values)).fillna(value=0)
        # Number of transations that were DealShield-eligible
        tfmask = df['DSEligible'] == 1
        df[col+'_numDSEligible1'] = df[col].map(
            collections.Counter(df.loc[tfmask, col].values)).fillna(value=0)
        # Fraction of transactions that were DealShield-eligible (0=bad, 1=good)
        df[col+'_fracDSEligible1DivTransactions'] = \
            (df[col+'_numDSEligible1']/df[col+'_numTransactions']).fillna(value=1)
        ####################
        # DSEligible and Returned
        # Note:
        # * DealShield-purchased ==> Returned != -1 (not null)
        # * below requires
        #     DSEligible == 0 ==> Returned == -1 (is null)
        #     Returned != -1 (not null) ==> DSEligible == 1
        assert (df.loc[df['DSEligible']==0, 'Returned'] == -1).all()
        assert (df.loc[df['Returned']!=-1, 'DSEligible'] == 1).all()
        # Number of transactions that were DealShield-eligible and DealShield-purchased
        tfmask = df['Returned'] != -1
        df[col+'_numReturnedNotNull'] = df[col].map(
            collections.Counter(df.loc[tfmask, col].values)).fillna(value=0)
        # Fraction of DealShield-eligible transactions that were DealShield-purchased (0=mode)
        df[col+'_fracReturnedNotNullDivDSEligible1'] = \
            (df[col+'_numReturnedNotNull']/df[col+'_numDSEligible1']).fillna(value=0)
        # Number of transactions that were DealShield-elegible and DealShield-purchased and DealShield-returned
        tfmask = df['Returned'] == 1
        df[col+'_numReturned1'] = df[col].map(
            collections.Counter(df.loc[tfmask, col].values)).fillna(value=0)
        # Fraction of DealShield-eligible, DealShield-purchased transactions that were DealShield-returned (0=good, 1=bad)
        # Note: BuyerID_fracReturned1DivReturnedNotNull is the return rate for a buyer.
        df[col+'_fracReturned1DivReturnedNotNull'] = \
            (df[col+'_numReturned1']/df[col+'_numReturnedNotNull']).fillna(value=0)
        # Check that weighted average of return rate equals overall return rate.
        assert np.isclose(
            (df[[col, col+'_fracReturned1DivReturnedNotNull', col+'_numReturnedNotNull']].groupby(by=col).mean().product(axis=1).sum()/\
             df[[col, col+'_numReturnedNotNull']].groupby(by=col).mean().sum()).values[0],
            sum(df['Returned']==1)/sum(df['Returned'] != -1))
        ####################
        # DSEligible and Returned_asm
        # NOTE:
        # * Below requires
        #     DSEligible == 0 ==> Returned_asm == 1
        #     Returned_asm == 0 ==> DSEligible == 1
        assert (df.loc[df['DSEligible']==0, 'Returned_asm'] == 1).all()
        assert (df.loc[df['Returned_asm']==0, 'DSEligible'] == 1).all()
        # Number of transactions that were assumed to be returned.
        tfmask = df['Returned_asm'] == 1
        df[col+'_numReturnedasm1'] = df[col].map(
            collections.Counter(df.loc[tfmask, col].values)).fillna(value=0)
        # Fraction of transactions that were assumed to be returned (0=mode)
        df[col+'_fracReturnedasm1DivTransactions'] = \
            (df[col+'_numReturnedasm1']/df[col+'_numTransactions']).fillna(value=0)
        # Check that weighted average of assumed return rate equals overall assumed return rate.
        assert np.isclose(
            (df[[col, col+'_fracReturnedasm1DivTransactions', col+'_numTransactions']].groupby(by=col).mean().product(axis=1).sum()/\
             df[[col, col+'_numTransactions']].groupby(by=col).mean().sum()).values[0],
            sum(df['Returned_asm']==1)/sum(df['Returned_asm'] != -1))
        # Note:
        #   * Number of transactions that were DealShield-eligible and assumed to be returned ==
        #     number of transactions that were DealShield-elegible and DealShield-purchased and DealShield-returned
        #     (numReturned1)
    print()
    return df


def plot_freq_dists(
    df:pd.DataFrame,
    ftrs:list
    ) -> None:
    r"""Plot frequency distributions of features.

    Args:
        df (pandas.DataFrame): Dataframe of formatted data.
        ftrs (list): List of strings of features (columns) in `df` to plot.

    Returns:
        None
    """
    # Plot frequency distributions.
    for ftr in ftrs:
        print('#'*80)
        print('Feature: {ftr}'.format(ftr=ftr))
        # ...by transaction
        sns.distplot(df[ftr].values, hist=True, kde=False, norm_hist=False)
        plt.title('{ftr}\nfrequency distribution'.format(ftr=ftr))
        plt.xlabel(ftr)
        plt.ylabel('Number of transactions with\n{ftr}=X'.format(ftr=ftr))
        plt.show()
        # ...by buyer
        sns.distplot(df[['BuyerID', ftr]].groupby(by='BuyerID').mean(), hist=True, kde=False, norm_hist=False)
        plt.title('Mean {ftr} per buyer\nfrequency distribution'.format(ftr=ftr))
        plt.xlabel(ftr)
        plt.ylabel('Number of buyers with\n{ftr}=X'.format(ftr=ftr))
        plt.show()
    return None


def plot_heuristic(
    df:pd.DataFrame
    ) -> None:
    r"""Plot heuristic to predict bad dealers.

    Args:
        df (pandas.DataFrame): DataFrame of formatted data.

    Returns:
        None

    """
    # Plot timeseries histogram of Returned vs SalesDate.
    df_plot = df[['SaleDate_decyear', 'Returned']].copy()
    itemized_counts = {
        ret: collections.Counter(grp['SaleDate_decyear'])
        for (ret, grp) in df_plot.groupby(by='Returned')}
    itemized_counts = collections.OrderedDict(
        sorted(itemized_counts.items(), key=lambda tup: tup[0], reverse=True))
    keys = itemized_counts.keys()
    bins = int(np.ceil((df_plot['SaleDate_decyear'].max() - df_plot['SaleDate_decyear'].min())/(1.0/52.0)))
    colors = sns.color_palette(n_colors=len(keys))[::-1]
    plt.hist(
        [list(itemized_counts[key].elements()) for key in itemized_counts.keys()],
        bins=bins, stacked=True, rwidth=1.0, label=keys, color=colors)
    plt.title('Returned vs SaleDate')
    plt.xlabel('SaleDate (decimal year)')
    plt.ylabel('Returned')
    plt.legend(title='Returned status', loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.show()
    # Plot timeseries histogram of Returned (0,1) vs SalesDate.
    df_plot = df.loc[df['Returned']!=-1, ['SaleDate_decyear', 'Returned']].copy()
    itemized_counts = {
        ret: collections.Counter(grp['SaleDate_decyear'])
        for (ret, grp) in df_plot.groupby(by='Returned')}
    itemized_counts = collections.OrderedDict(
        sorted(itemized_counts.items(), key=lambda tup: tup[0], reverse=True))
    keys = itemized_counts.keys()
    bins = int(np.ceil((df_plot['SaleDate_decyear'].max() - df_plot['SaleDate_decyear'].min())/(1.0/52.0)))
    plt.hist(
        [list(itemized_counts[key].elements()) for key in itemized_counts.keys()],
        bins=bins, stacked=True, rwidth=1.0, label=keys, color=colors[:2])
    plt.title('Returned vs SaleDate')
    plt.xlabel('SaleDate (decimal year)')
    plt.ylabel('Returned')
    plt.legend(title='Returned status', loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.show()
    return None

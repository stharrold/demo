#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Prediction application.

"""


# Import standard packages.
import bs4
import collections
import inspect
import itertools
import logging
import os
import pickle
import requests
import shelve
import sys
import textwrap
import time
import warnings
# Import installed packages.
import geopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn.cross_validation as sk_cv
import sklearn.cluster as sk_cl
import sklearn.decomposition as sk_dc
import sklearn.ensemble as sk_ens
import sklearn.metrics as sk_met
import sklearn.preprocessing as sk_pre
# Import local packages.
from .. import utils


# Define module exports:
__all__ = [
    'etl',
    'create_features',
    'plot_eda',
    'plot_heursitic',
    'update_features',
    'update_features_append',
    'create_features_new_data',
    'create_pipeline_model']


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
    # Note: Skip if cleaning new data for which Returned is unknown.
    if 'Returned' in df.columns:
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
    # Fill null values with -1 and cast to int.
    # Note: Skip if cleaning new data for which Returned is unknown.
    if 'Returned' in df.columns:
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
    # JDPowersCat: Replace nan with UNKNOWN
    logger.info("JDPowersCat: Replace 'nan' with 'UNKNOWN'.")
    df['JDPowersCat'] = df['JDPowersCat'].str.replace(' ', '').apply(
        lambda cat: 'UNKNOWN' if cat == 'nan' else cat)
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
        if col != buyer_retrate:
            df_plot = df[['BuyerID', col, buyer_retrate]].copy()
        else:
            df_plot = df[['BuyerID', buyer_retrate]].copy()
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


def update_features(
    df:pd.DataFrame
    ) -> pd.DataFrame:
    r"""Update features for timeseries training.

    Args:
        df (pandas.DataFrame): Dataframe of featurized data.

    Returns:
        df (pandas.DataFrame): Dataframe of updated featurized data.

    See Also:
        create_features

    Notes:
        * BuyerID_fracReturned1DivReturnedNotNull is the return rate for a buyer.

    TODO:
        * Modularize script into separate helper functions.
        * Modify dataframe in place

    """
    # Check input.
    # Copy dataframe to avoid in place modification.
    df = df.copy()
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
    # BuyerID, SellerID, VIN, SellingLocation, CarMake, JDPowersCat:
    # Make cumulative informative priors (*_num*, *_frac*) for string features.
    logger.info(textwrap.dedent("""\
        BuyerID, SellerID, VIN, SellingLocation, CarMake, JDPowersCat:
        Make cumulative informative priors (*_num*, *_frac*) for string features."""))
    # Cumulative features require sorting by time.
    assert (df['SaleDate'].diff().iloc[1:] >= np.timedelta64(0, 'D')).all()
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


def update_features_append(
    df_prev:pd.DataFrame,
    df_next:pd.DataFrame,
    debug:bool=False
    ) -> pd.DataFrame:
    r"""Update features and merge for timeseries training.

    Args:
        df_prev (pandas.DataFrame): Dataframe of old data.
        df_next (pandas.DataFrame): Dataframe of new data to be updated
            and appended to df_prev.
        debug (bool, optional, False): Flag to enforce assertions.
            True: Execute assertions. Slower runtime by 3x.
            False (default): Do not execute assertions. Faster runtime.

    Returns:
        df (pandas.DataFrame): Dataframe of updated, appended data.

    See Also:
        create_features

    Notes:
        * Only df_next is updated.
        * BuyerID_fracReturned1DivReturnedNotNull is the return rate for a buyer.

    TODO:
        * Modularize script into separate helper functions.
        * Modify dataframe in place

    """
    # Check input.
    # Copy dataframe to avoid in place modification.
    (df_prev, df_next) = (df_prev.copy(), df_next.copy())
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
    df_next['Returned_asm'] = df_next['Returned']
    df_next.loc[
        np.logical_and(df_next['Returned'] == -1, df_next['DSEligible'] == 0),
        'Returned_asm'] = 1
    df_next.loc[
        np.logical_and(df_next['Returned'] == -1, df_next['DSEligible'] == 1),
        'Returned_asm'] = 0
    logger.info("Relationship between DSEligible and Returned:\n{pt}".format(
        pt=pd.pivot_table(
            df_next[['DSEligible', 'Returned']].astype(str),
            index='DSEligible', columns='Returned',
            aggfunc=len, margins=True, dropna=False)))
    logger.info("Relationship between DSEligible and Returned_asm:\n{pt}".format(
        pt=pd.pivot_table(
            df_next[['DSEligible', 'Returned_asm']].astype(str),
            index='DSEligible', columns='Returned_asm',
            aggfunc=len, margins=True, dropna=False)))
    logger.info("Relationship between Returned and Returned_asm:\n{pt}".format(
        pt=pd.pivot_table(
            df_next[['Returned', 'Returned_asm']].astype(str),
            index='Returned', columns='Returned_asm',
            aggfunc=len, margins=True, dropna=False)))
    ########################################
    # BuyerID, SellerID, VIN, SellingLocation, CarMake, JDPowersCat:
    # Make cumulative informative priors (*_num*, *_frac*) for string features.
    logger.info(textwrap.dedent("""\
        BuyerID, SellerID, VIN, SellingLocation, CarMake, JDPowersCat:
        Make cumulative informative priors (*_num*, *_frac*) for string features."""))
    # Cumulative features require sorting by time.
    if debug:
        assert (df_prev['SaleDate'].diff().iloc[1:] >= np.timedelta64(0, 'D')).all()
        assert (df_next['SaleDate'].diff().iloc[1:] >= np.timedelta64(0, 'D')).all()
    for col in ['BuyerID', 'SellerID', 'VIN', 'SellingLocation', 'CarMake', 'JDPowersCat']:
        logger.info("Processing {col}".format(col=col))
        prev_nums = df_prev.groupby(by=col).last()
        ####################
        # Cumulative count of transactions and DSEligible:
        # Cumulative count of transactions (yes including current).
        df_next[col+'_numTransactions'] = df_next[[col]].groupby(by=col).cumcount().astype(int) + 1
        df_next[col+'_numTransactions'].fillna(value=1, inplace=True)
        df_next[col+'_numTransactions'] += df_next[col].map(prev_nums[col+'_numTransactions']).fillna(value=0)
        # Cumulative count of transations that were DealShield-eligible (yes including current).
        df_next[col+'_numDSEligible1'] = df_next[[col, 'DSEligible']].groupby(by=col)['DSEligible'].cumsum().astype(int)
        df_next[col+'_numDSEligible1'].fillna(value=0, inplace=True)
        df_next[col+'_numDSEligible1'] += df_next[col].map(prev_nums[col+'_numDSEligible1']).fillna(value=0)
        # Cumulative ratio of transactions that were DealShield-eligible (0=bad, 1=good).
        df_next[col+'_fracDSEligible1DivTransactions'] = df_next[col+'_numDSEligible1']/df_next[col+'_numTransactions']
        df_next[col+'_fracDSEligible1DivTransactions'].fillna(value=1, inplace=True)
        ####################
        # DSEligible and Returned
        # Note:
        # * DealShield-purchased ==> Returned != -1 (not null)
        # * below requires
        #     DSEligible == 0 ==> Returned == -1 (is null)
        #     Returned != -1 (not null) ==> DSEligible == 1
        if debug:
            assert (df_prev.loc[df_prev['DSEligible']==0, 'Returned'] == -1).all()
            assert (df_prev.loc[df_prev['Returned']!=-1, 'DSEligible'] == 1).all()
            assert (df_next.loc[df_next['DSEligible']==0, 'Returned'] == -1).all()
            assert (df_next.loc[df_next['Returned']!=-1, 'DSEligible'] == 1).all()
        # Cumulative count of transactions that were DealShield-eligible and DealShield-purchased.
        df_tmp = df_next[[col, 'Returned']].copy()
        df_tmp['ReturnedNotNull'] = df_tmp['Returned'] != -1
        df_next[col+'_numReturnedNotNull'] = df_tmp[[col, 'ReturnedNotNull']].groupby(by=col)['ReturnedNotNull'].cumsum().astype(int)
        df_next[col+'_numReturnedNotNull'].fillna(value=0, inplace=True)
        df_next[col+'_numReturnedNotNull'] += df_next[col].map(prev_nums[col+'_numReturnedNotNull']).fillna(value=0)
        del df_tmp
        # Cumulative ratio of DealShield-eligible transactions that were DealShield-purchased (0=mode).
        df_next[col+'_fracReturnedNotNullDivDSEligible1'] = df_next[col+'_numReturnedNotNull']/df_next[col+'_numDSEligible1']
        df_next[col+'_fracReturnedNotNullDivDSEligible1'].fillna(value=0, inplace=True)
        # Cumulative count of transactions that were DealShield-elegible and DealShield-purchased and DealShield-returned.
        df_tmp = df_next[[col, 'Returned']].copy()
        df_tmp['Returned1'] = df_tmp['Returned'] == 1
        df_next[col+'_numReturned1'] = df_tmp[[col, 'Returned1']].groupby(by=col)['Returned1'].cumsum().astype(int)
        df_next[col+'_numReturned1'].fillna(value=0, inplace=True)
        df_next[col+'_numReturned1'] += df_next[col].map(prev_nums[col+'_numReturned1']).fillna(value=0)
        del df_tmp
        # Cumulative ratio of DealShield-eligible, DealShield-purchased transactions that were DealShield-returned (0=good, 1=bad).
        # Note: BuyerID_fracReturned1DivReturnedNotNull is the cumulative return rate for a buyer.
        df_next[col+'_fracReturned1DivReturnedNotNull'] = df_next[col+'_numReturned1']/df_next[col+'_numReturnedNotNull']
        df_next[col+'_fracReturned1DivReturnedNotNull'].fillna(value=0, inplace=True)
        # Check that weighted average of return rate equals overall return rate.
        # Note: Requires groups sorted by date, ascending.
        if debug:
            df_tmp = df_prev.append(df_next)
            assert np.isclose(
                (df_tmp[[col, col+'_fracReturned1DivReturnedNotNull', col+'_numReturnedNotNull']].groupby(by=col).last().product(axis=1).sum()/\
                 df_tmp[[col, col+'_numReturnedNotNull']].groupby(by=col).last().sum()).values[0],
                sum(df_tmp['Returned']==1)/sum(df_tmp['Returned'] != -1),
                equal_nan=True)
            del df_tmp
        ####################
        # DSEligible and Returned_asm
        # NOTE:
        # * Below requires
        #     DSEligible == 0 ==> Returned_asm == 1
        #     Returned_asm == 0 ==> DSEligible == 1
        if debug:
            assert (df_prev.loc[df_prev['DSEligible']==0, 'Returned_asm'] == 1).all()
            assert (df_prev.loc[df_prev['Returned_asm']==0, 'DSEligible'] == 1).all()
            assert (df_next.loc[df_next['DSEligible']==0, 'Returned_asm'] == 1).all()
            assert (df_next.loc[df_next['Returned_asm']==0, 'DSEligible'] == 1).all()
        # Cumulative number of transactions that were assumed to be returned.
        df_tmp = df_next[[col, 'Returned_asm']].copy()
        df_tmp['Returnedasm1'] = df_tmp['Returned_asm'] == 1
        df_next[col+'_numReturnedasm1'] = df_tmp[[col, 'Returnedasm1']].groupby(by=col)['Returnedasm1'].cumsum().astype(int)
        df_next[col+'_numReturnedasm1'].fillna(value=0, inplace=True)
        df_next[col+'_numReturnedasm1'] += df_next[col].map(prev_nums[col+'_numReturnedasm1']).fillna(value=0)
        del df_tmp
        # Cumulative ratio of transactions that were assumed to be returned (0=mode).
        df_next[col+'_fracReturnedasm1DivTransactions'] = df_next[col+'_numReturnedasm1']/df_next[col+'_numTransactions']
        df_next[col+'_fracReturnedasm1DivTransactions'].fillna(value=0, inplace=True)
        # Check that weighted average of assumed return rate equals overall assumed return rate.
        if debug:
            df_tmp = df_prev.append(df_next)
            assert np.isclose(
                (df_tmp[[col, col+'_fracReturnedasm1DivTransactions', col+'_numTransactions']].groupby(by=col).last().product(axis=1).sum()/\
                 df_tmp[[col, col+'_numTransactions']].groupby(by=col).last().sum()).values[0],
                sum(df_tmp['Returned_asm']==1)/sum(df_tmp['Returned_asm'] != -1),
                equal_nan=True)
            del df_tmp
        # Note:
        #   * Number of transactions that were DealShield-eligible and assumed to be returned ==
        #     number of transactions that were DealShield-elegible and DealShield-purchased and DealShield-returned
        #     (numReturned1)
    # Return updated, appended dataframe.
    return df_prev.append(df_next)


def create_features_new_data(
    df_prev:pd.DataFrame,
    df_next:pd.DataFrame,
    path_data_dir:str,
    debug:bool=False
    ) -> pd.DataFrame:
    r"""Create features for post-ETL data.

    Args:
        df_prev (pandas.DataFrame): Dataframe of old data.
        df_next (pandas.DataFrame): Dataframe of new data with missing target column
            ('Returned') for which features are extracted.
        path_data_dir (str): Path to data directory for caching geocode shelf file.
        debug (bool, optional, False): Flag to enforce assertions.
            True: Execute assertions. Slower runtime by 3x.
            False (default): Do not execute assertions. Faster runtime.

    Returns:
        df (pandas.DataFrame): Dataframe of extracted data.

    See Also:
        etl

    Notes:
        * BuyerID_fracReturned1DivReturnedNotNull is the return rate for a buyer.
        * df_prev and df_next have overlapping indexes.

    TODO:
        * Modularize script into separate helper functions.
        * Modify dataframe in place

    """
    # Check input.
    # Copy dataframe to avoid in place modification.
    (df_prev, df_next) = (df_prev.copy(), df_next.copy())
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
    #   If Returned=-1 and DSEligible=0, then Returned_asm=1
    #   If Returned=-1 and DSEligible=1, then Returned_asm=0
    # For new data:
    #   If DSEligible=0, then Returned=-1, then Returned_asm=1
    #   If DSEligible=1, then Returned_asm is the average of the buyer's Returned_asm, or if new buyer, then 0.
    logger.info(textwrap.dedent("""\
        Returned_asm: Assume returned status to fill nulls as new feature.
        If Returned=-1 and DSEligible=0, then Returned_asm=1 (assumes low P(resale|buyer, car))
        If Returned=-1 and DSEligible=1, then Returned_asm=0 (assumes high P(resale|buyer, car))"""))
    logger.info(textwrap.dedent("""\
        For new data:
        If DSEligible=0, then Returned=-1, then Returned_asm=1
        If DSEligible=1, then Returned_asm is the average of the buyer's Returned_asm, or if new buyer, then 0."""))
    df_next.loc[df_next['DSEligible']==0, 'Returned_asm'] = 1
    prev_nums = df_prev.loc[df_prev['DSEligible']==1, ['BuyerID', 'Returned_asm']].groupby(by='BuyerID').mean()
    df_next.loc[df_next['DSEligible']==1, 'Returned_asm'] = \
        df_next.loc[df_next['DSEligible']==1, 'BuyerID'].map(prev_nums['Returned_asm']).fillna(value=0)
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
        for loc in df_next['SellingLocation'].unique():
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
    df_next['SellingLocation_lat'] = df_next['SellingLocation'].map(sellloc_lat)
    df_next['SellingLocation_lon'] = df_next['SellingLocation'].map(sellloc_lon)
    # # TODO: experiment with one-hot encoding (problems is that it doesn't scale)
    # df_next = pd.merge(
    #     left=df_next,
    #     right=pd.get_dummies(df_next['SellingLocation'], prefix='SellingLocation'),
    #     how='inner',
    #     left_index=True,
    #     right_index=True)
    ########################################
    # JDPowersCat: One-hot encoding
    # TODO: Estimate sizes from Wikipedia, e.g. https://en.wikipedia.org/wiki/Vehicle_size_class.
    logger.info("JDPowersCat: One-hot encoding.")
    # Cast to string, replacing 'nan' with 'UNKNOWN'.
    df_next['JDPowersCat'] = (df_next['JDPowersCat'].astype(str)).str.replace(' ', '').apply(
        lambda cat: 'UNKNOWN' if cat == 'nan' else cat)
    # One-hot encoding.
    df_next = pd.merge(
        left=df_next,
        right=pd.get_dummies(df_next['JDPowersCat'], prefix='JDPowersCat'),
        left_index=True,
        right_index=True)
    ########################################
    # LIGHT_N0G1Y2R3
    # Rank lights by warning level.
    logger.info("LIGHT_N0G1Y2R3: Rank lights by warning level (null=0, green=1, yellow=2, red=3).")
    df_next['LIGHT_N0G1Y2R3'] = df_next['LIGHTG']*1 + df_next['LIGHTY']*2 + df_next['LIGHTR']*3
    ########################################
    # SaleDate_*: Extract timeseries features.
    logger.info("SaleDate: Extract timeseries features.")
    df_next['SaleDate_dow'] = df_next['SaleDate'].dt.dayofweek
    df_next['SaleDate_doy'] = df_next['SaleDate'].dt.dayofyear
    df_next['SaleDate_day'] = df_next['SaleDate'].dt.day
    df_next['SaleDate_decyear'] = df_next['SaleDate'].dt.year + (df_next['SaleDate'].dt.dayofyear-1)/366
    ########################################
    # BuyerID, SellerID, VIN, SellingLocation, CarMake, JDPowersCat:
    # Make cumulative informative priors (*_num*, *_frac*) for string features.
    logger.info(textwrap.dedent("""\
        BuyerID, SellerID, VIN, SellingLocation, CarMake, JDPowersCat:
        Make cumulative informative priors (*_num*, *_frac*) for string features."""))
    # Cumulative features require sorting by time.
    # Note: df_prev and df_next have overlapping indexes after `reset_index`.
    df_next.sort_values(by=['SaleDate'], inplace=True)
    df_next.reset_index(drop=True, inplace=True)
    if debug:
        assert (df_prev['SaleDate'].diff().iloc[1:] >= np.timedelta64(0, 'D')).all()
        assert (df_next['SaleDate'].diff().iloc[1:] >= np.timedelta64(0, 'D')).all()
    for col in ['BuyerID', 'SellerID', 'VIN', 'SellingLocation', 'CarMake', 'JDPowersCat']:
        logger.info("Processing {col}".format(col=col))
        prev_nums = df_prev.groupby(by=col).last()
        ####################
        # Cumulative count of transactions and DSEligible:
        # Cumulative count of transactions (yes including current).
        df_next[col+'_numTransactions'] = df_next[[col]].groupby(by=col).cumcount().astype(int) + 1
        df_next[col+'_numTransactions'].fillna(value=1, inplace=True)
        df_next[col+'_numTransactions'] += df_next[col].map(prev_nums[col+'_numTransactions']).fillna(value=0)
        # Cumulative count of transations that were DealShield-eligible (yes including current).
        df_next[col+'_numDSEligible1'] = df_next[[col, 'DSEligible']].groupby(by=col)['DSEligible'].cumsum().astype(int)
        df_next[col+'_numDSEligible1'].fillna(value=0, inplace=True)
        df_next[col+'_numDSEligible1'] += df_next[col].map(prev_nums[col+'_numDSEligible1']).fillna(value=0)
        # Cumulative ratio of transactions that were DealShield-eligible (0=bad, 1=good).
        df_next[col+'_fracDSEligible1DivTransactions'] = df_next[col+'_numDSEligible1']/df_next[col+'_numTransactions']
        df_next[col+'_fracDSEligible1DivTransactions'].fillna(value=1, inplace=True)
        ####################
        # DSEligible and Returned
        # Note:
        # * DealShield-purchased ==> Returned != -1 (not null)
        # * below requires
        #     DSEligible == 0 ==> Returned == -1 (is null)
        #     Returned != -1 (not null) ==> DSEligible == 1
        if debug:
            assert (df_prev.loc[df_prev['DSEligible']==0, 'Returned'] == -1).all()
            assert (df_prev.loc[df_prev['Returned']!=-1, 'DSEligible'] == 1).all()
        # Cumulative count of transactions that were DealShield-eligible and DealShield-purchased.
        df_next[col+'_numReturnedNotNull'] = df_next[col].map(prev_nums[col+'_numReturnedNotNull']).fillna(value=0)
        # Cumulative ratio of DealShield-eligible transactions that were DealShield-purchased (0=mode).
        df_next[col+'_fracReturnedNotNullDivDSEligible1'] = df_next[col+'_numReturnedNotNull']/df_next[col+'_numDSEligible1']
        df_next[col+'_fracReturnedNotNullDivDSEligible1'].fillna(value=0, inplace=True)
        # Cumulative count of transactions that were DealShield-elegible and DealShield-purchased and DealShield-returned.
        df_next[col+'_numReturned1'] = df_next[col].map(prev_nums[col+'_numReturned1']).fillna(value=0)
        # Cumulative ratio of DealShield-eligible, DealShield-purchased transactions that were DealShield-returned (0=good, 1=bad).
        # Note: BuyerID_fracReturned1DivReturnedNotNull is the cumulative return rate for a buyer.
        df_next[col+'_fracReturned1DivReturnedNotNull'] = df_next[col+'_numReturned1']/df_next[col+'_numReturnedNotNull']
        df_next[col+'_fracReturned1DivReturnedNotNull'].fillna(value=0, inplace=True)
        # Check that weighted average of return rate equals overall return rate.
        # Note: Requires groups sorted by date, ascending.
        if debug:
            assert np.isclose(
                (df_prev[[col, col+'_fracReturned1DivReturnedNotNull', col+'_numReturnedNotNull']].groupby(by=col).last().product(axis=1).sum()/\
                 df_prev[[col, col+'_numReturnedNotNull']].groupby(by=col).last().sum()).values[0],
                sum(df_prev['Returned']==1)/sum(df_prev['Returned'] != -1),
                equal_nan=True)
        ####################
        # DSEligible and Returned_asm
        # NOTE:
        # * Below requires
        #     DSEligible == 0 ==> Returned_asm == 1
        #     Returned_asm == 0 ==> DSEligible == 1
        if debug:
            assert (df_prev.loc[df_prev['DSEligible']==0, 'Returned_asm'] == 1).all()
            assert (df_prev.loc[df_prev['Returned_asm']==0, 'DSEligible'] == 1).all()
            assert (df_next.loc[df_next['DSEligible']==0, 'Returned_asm'] == 1).all()
            assert (df_next.loc[df_next['Returned_asm']==0, 'DSEligible'] == 1).all()
        # Cumulative number of transactions that were assumed to be returned.
        # Note: For new data, 'Returned_asm' may be a float.
        df_tmp = df_next[[col, 'Returned_asm']].copy()
        df_tmp['Returnedasm1'] = df_tmp['Returned_asm']
        df_next[col+'_numReturnedasm1'] = df_tmp[[col, 'Returnedasm1']].groupby(by=col)['Returnedasm1'].cumsum()
        df_next[col+'_numReturnedasm1'].fillna(value=0, inplace=True)
        df_next[col+'_numReturnedasm1'] += df_next[col].map(prev_nums[col+'_numReturnedasm1']).fillna(value=0)
        del df_tmp
        # Cumulative ratio of transactions that were assumed to be returned (0=mode).
        df_next[col+'_fracReturnedasm1DivTransactions'] = df_next[col+'_numReturnedasm1']/df_next[col+'_numTransactions']
        df_next[col+'_fracReturnedasm1DivTransactions'].fillna(value=0, inplace=True)
        # Check that weighted average of assumed return rate equals overall assumed return rate.
        if debug:
            assert np.isclose(
                (df_prev[[col, col+'_fracReturnedasm1DivTransactions', col+'_numTransactions']].groupby(by=col).last().product(axis=1).sum()/\
                 df_prev[[col, col+'_numTransactions']].groupby(by=col).last().sum()).values[0],
                sum(df_prev['Returned_asm']==1)/sum(df_prev['Returned_asm'] != -1),
                equal_nan=True)
        # Note:
        #   * Number of transactions that were DealShield-eligible and assumed to be returned ==
        #     number of transactions that were DealShield-elegible and DealShield-purchased and DealShield-returned
        #     (numReturned1)
    return df_next


def create_pipeline_model(
    df:pd.DataFrame,
    path_data_dir:str,
    show_plots:bool=False):
    r"""Create pipeline model.

    """
    # Check arguments.
    path_plot_dir = os.path.join(path_data_dir, 'plot_model')
    ########################################
    print('#'*80)
    # Define target and features
    target = 'Returned'
    features = set(df.columns[np.logical_or(df.dtypes=='int64', df.dtypes=='float64')])
    features.difference_update([target])
    features = sorted(features)
    print('Features:')
    print(features)
    print()
    ########################################
    print('#'*80)
    print(textwrap.dedent("""\
        `Container`: Create an empty container class and
        dynamically allocate attributes to hold variables for specific steps
        of the pipeline. """))
    Container = utils.utils.Container
    step = Container()

    print(textwrap.dedent("""\
        `step.s0.[df,ds]_[features,target]`: Save initial state of features, target."""))
    step.s0 = Container()
    step.s0.dfs = Container()
    step.s0.dfs.df_features = df[features].copy()
    step.s0.dfs.ds_target = df[target].copy()

    # TODO: REDO after this point with step.sN.dfs.[df_features,ds_target]
    # rather than redefining [df_features,ds_target]
    df_features = step.s0.dfs.df_features
    ds_target = step.s0.dfs.ds_target
    print()
    ########################################
    print('#'*80)
    print(textwrap.dedent("""\
        `transformer_scaler`, `transformer_pca`: Scale data
        then make groups of similar records with k-means clustering,
        both with and without PCA. Use the silhouette score to determine
        the number of clusters.
        """))
    time_start = time.perf_counter()

    # Scale data prior to comparing clusters with/without PCA. 
    # Note: Using sklearn.preprocessing.RobustScaler with
    #     sklearn.decomposition.IncrementalPCA(whiten=False)
    #     is often the most stable (slowly varying scores)
    #     with highest scores. Centroid agreement can still be
    #     off due to outliers.
    transformer_scaler = sk_pre.RobustScaler()
    features_scaled = transformer_scaler.fit_transform(X=df_features)
    transformer_pca = sk_dc.IncrementalPCA(whiten=False)
    features_scaled_pca = transformer_pca.fit_transform(X=features_scaled)

    print("`columns.pkl`, `transformer_scaler.pkl`, `transformer_pca.pkl`: Save column order and transformers.")
    path_data = path_data_dir
    path_cols = os.path.join(path_data, 'columns.pkl')
    with open(path_cols, mode='wb') as fobj:
        pickle.dump(obj=df_features.columns, file=fobj)
    path_tform_scl = os.path.join(path_data, 'transformer_scaler.pkl')
    with open(path_tform_scl, mode='wb') as fobj:
        pickle.dump(obj=transformer_scaler, file=fobj)
    path_tform_pca = os.path.join(path_data, 'transformer_pca.pkl')
    with open(path_tform_pca, mode='wb') as fobj:
        pickle.dump(obj=transformer_pca, file=fobj)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        print("Plot scores for scaled features:")
        utils.utils.calc_silhouette_scores(
            df_features=features_scaled, n_clusters_min=2, n_clusters_max=10,
            size_sub=None, n_scores=10, show_progress=True, show_plot=True)

        print("Plot scores for scaled PCA features:")
        utils.utils.calc_silhouette_scores(
            df_features=features_scaled_pca, n_clusters_min=2, n_clusters_max=10,
            size_sub=None, n_scores=10, show_progress=True, show_plot=True)

    time_stop = time.perf_counter()
    print("Time elapsed (sec) = {diff:.1f}".format(diff=time_stop-time_start))
    print()
    ########################################
    print('#'*80)
    print(textwrap.dedent("""\
        `transformer_kmeans`, `transformer_kmeans_pca`:
        Fit k-means to the data with/without PCA and
        compare the centroids for the clusters."""))

    # TODO: Fix plot. Assign clusters IDs in a deterministic way so that
    #   cluster 0 raw matches cluster 0 transformed.
    time_start = time.perf_counter()

    n_clusters = 2 # from silhouette scores

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Cluster scaled features with/without PCA using minibatch k-means
        transformer_kmeans = sk_cl.MiniBatchKMeans(n_clusters=n_clusters)
        transformer_kmeans.fit(X=features_scaled)
        transformer_kmeans_pca = sk_cl.MiniBatchKMeans(n_clusters=n_clusters)
        transformer_kmeans_pca.fit(X=features_scaled_pca)

    print("`transformer_kmeans.pkl`, `transformer_kmeans_pca.pkl`: Save transformers.")
    path_tform_km = os.path.join(path_data, 'transformer_kmeans.pkl')
    with open(path_tform_km, mode='wb') as fobj:
        pickle.dump(obj=transformer_kmeans, file=fobj)
    path_tform_km_pca = os.path.join(path_data, 'transformer_kmeans_pca.pkl')
    with open(path_tform_km_pca, mode='wb') as fobj:
        pickle.dump(obj=transformer_kmeans_pca, file=fobj)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Plot clusters in scaled feature space.
        centroids = transformer_kmeans.cluster_centers_
        transformed_centroids = transformer_pca.inverse_transform(transformer_kmeans_pca.cluster_centers_)
        (col_1, col_0) = np.argsort(np.var(features_scaled, axis=0))[-2:]
        (name_1, name_0) = (df_features.columns.values[col_1], df_features.columns.values[col_0])
        plt.title("Data and centroids within scaled feature space")
        tfmask_gt01 = df_features[buyer_retrate] > buyer_retrate_max
        plt.plot(features_scaled[tfmask_gt01, col_0], features_scaled[tfmask_gt01, col_1],
                 marker='o', linestyle='', color=sns.color_palette()[2], alpha=0.5,
                 label='data, buyer_retrate_gt01')
        tfmask_lt01 = np.logical_not(tfmask_gt01)
        plt.plot(features_scaled[tfmask_lt01, col_0], features_scaled[tfmask_lt01, col_1],
                 marker='.', linestyle='', color=sns.color_palette()[1], alpha=0.5,
                 label='data, buyer_retrate_lt01')
        plt.plot(centroids[:, col_0], centroids[:, col_1],
                 marker='+', linestyle='', markeredgewidth=2, markersize=12,
                 color=sns.color_palette()[0], label='centroids')
        for (idx, centroid) in enumerate(centroids):
            plt.annotate(
                str(idx), xy=(centroid[col_0], centroid[col_1]),
                xycoords='data', xytext=(0, 0), textcoords='offset points', color='black',
                fontsize=18, rotation=0)
        plt.plot(transformed_centroids[:, col_0], transformed_centroids[:, col_1],
                 marker='x', linestyle='', markeredgewidth=2, markersize=10,
                 color=sns.color_palette()[1], label='transformed centroids')
        for (idx, transformed_centroid) in enumerate(transformed_centroids):
            plt.annotate(
                str(idx), xy=(transformed_centroid[col_0], transformed_centroid[col_1]),
                xycoords='data', xytext=(0, 0), textcoords='offset points', color='black',
                fontsize=18, rotation=0)
        plt.xlabel("Scaled '{name}', highest variance".format(name=name_0))
        plt.ylabel("Scaled '{name}', next highest variance".format(name=name_1))
        plt.legend(loc='upper left')
        if show_plots:
            plt.show()
        plt.gcf().clear()
        plt.clf()
        plt.cla()
        plt.close()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Plot clusters in scaled feature PCA space.
        transformed_centroids = transformer_pca.transform(transformer_kmeans.cluster_centers_)
        centroids = transformer_kmeans_pca.cluster_centers_
        plt.title("Data and centroids within scaled feature PCA space")
        plt.plot(features_scaled_pca[tfmask_gt01, 0], features_scaled_pca[tfmask_gt01, 1],
                 marker='o', linestyle='', color=sns.color_palette()[2], alpha=0.5,
                 label='transformed data, buyer_retrate_gt01')
        plt.plot(features_scaled_pca[tfmask_lt01, 0], features_scaled_pca[tfmask_lt01, 1],
                 marker='.', linestyle='', color=sns.color_palette()[1], alpha=0.5,
                 label='transformed data, buyer_retrate_lt01')
        plt.plot(transformed_centroids[:, 0], transformed_centroids[:, 1],
                 marker='+', linestyle='', markeredgewidth=2, markersize=12,
                 color=sns.color_palette()[0], label='transformed centroids')
        for (idx, transformed_centroid) in enumerate(transformed_centroids):
            plt.annotate(
                str(idx), xy=(transformed_centroid[0], transformed_centroid[1]),
                xycoords='data', xytext=(0, 0), textcoords='offset points', color='black',
                fontsize=18, rotation=0)
        plt.plot(centroids[:, 0], centroids[:, 1],
                 marker='x', linestyle='', markeredgewidth=2, markersize=10,
                 color=sns.color_palette()[1], label='centroids')
        for (idx, centroid) in enumerate(centroids):
            plt.annotate(
                str(idx), xy=(centroid[0], centroid[1]),
                xycoords='data', xytext=(0, 0), textcoords='offset points', color='black',
                fontsize=18, rotation=0)
        plt.xlabel('Principal component 0')
        plt.ylabel('Principal component 1')
        plt.legend(loc='upper left')
        if show_plots:
            plt.show()
        plt.gcf().clear()
        plt.clf()
        plt.cla()
        plt.close()

    time_stop = time.perf_counter()
    print("Time elapsed (sec) = {diff:.1f}".format(diff=time_stop-time_start))
    print()
    ########################################
    print('#'*80)
    print(textwrap.dedent("""\
        `df_features2`: Combine `df_features` with
        cluster labels, cluster distances, PCA components, PCA cluster labels,
        and PCA cluster distances into `df_features`."""))
    time_start = time.perf_counter()

    # Cluster labels and distances in feature space.
    ds_clusters = pd.Series(
        transformer_kmeans.predict(X=features_scaled),
        index=df_features.index, name='cluster')
    n_digits = len(str(len(transformer_kmeans.cluster_centers_)))
    columns = [
        'cluster_{num}_dist'.format(num=str(num).rjust(n_digits, '0'))
        for num in range(len(transformer_kmeans.cluster_centers_))]
    df_cluster_dists = pd.DataFrame(
        transformer_kmeans.transform(X=features_scaled),
        index=df_features.index, columns=columns)
    if not np.all(ds_clusters.values == np.argmin(df_cluster_dists.values, axis=1)):
        raise AssertionError(
            ("Program error. Not all cluster labels match cluster label\n" +
             "with minimum distance to record.\n" +
             "Required: np.all(ds_clusters.values == np.argmin(df_cluster_dists.values, axis=1))"))

    # PCA features.
    n_digits = len(str(transformer_pca.n_components_))
    columns = [
        'pca_comp_{num}'.format(num=str(num).rjust(n_digits, '0'))
        for num in range(transformer_pca.n_components_)]
    df_features_pca = pd.DataFrame(
        features_scaled_pca, index=df_features.index, columns=columns)

    # Cluster labels and distances in PCA feature space.
    ds_clusters_pca = pd.Series(
        transformer_kmeans_pca.predict(X=features_scaled_pca),
        index=df_features.index, name='pca_cluster')
    n_digits = len(str(len(transformer_kmeans_pca.cluster_centers_)))
    columns = [
        'pca_cluster_{num}_dist'.format(num=str(num).rjust(n_digits, '0'))
        for num in range(len(transformer_kmeans_pca.cluster_centers_))]
    df_cluster_dists_pca = pd.DataFrame(
        transformer_kmeans_pca.transform(X=features_scaled_pca),
        index=df_features.index, columns=columns)
    if not np.all(ds_clusters_pca.values == np.argmin(df_cluster_dists_pca.values, axis=1)):
        raise AssertionError(
            ("Program error. Not all PCA cluster labels match PCA cluster label\n" +
             "with minimum distance to record.\n" +
             "Required: np.all(ds_clusters_pca.values == np.argmin(df_cluster_dists_pca.values, axis=1))"))

    # Combine with original `df_features` into new `df_features2`.
    df_features2 = pd.concat(
        [df_features, ds_clusters, df_cluster_dists,
         df_features_pca, ds_clusters_pca, df_cluster_dists_pca],
        axis=1, copy=True)

    time_stop = time.perf_counter()
    print("Time elapsed (sec) = {diff:.1f}".format(diff=time_stop-time_start))
    print()
    ########################################
    print('#'*80)
    print(textwrap.dedent("""\
        `df_importances` , `important_features`, `df_features3`:
        `df_features3` is a view into (not a copy) of `df_features2` with only
        `important_features`. Feature importance is the normalized reduction
        in the loss score. A feature is selected as 'important' if its average
        importance is greater than the average importance of the random feature."""))
    time_start = time.perf_counter()

    # Calculate feature importances.
    # Note:
    # * `n_estimators` impact the feature importances but only have a small
    #     effect on the relative importances.
    # * `n_estimators` impact the scores but only have a small effect on the relative scores.
    # * Use replace=False for maximum data variety.
    # TODO: Use a significance test for feature importance.
    estimator = sk_ens.ExtraTreesRegressor(n_estimators=10, n_jobs=-1)
    df_importances = utils.utils.calc_feature_importances(
        estimator=estimator, df_features=df_features2, ds_target=ds_target,
        replace=False, show_progress=True, show_plot=True)
    important_features = df_importances.columns[
        df_importances.mean() > df_importances['random'].mean()]
    important_features = list(
        df_importances[important_features].mean().sort_values(ascending=False).index)
    df_features3 = df_features2[important_features]
    print("`important_features` =")
    print(important_features)
    print()

    time_stop = time.perf_counter()
    print("Time elapsed (sec) = {diff:.1f}".format(diff=time_stop-time_start))
    
    print("`df_features`: Most significant projections of PCA component 78:")
    print(sorted(list(zip(df_features, transformer_pca.components_[78])), key=lambda tup: tup[1])[:3])
    print('...')
    print(sorted(list(zip(df_features, transformer_pca.components_[78])), key=lambda tup: tup[1])[-3:])
    print()
    ########################################
    print('#'*80)
    print(textwrap.dedent("""\
        Tune feature space by optimizing the model score
        with cross validation. Model scores are R^2,
        the coefficient of determination."""))
    time_start = time.perf_counter()

    print("Progress:", end=' ')
    size_data = len(df_features3)
    size_sub = 1000
    frac_test = 0.2
    replace = False
    n_scores = 10
    estimator = sk_ens.ExtraTreesRegressor(n_estimators=10, n_jobs=-1)
    nftrs_scores = list()
    idxs = itertools.chain(range(0, 10), range(10, 30, 3), range(30, len(important_features), 10))
    for idx in idxs:
        n_ftrs = idx+1
        ftrs = important_features[:n_ftrs]
        scores = list()
        for _ in range(0, n_scores):
            idxs_sub = np.random.choice(a=size_data, size=size_sub, replace=replace)
            (ftrs_train, ftrs_test,
             trg_train, trg_test) = sk_cv.train_test_split(
                df_features3[ftrs].values[idxs_sub], ds_target.values[idxs_sub],
                test_size=frac_test)
            estimator.fit(X=ftrs_train, y=trg_train)
            scores.append(estimator.score(X=ftrs_test, y=trg_test))
        nftrs_scores.append([n_ftrs, scores])
        if idx % 10 == 0:
            print("{frac:.0%}".format(frac=(idx+1)/len(important_features)), end=' ')
    print('\n')

    nftrs_pctls = np.asarray(
        [np.append(tup[0], np.percentile(tup[1], q=[5,50,95]))
         for tup in nftrs_scores])
    plt.plot(
        nftrs_pctls[:, 0], nftrs_pctls[:, 2],
        marker='.', color=sns.color_palette()[0],
        label='50th pctl score')
    plt.fill_between(
        nftrs_pctls[:, 0],
        y1=nftrs_pctls[:, 1],
        y2=nftrs_pctls[:, 3],
        alpha=0.5, color=sns.color_palette()[0],
        label='5-95th pctls of scores')
    plt.title("Model score vs number of features")
    plt.xlabel("Number of features")
    plt.ylabel("Model score")
    plt.legend(loc='upper left')
    plt.savefig(
        os.path.join(path_plot_dir, 'model_tune_nfeatures.png'),
        bbox_inches='tight', dpi=300)
    if show_plots:
        plt.show()
    plt.gcf().clear()
    plt.clf()
    plt.cla()
    plt.close()

    time_stop = time.perf_counter()
    print("Time elapsed (sec) = {diff:.1f}".format(diff=time_stop-time_start))
    print()
    ########################################
    print('#'*80)
    print("""`important_features2`, `df_features4`:
    `df_features4` is a view into (not a copy) of `df_features3` with only
    `important_features2`. Feature importance is the normalized reduction
    in the loss score. A feature is selected as 'important' from the
    model score vs features plot.
    """)
    time_start = time.perf_counter()

    # Keep top 10 features from score vs features plot.
    important_features2 = important_features[:10]

    df_features4 = df_features3[important_features2]
    print("`important_features2` =")
    print(important_features2)
    print()

    print("""Cluster map of important feature correlations with heirarchical relationships.
    The deeper of the dendrogram node, the higher (anti)correlated the features are.
    The Spearman rank correlation accommodates non-linear features.
    The pair plot is a scatter matrix plot of columns vs each other.
    """)

    # Notes:
    # * `size_sub` for computing correlations should be <= 1e3 else runtime is long.
    # * Use replace=False to show most data variety.
    # * For pairplot, only plot the target variable with the top 5 important
    #     features for legibility.
    # * For clustermap, `nlabels` shows every `nlabels`th label, so 20 labels total.
    size_sub = min(int(1e3), len(df_features4.index))
    idxs_sub = np.random.choice(a=df_features4.index, size=size_sub, replace=False)
    df_plot_sub = df_features4.loc[idxs_sub].copy()
    df_plot_sub[target] = ds_target.loc[idxs_sub].copy()
    df_plot_sub['buyer_retrate_gt01'] = df_features3.loc[idxs_sub, buyer_retrate] > buyer_retrate_max

    print(("Clustermap of target, '{target}', top 10 important features, buyer_retrate_gt01:").format(
            target=target))
    sns.clustermap(df_plot_sub[[target]+important_features2[:10]+['buyer_retrate_gt01']].corr(method='spearman'))
    plt.savefig(
        os.path.join(path_plot_dir, 'model_clustermap.png'),
        bbox_inches='tight', dpi=300)
    if show_plots:
        plt.show()
    plt.gcf().clear()
    plt.clf()
    plt.cla()
    plt.close()

    print(("Pairplot of target, '{target}', top 5 important features, buyer_retrate_gt01:").format(
            target=target))
    df_pairplot = df_plot_sub[[target]+important_features2[:5]+['buyer_retrate_gt01']]
    print(df_pairplot.columns)
    ds_columns = pd.Series(df_pairplot.columns, name='column')
    ds_columns.to_csv(
        os.path.join(path_plot_dir, 'model_pairplot_index_column_map.csv'),
        header=True, index_label='index')
    df_pairplot.columns = ds_columns.index
    df_pairplot.loc[:, target] = df_pairplot[np.where(ds_columns.values == target)[0][0]]
    df_pairplot.loc[:, 'buyer_retrate_gt01'] = df_pairplot[np.where(ds_columns.values == 'buyer_retrate_gt01')[0][0]]
    df_pairplot.drop([np.where(ds_columns.values == target)[0][0]], axis=1, inplace=True)
    df_pairplot.drop([np.where(ds_columns.values == 'buyer_retrate_gt01')[0][0]], axis=1, inplace=True)
    sns.pairplot(
        df_pairplot,
        hue='buyer_retrate_gt01', diag_kind='hist', markers=['.', 'o'],
        palette=[sns.color_palette()[1], sns.color_palette()[2]],
        plot_kws={'alpha':1.0})
    plt.savefig(
        os.path.join(path_plot_dir, 'model_pairplot.png'),
        bbox_inches='tight', dpi=300)
    if show_plots:
        plt.show()
    plt.gcf().clear()
    plt.clf()
    plt.cla()
    plt.close()

    print("Summarize top 5 important features:")
    print(df_features4[important_features2[:5]].describe(include='all'))
    print()
    print("First 5 records for top 5 important features:")
    print(df_features4[important_features2[:5]].head())
    print()
    print("""Describe top 5 important features. Format:
    Feature: importance score.
    Histogram of feature values.""")
    cols_scores = df_importances[important_features2[:5]].mean().items()
    for (col, score) in cols_scores:
        # Describe feature variables.
        print(
            ("{col}:\n" +
             "    importance: {score:.3f}").format(col=col, score=score))
        # Plot histogram of feature variables.
        tfmask_gt01 = df_features3[buyer_retrate] > buyer_retrate_max
        sns.distplot(
            df_features4.loc[np.logical_not(tfmask_gt01), col], hist=True, kde=False, norm_hist=False,
            label='buyer_retrate_lt01', color=sns.color_palette()[1])
        sns.distplot(
            df_features4.loc[tfmask_gt01, col], hist=True, kde=False, norm_hist=False,
            label='buyer_retrate_gt01', color=sns.color_palette()[2])
        plt.title('Feature value histogram')
        plt.xlabel("Feature value, '{ftr}'".format(ftr=col))
        plt.ylabel('Number of feature values')
        plt.legend(loc='upper left')
        if show_plots:
            plt.show()
        plt.gcf().clear()
        plt.clf()
        plt.cla()
        plt.close()

    time_stop = time.perf_counter()
    print("Time elapsed (sec) = {diff:.1f}".format(diff=time_stop-time_start))
    print()
    ########################################
    print('#'*80)
    print("""Tune model hyperparameters by optimizing the model score
    with cross validation. Model scores are R^2,
    the coefficient of determination.
    """)
    time_start = time.perf_counter()

    print("Progress:", end=' ')
    size_data = len(df_features4)
    size_sub = min(len(df_features4), int(2e3))
    frac_test = 0.2
    replace = False
    nest_list = [10, 30, 100, 300]
    n_scores = 10
    nest_scores = list()
    for (inum, n_est) in enumerate(nest_list):
        estimator = sk_ens.ExtraTreesRegressor(n_estimators=n_est, n_jobs=-1)
        scores = list()
        for _ in range(0, n_scores):
            idxs_sub = np.random.choice(a=size_data, size=size_sub, replace=replace)
            (ftrs_train, ftrs_test,
             trg_train, trg_test) = sk_cv.train_test_split(
                df_features4.values[idxs_sub], ds_target.values[idxs_sub],
                test_size=frac_test)
            estimator.fit(X=ftrs_train, y=trg_train)
            scores.append(estimator.score(
                    X=ftrs_test, y=trg_test))
        nest_scores.append([n_est, scores])
        print("{frac:.0%}".format(frac=(inum+1)/len(nest_list)), end=' ')
    print('\n')

    nest_pctls = np.asarray(
        [np.append(tup[0], np.percentile(tup[1], q=[5,50,95]))
         for tup in nest_scores])
    plt.plot(
        nest_pctls[:, 0], nest_pctls[:, 2],
        marker='.', color=sns.color_palette()[0],
        label='50th pctl score')
    plt.fill_between(
        nest_pctls[:, 0],
        y1=nest_pctls[:, 1],
        y2=nest_pctls[:, 3],
        alpha=0.5, color=sns.color_palette()[0],
        label='5-95th pctls of scores')
    plt.title("Model score vs number of estimators")
    plt.xlabel("Number of estimators")
    plt.ylabel("Model score")
    plt.legend(loc='lower left')
    plt.savefig(
        os.path.join(path_plot_dir, 'model_tune_nestimators.png'),
        bbox_inches='tight', dpi=300)
    if show_plots:
        plt.show()
    plt.gcf().clear()
    plt.clf()
    plt.cla()
    plt.close()

    time_stop = time.perf_counter()
    print("Time elapsed (sec) = {diff:.1f}".format(diff=time_stop-time_start))
    print()
    ########################################
    print('#'*80)
    print("""Test significance of predictions by shuffling the target values.
    Model scores are r^2, the coefficient of determination.
    """)
    n_estimators = 50 # from tuning curve
    time_start = time.perf_counter()

    # Calculate significance of score.
    estimator = sk_ens.ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=-1)
    utils.utils.calc_score_pvalue(
        estimator=estimator, df_features=df_features4, ds_target=ds_target,
        n_iter=20, size_sub=None, frac_test=0.2,
        replace=False, show_progress=True, show_plot=True)
    print()

    time_stop = time.perf_counter()
    print("Time elapsed (sec) = {diff:.1f}".format(diff=time_stop-time_start))
    print()
    ########################################
    print('#'*80)
    print("""Predict target values with cross-validation,
    plot actual vs predicted and score.
    """)
    n_estimators = 50 # from tuning curve
    time_start = time.perf_counter()

    print("Progress:", end=' ')
    n_folds = 5
    estimator = sk_ens.ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=-1)
    kfolds = sk_cv.KFold(n=len(df_features4), n_folds=n_folds, shuffle=True)
    ds_predicted = pd.Series(index=ds_target.index, name=target+'_pred')
    idxs_pred = set()
    for (inum, (idxs_train, idxs_test)) in enumerate(kfolds):
        if not idxs_pred.isdisjoint(idxs_test):
            raise AssertionError(
                ("Program error. Each record must be predicted only once.\n" +
                 "Required: idxs_pred.isdisjoint(idxs_test)"))
        idxs_pred.update(idxs_test)
        ftrs_train = df_features4.values[idxs_train]
        ftrs_test  = df_features4.values[idxs_test]
        trg_train  = ds_target.values[idxs_train]
        trg_test   = ds_target.values[idxs_test]
        estimator.fit(X=ftrs_train, y=trg_train)
        ds_predicted.iloc[idxs_test] = estimator.predict(X=ftrs_test)
        print("{frac:.0%}".format(frac=(inum+1)/n_folds), end=' ')
    print('\n')

    score = sk_met.r2_score(
        y_true=ds_target, y_pred=ds_predicted)
    print("Model score = {score:.3f}".format(score=score))
    utils.utils.plot_actual_vs_predicted(
        y_true=ds_target.values, y_pred=ds_predicted.values,
        loglog=False, xylims=(-1.1, 1.1),
        path=os.path.join(path_plot_dir, 'model_actual_vs_predicted.jpg'))

    print("""`features.pkl`, `estimator.pkl`: Save features and estimator.
    """)
    path_ftr = os.path.join(path_data, 'features.pkl')
    with open(path_ftr, mode='wb') as fobj:
        pickle.dump(obj=df_features4.columns, file=fobj)
    path_est = os.path.join(path_data, 'estimator.pkl')
    with open(path_est, mode='wb') as fobj:
        pickle.dump(obj=estimator, file=fobj)

    time_stop = time.perf_counter()
    print("Time elapsed (sec) = {diff:.1f}".format(diff=time_stop-time_start))
    print()
    ########################################
    return None

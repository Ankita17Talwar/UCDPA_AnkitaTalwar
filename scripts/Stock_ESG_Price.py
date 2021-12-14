import pandas as pd
import numpy as np
import yfinance as yf
import sys
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import re
import matplotlib.pyplot as plt
import seaborn as sns

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class ImportData:

    def __init__(self, abs_file_name=None):

        if abs_file_name is None:
            self.file_name = '../dataset/orig_ind_nifty5list.csv'
        else:
            self.file_name = abs_file_name

        # create Empty Data Frame to Load Stock list data from File
        self.stock_df = pd.DataFrame()

        # Create Empty Data Frame to load esg data for all stocks from yahoo finance
        self.esg_data = pd.DataFrame()
        # Create data frame to hold Subset of Sustainability data
        self.esg_df = pd.DataFrame()

        # Create Data frame to store price info attributes of stocks
        self.stock_info = pd.DataFrame()

        self.stock_info_subset = pd.DataFrame()

    def load_file(self):
        """
        Loads NIFTY 500 stock CSV file to Dataframe : stock_df
        Columns in File :'Company Name', 'Industry', 'Symbol', 'Series', 'ISIN Code'

        :return:
        """

        # read csv file
        self.stock_df = pd.read_csv(self.file_name)
        print(self.stock_df.head())

        print(self.stock_df.columns)
        # 'Company Name', 'Industry', 'Symbol', 'Series', 'ISIN Code'

    def create_ticker_list(self):
        """
        It creates the list of Tickers from Stock DataFrame (NIFTY 500).
        Assumption Stock df(/file) contains 'Symbol' column.
        It appends '.ns' to symbol to make it compatible with expected format for yfinance

        :return: list of tickers (ticker_list)
        """
        self.stock_df['Symbol'] = self.stock_df['Symbol'] + '.ns'
        # create list of symbols from stock_df
        ticker_list = self.stock_df['Symbol'].tolist()

        return ticker_list

    def get_esg_data(self, ticker_list):

        """
        calls yfinance to fetch ESG and stock info data

        :param ticker_list: list of stocks for which we need to fetch data from yfinance
        :return:
        """

        # for each ticker in ticker list get esg and stock info data
        for ticker in ticker_list:
            print(ticker)
            # Call yahoo finance to get ticker name
            ticker_name = yf.Ticker(ticker)
            # print(ticker_name)
            # print(type(ticker_name))

            try:
                if ticker_name.sustainability is not None:
                    # print('not none')

                    # Get Sustainability data and Transpose DataFrame using '.T'
                    # It returns 27 attributes of tickers
                    ticker_df = ticker_name.sustainability.T
                    # print(ticker_df)

                    # add symbol column to dataframe: ticker_df
                    ticker_df['symbol'] = ticker

                    # Append to ESG data Frame : Sustainability and Ticker Symbol
                    self.esg_data = self.esg_data.append(ticker_df)
                    # print(self.esg_data)

                    # add delay to avoid sending too many request : time 3 seconds
                    time.sleep(3)
            except():
                print(f'{ticker} did not return ESG data from Yahoo finance')
                pass

            # Fetch Stock Info Data from yfinance
            try:
                ticker_info = ticker_name.info
                # print(ticker_info)

                # parse the dict response into a DataFrame
                ticker_df = pd.DataFrame.from_dict(ticker_info.items()).T

                # sets column names to first row of DataFrame
                ticker_df.columns = ticker_df.iloc[0]

                # drop first line - containing column names
                ticker_df = ticker_df.drop(ticker_df.index[0])

                # append data to stock_info DataFrame
                self.stock_info = self.stock_info.append(ticker_df)

                # add delay to avoid sending too many request : time 3 seconds
                time.sleep(3)
                print(f'{ticker} + Complete')
            except (IndexError, ValueError) as e:
                print(f'{ticker} + Data Not Found')

        self.esg_data.to_csv('../output_files/out_esg_info/ESG_yfin_Data.csv')

        # Lets strip the required ESG columns from esg_data_frame :
        self.esg_df = self.esg_data[['symbol', 'socialScore', 'governanceScore', 'environmentScore', 'totalEsg',
                                     'percentile', 'peerCount', 'peerGroup', 'highestControversy', 'esgPerformance']]
        print(self.esg_df.head())

        # Store result in csv for reference
        self.esg_df.to_csv('../output_files/out_esg_info/Stocks_ESG_data.csv')

        #######

        self.stock_info.to_csv('../output_files/out_esg_info/Stock_Info_yFin_Data.csv')

        self.stock_info_subset = self.stock_info[['symbol', 'sector', 'previousClose', 'sharesOutstanding']]
        self.stock_info_subset.to_csv('../output_files/out_esg_info/Stock_info_data.csv')

    def transform_data(self, ticker_list):
        """
        Identify the Stocks for which we haven't received data from yfinance :no_esg_data_df
        Derive new column in Stock info Dataframe: market cap
        Transform data : ESG and Stock Info. Merge dataframes and drop null data
        Convert Datatypes, Find unique list of sectors
        :param ticker_list:
        :return: stk_data_df (merged and transformed Dataframe)
        """
        # Identify Stocks for which we haven't received Sustainability data from yahoo finance
        # esg_data contains data for stocks for which we have received response from yfinance
        # find the difference in ticker_list and esg_data[symbol]
        esg_data_list = self.esg_data['symbol'].tolist()
        no_esg_data_stocks = [symbol for symbol in ticker_list if symbol not in esg_data_list]
        print('Stocks with no ESG data')
        print(len(no_esg_data_stocks))
        print(no_esg_data_stocks)

        no_esg_data_df = pd.DataFrame(no_esg_data_stocks)

        no_esg_data_df.to_csv('../output_files/out_esg_info/No_ESG_DATA.csv')

        # add new column market_cap to stock_info :
        # Market Capitalization Formula = Current Market Price per share * Total Number of Outstanding Shares.
        self.stock_info_subset['market_cap'] = self.stock_info_subset['previousClose'] * \
                                               self.stock_info_subset['sharesOutstanding']

        self.stock_info_subset['symbol'] = self.stock_info_subset['symbol'].str.replace('.NS', '.ns')

        # Merge esg_df (containing ESG data of stocks) and stock_info_subset
        stk_data_df = self.stock_info_subset.merge(self.esg_df, how='left', on='symbol')

        # print(stk_data_df)
        stk_data_df.to_csv('../output_files/out_esg_info/Merged_DF.csv')

        # stk_data_df will contain few rows where ESG score is not present - stocks captured in no_esg_data
        # we will drop the rows with NAN's (null ESG data)
        stk_data_df.dropna(subset=['socialScore', 'governanceScore', 'totalEsg', 'environmentScore'], inplace=True)
        # print(stk_data_df)
        stk_data_df.to_csv('../output_files/out_esg_info/Transformed_data.csv')

        stk_data_df['totalEsg'] = pd.to_numeric(stk_data_df['totalEsg'], errors='ignore')
        stk_data_df['market_cap'] = pd.to_numeric(stk_data_df['market_cap'], errors='ignore')

        sector_list = stk_data_df['sector'].unique().tolist()
        print(sector_list)

        return stk_data_df

    def plot_group_size(self, df, gp_column, title):
        """
        Plot number of stocks per sector
        :param df: Dataframe
        :param gp_column: column on which grouping is required
        :param title: Title of Plot
        :return:
        """

        # group by gp_column
        result = df.groupby([gp_column]).size()
        fig, ax = plt.subplots(figsize=(10, 6))

        # plot the result : Number of stocks in each sector (in NIFTY50)
        # add data to plot
        sns.barplot(x=result.index, y=result.values, ax=ax)

        ax.set_xticklabels(result.index, rotation=20)

        # set title and label
        ax.set_title(title, color='gray')
        ax.set_xlabel('Sectors')
        ax.set_ylabel('No. of Companies')

        # show plot
        plt.show()

    def plot_bar(self, axes, X, Y, yerr=None, label=None, title=None, xlabel=None, ylabel=None,
                 xticklabels=None, rotation=45, plt_save='P', fig=None, plot_name='None'):
        """
        Generic Function to create bar plots
        :param axes: axes for plot. Mandatory attribute
        :param X: data for X axis . Mandatory attribute
        :param Y: data for Y axis . Mandatory attribute
        :param fig: figure for the plot. Optional attribute
        :param label: label of plots. Optional attribute
        :param title: Title of Graph. Optional attribute
        :param xlabel: x-ais label. Optional attribute
        :param ylabel: y-axis label. Optional attribute
        :param xticklabels: xtick labels. Optional attribute
        :param rotation: rotation for xticks labels. Optional attribute
        :param yerr: error in plot. Optional attribute
        :param plt_save: Save or plot graph. Default - P (Plot). Domain : P , S
        :param plot_name: Name of plot file to save
        :return:
        """

        if yerr:
            if label:
                axes.bar(X, Y, yerr=yerr, label=label)
            else:
                axes.bar(X, Y, yerr=yerr)
        else:
            if label:
                axes.bar(X, Y, label=label)
            else:
                axes.bar(X, Y)

        if title:
            axes.set_title(title)

        if xlabel:
            axes.set_xlabel(xlabel)

        if ylabel:
            axes.set_ylabel(ylabel)

        if xticklabels:
            axes.xaxis.set_ticks(xticklabels)
            axes.set_xticklabels(xticklabels, rotation=rotation)

        if label:
            axes.legend()

        if plt_save == 'P':
            plt.show()
        elif plt_save == 'S':
            fl_nm = 'plots/' + plot_name + '.png'
            fig.savefig(fl_nm)
            # plt.show()

        # return axes

    def plot_line(self, axes, fig, X, Y, label=None, title=None, color=None, plt_save='P',
                  plot_name=None, marker=None, twin_axis=None, x_label=None, y_label=None, loc=None):
        """
        Generic Function to plot line graph
        :param axes: axes for plot. Mandatory attribute
        :param fig: figure for the plot. Optional attribute
        :param X: data for X axis . Mandatory attribute
        :param Y: data for Y axis . Mandatory attribute
        :param label: label of plots. Optional attribute
        :param title: Title of Graph. Optional attribute
        :param color: Optional
        :param plt_save: Save or plot graph. Default - P (Plot). Domain : P , S
        :param plot_name: Name of plot file to save
        :param marker: Optional . Default 'o'
        :param twin_axis: To identify if Twin axes is required
        :param x_label: label for x axis. Optional
        :param y_label : label for y axis. Optional
        :param loc : Location of legend. Optional
        :return:
        """

        if marker is None:
            marker = 'o'

        if label and color:
            axes.plot(X, Y, label=label, color=color, marker=marker)
            axes.legend()
        elif label:
            axes.plot(X, Y, label=label, marker=marker)
            axes.legend()
        elif color:
            axes.plot(X, Y, color=color, marker=marker)

        if title:
            axes.set_title(title)

        if twin_axis:
            axes.tick_params(twin_axis, color=color)
            if loc:
                plt.legend(loc=loc)

        if x_label:
            axes.set_xlabel(x_label)
        if y_label:
            axes.set_ylabel(y_label)

        if plt_save == 'P':
            plt.show()
        elif plt_save == 'S':
            print('Save')
            fl_nm = 'plots/' + plot_name + '.png'
            fig.savefig(fl_nm)
            # plt.show()

    def plot_hist(self, axes, fig, data, plot_name, bins=None, xlabel=None, ylabel=None, title=None):

        """
        Generic Function to plot Histogram
        :param axes: For the Plot. Mandatory
        :param data: Mandatory
        :param bins: Number of Bins. Optional. Defaults used if not passed
        :param xlabel: x-ais label. Optional attribute
        :param ylabel: y-axis label. Optional attribute
        :return:
        """

        if bins:
            axes.hist(data, bins=bins, histtype='bar', edgecolor='black', color='blue')
        else:
            axes.hist(data, histtype='bar', edgecolor='black', color='blue')

        if xlabel:
            axes.set_xlabel(xlabel)

        if ylabel:
            axes.set_ylabel(ylabel)

        if title:
            axes.set_title(title)

        fl_nm = 'plots/' + plot_name + '.png'
        fig.savefig(fl_nm)

    def box_plot(self, X, Y):
        '''
        Horizontal Box Plot

        :param X: Mandatory
        :param Y: Mandatory
        :return:
        '''
        boxplot = sns.boxplot(x=X, y=Y)
        boxplot = sns.stripplot(x=X, y=Y, marker='o', alpha=0.3, color='red')
        boxplot.axes.set_title('ESG Score Distribution Per Sector')

        plt.show()

    def controversy_esg_rel(self, transformed_df):
        print('Relation between Controversy and ESG Score')

        sns.lmplot(x="market_cap", y="totalEsg", line_kws={'color': 'g'}, data=transformed_df, lowess=True,
                   height=3.5, aspect=1.5)
        ax = plt.gca()
        ax.set_title('market_cap Vs totalEsg ')
        plt.show()

        sns.regplot(x="market_cap", y="totalEsg", data=transformed_df, line_kws={'color': 'g'}, lowess=True)
        ax = plt.gca()
        ax.set_title('market_cap Vs totalEsg ')
        plt.show()


def main():
    if len(sys.argv) > 2:
        abs_file_name = sys.argv[1]
        import_obj = ImportData(abs_file_name)
    else:
        import_obj = ImportData()

    ###############################################
    # 1. Load  Nifty Stock csv
    ###############################################
    import_obj.load_file()

    ###############################################
    # 2.create ticker list
    ###############################################
    ticker_list = import_obj.create_ticker_list()
    print(type(ticker_list))

    ###############################################
    # 3.Get ESG and Stock Info data for stocks from yahoo finance
    ###############################################
    import_obj.get_esg_data(ticker_list)

    ###############################################
    # 4.Transform ESG data
    ###############################################
    transformed_df = import_obj.transform_data(ticker_list)

    ###############################################
    # 5.Plot size of each sector :
    # number of stocks in each sector in Nifty50
    ###############################################
    import_obj.plot_group_size(transformed_df, 'sector', "Number of Companies per Sector(NIFTY500)")

    ###############################################
    # 6.plot sector-wise mean and standard deviation of ESG Score
    ###############################################
    sector_list = transformed_df['sector'].unique().tolist()
    distinct_count = len(sector_list)
    counter = 1
    plt.style.use('seaborn-colorblind')
    fig, ax = plt.subplots()
    print('########################')
    print('Sector list::')
    print(sector_list)
    print('Length :', distinct_count)
    print('########################')

    for sector in sector_list:
        print(sector)
        # subset of data frame with specific sector data
        data_df = transformed_df[transformed_df['sector'] == sector]
        # if last call -add Plot specific details
        if counter == distinct_count:
            import_obj.plot_bar(ax, sector, data_df['totalEsg'].mean(), data_df['totalEsg'].std(),
                                None, "Sector Average ESG Score", "Sectors", "Average ESG Score", sector_list, 45, 'S',
                                fig, 'Sector Average ESG Score')
        else:
            print('calls to plot')
            import_obj.plot_bar(ax, sector, data_df['totalEsg'].mean(), data_df['totalEsg'].std())
            counter += 1

    ###############################################
    # 7.Plot Histogram showing Total ESG Score Distribution
    ###############################################
    print('Plot ESG Distribution- histogram')
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    # score_bin = [0, 40, 70, 100]
    score_bin = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # score_bin = [0, 20, 40, 60, 80, 100]
    import_obj.plot_hist(ax, fig, transformed_df['totalEsg'], 'ESG_Distribution', score_bin, None, 'ESG Score Bucket',
                         "ESG Score Distribution")

    print('Plot ESG Distribution per sector - Histogram')
    import_obj.box_plot(transformed_df['totalEsg'], transformed_df['sector'])

    print(" Plotting Top and Lowest Score")
    ###############################################
    # 8.Plot Top and Lowest Score in NIFTY50 Index
    ###############################################
    fig, ax = plt.subplots()
    import_obj.plot_bar(ax, "Highest Score", transformed_df['totalEsg'].max(),
                        title='Highest and Lowest ESG Score (NIFTY50)', xlabel='ESG Score',
                        xticklabels=None, rotation=None)

    import_obj.plot_bar(ax, "Lowest Score", transformed_df['totalEsg'].min(), xlabel='ESG Score',
                        xticklabels=None, rotation=None, plt_save='S', fig=fig,
                        plot_name='Highest_Lowest_NIFTY')

    ###############################################
    # 9.Plot Stocks with top5 and lowest 5 ESG Score
    ###############################################
    plt.style.use('ggplot')
    fig, ax = plt.subplots()

    data_df_top_5 = transformed_df.nlargest(n=5, columns=['totalEsg'])
    data_df_low_5 = transformed_df.nsmallest(n=5, columns=['totalEsg'])

    import_obj.plot_line(ax, fig, data_df_top_5['symbol'], data_df_top_5['totalEsg'], None,
                         'Top 5 ESG Risk Score', color='b', plt_save='S', plot_name='Top_5_H_ESG', marker='v')

    fig, ax = plt.subplots()
    import_obj.plot_line(ax, fig, data_df_low_5['symbol'], data_df_low_5['totalEsg'], None,
                         'Lowest 5 ESG Risk Score', color='b', plt_save='S', plot_name='Top_5_L_ESG',
                         x_label='Stocks', y_label='ESG Score')

    # print('Scatter plot between HighestControvery and ESG Score')
    # sns.lmplot(x="highestControversy", y="totalEsg", data=transformed_df)
    # ax = plt.gca()
    # ax.set_title('HighestControversy Vs TotaESG Score')
    # plt.show()

    ###############################################
    # 10.Plot ESG Score and Total MarketCap (using twin y- axis)
    # plot for companies with high ESG Risk Score
    ###############################################
    plt.style.use('default')
    fig, ax = plt.subplots()
    import_obj.plot_line(ax, fig, data_df_top_5['symbol'], data_df_top_5['market_cap'], 'Market Cap', None,
                         'b', None, None, 'o', 'y', None, "MarketCap")

    ax2 = ax.twinx()
    import_obj.plot_line(ax2, fig, data_df_top_5['symbol'], data_df_top_5['totalEsg'], 'Total ESG Score',
                         'ESG/Market Cap', 'r', 'S', 'ESG_MKT_top_5_NIFTY', 'v', 'y', 'Stock', 'ESG Score', loc=3)
    #########################################################################

    import_obj.controversy_esg_rel(transformed_df)

    ###########################################################################
    print('Relation between market_cap and ESG Score')

    sns.lmplot(x="market_cap", y="totalEsg", line_kws={'color': 'g'}, data=transformed_df, lowess=True,
               height=3.5, aspect=1.5)
    ax = plt.gca()
    ax.set_title('market_cap Vs totalEsg ')
    plt.show()


if __name__ == '__main__':
    main()

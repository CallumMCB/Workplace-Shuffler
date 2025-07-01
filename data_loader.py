import pandas as pd
import numpy as np
from typing import List, Union

class WorkDataLoader:
    """
    Loader for census and workplace data, optionally for multiple LADs.
    If `lad_name` is a string or list of strings, filters to those LAD(s).
    If `lad_name` is None, loads data for all LADs present in the files.
    """
    def __init__(self, lad_name: Union[str, List[str], None] = None):
        # Normalize lad_name into a list or None
        self.all_lads = None
        if lad_name is None:
            self.lad_names = None
        elif isinstance(lad_name, str):
            self.lad_names = [lad_name]
        else:
            self.lad_names = list(lad_name)

        self.data_folder = 'data'
        self.employees_folder = self.data_folder + '/employees'

    def load_geog_hierarchy(self):
        # 2011 hierarchy (IZ)
        df2011 = pd.read_csv(
            self.data_folder + '/2011_geography_hierarchy.csv',
            encoding='latin1', usecols=['LA_Name', 'IZ2011_Code']
        ).rename(columns={'LA_Name':'lad', 'IZ2011_Code':'intermediate_zone'})
        df2011.set_index('lad', inplace=True)

        # 2022 hierarchy (OA)
        df2022 = pd.read_csv(
            self.data_folder + '/oa_msoa_lad_regions.csv', usecols=['area', 'lad']
        ).set_index('lad')

        if self.lad_names is not None:
            df2011 = df2011.loc[df2011.index.intersection(self.lad_names)]
            df2022 = df2022.loc[df2022.index.intersection(self.lad_names)]
        else:
            self.all_lads = df2022.index.unique()

        return df2011, df2022

    def load_area_iz11(self):
        df_area_iz11 = pd.read_csv(
            self.data_folder + '/area_iz11.csv'
        ).set_index('area')
        return df_area_iz11

    def load_centroids(self, df2011, df2022):
        # read all centroids
        iz_all = pd.read_csv(
            self.data_folder + '/iz2011_centroids_en.csv', index_col='intermediate_zone'
        )
        oa_all = pd.read_csv(
            self.data_folder + '/oa_centroids_en.csv', index_col='area'
        )

        # filter to LAD(s)
        if self.lad_names is None:
            iz = iz_all
            oa = oa_all
        else:
            iz_list = df2011.loc[self.lad_names]['intermediate_zone'].unique()
            oa_list = df2022.loc[self.lad_names]['area'].unique()
            iz = iz_all.loc[iz_list]
            oa = oa_all.loc[oa_list]

        return iz, oa

    def load_workplace_pop(self, iz):
        df = pd.read_csv(
            self.employees_folder + '/broad_industrial_grp_total.csv',
            index_col='intermediate_zone'
        )
        df.columns = (
            df.columns
            .str.split(" : ", n=1).str[0]  # ["18", "05", ...]
            .astype(int)  # [18, 5, ...]
            .astype(np.uint8)  # array of dtype uint8
        )
        return df.loc[df.index.intersection(iz.index)]

    def load_industry(self, oa):
        df = pd.read_csv(
            self.data_folder + '/industry.csv', index_col=0
        )
        df.drop(columns = 'All people aged 16 and over in employment the week before the census', inplace=True)
        df.columns = (
            df.columns
            .str.split(" : ", n=1).str[0]  # ["18", "05", ...]
        )
        return df.loc[df.index.intersection(oa.index)]


    def load_distance_age(self, oa):
        df = pd.read_csv(
            self.data_folder + '/distance_age.csv', index_col=[0,1]
        )
        df.drop(columns = 'All people aged 16+ in employment the week before the census (including full-time students if they gave a work address as the address they primarily travel - for work or study)', inplace=True)
        # filter top-level index by oa
        return df.loc[df.index.get_level_values(0).isin(oa.index)]

    def load_method_age(self, oa):
        df = pd.read_csv(
            self.data_folder + '/travel_method_age.csv', index_col=[0,1]
        )
        return df.loc[df.index.get_level_values(0).isin(oa.index)]

    def load_all(self):
        geog2011, geog2022 = self.load_geog_hierarchy()
        iz_centroids, oa_centroids = self.load_centroids(geog2011, geog2022)
        oa_iz = self.load_area_iz11()
        workplace_pop = self.load_workplace_pop(iz_centroids)
        industry = self.load_industry(oa_centroids)
        distance_age = self.load_distance_age(oa_centroids)
        method_age = self.load_method_age(oa_centroids)
        if self.lad_names is None:
            self.lad_names = self.all_lads

        return {
            'lads': self.lad_names,
            'iz_centroids': iz_centroids,
            'oa_centroids': oa_centroids,
            'oa_iz': oa_iz,
            'workplace_pop': workplace_pop,
            'industry': industry,
            'distance_age': distance_age,
            'method_age': method_age,
            'geog2011': geog2011,
            'geog2022': geog2022
        }

# Usage example
if __name__ == '__main__':
    # single LAD
    loader1 = WorkDataLoader('Aberdeen City')
    data1 = loader1.load_all()
    print({k: v.shape for k, v in data1.items()})

    # multiple LADs
    loader2 = WorkDataLoader(['Aberdeen City', 'Dundee City'])
    data2 = loader2.load_all()
    print({k: v.shape for k, v in data2.items()})

    # all LADs
    loader3 = WorkDataLoader()
    data3 = loader3.load_all()
    print({k: v.shape for k, v in data3.items()})

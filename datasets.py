import os
import numpy as np
import pandas as pd
import paddle
from paddle.io import Dataset
from sklearn.model_selection import train_test_split


class Scaler(object):
    """
    Desc: Normalization utilities
    """

    def __init__(self):
        self.mean = np.array([0.])
        self.std = np.array([1.])

    def fit(self, data):
        """
        Desc:
            Fit the data
        Args:
            data:
        Returns:
            None
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform(self, data):
        """
        Desc:
            Transform the data
        Args:
            data:
        Returns:
            The transformed data
        """
        mean = paddle.to_tensor(self.mean) if paddle.is_tensor(data) else self.mean
        std = paddle.to_tensor(self.std) if paddle.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        """
        Desc:
            Restore to the original data
        Args:
            data: the transformed data
        Returns:
            The original data
        """
        mean = paddle.to_tensor(self.mean) if paddle.is_tensor(data) else self.mean
        std = paddle.to_tensor(self.std) if paddle.is_tensor(data) else self.std
        return (data * std) + mean

    def inverse_transform_y(self, data):
        mean = paddle.to_tensor(self.mean[-3:]) if paddle.is_tensor(data) else self.mean
        std = paddle.to_tensor(self.std[-3:]) if paddle.is_tensor(data) else self.std
        return (data * std) + mean


def Split_csv(file, ratio_split=0.1):
    if isinstance(file, str):  # only str
        df = pd.read_csv(file, parse_dates=['DATATIME'], infer_datetime_format=True, dayfirst=True,
                         dtype={'WINDDIRECTION': np.float64, 'HUMIDITY': np.float64,
                                'PRESSURE': np.float64})
        df.sort_values(by='DATATIME', inplace=True, ignore_index=True)  # sorted
        df['DATATIME'] = pd.to_datetime(df['DATATIME']).astype('int64') / 10 ** 9
        df.replace(to_replace=np.nan, value=-99999, inplace=True)  # clean Nan

    elif isinstance(file, list):
        df_list = []
        idx = 1
        for file_path in file:
            df = pd.read_csv(file_path, parse_dates=['DATATIME'], infer_datetime_format=True, dayfirst=True,
                             dtype={'WINDDIRECTION': np.float64, 'HUMIDITY': np.float64,
                                    'PRESSURE': np.float64}).assign(PLANTID=idx)
            df_list.append(df)
            idx += 1

        df = pd.concat(df_list)
        df.sort_values(by='DATATIME', inplace=True, ignore_index=True)  # sorted
        df['DATATIME'] = pd.to_datetime(df['DATATIME']).astype('int64') / 10 ** 9
        df.replace(to_replace=np.nan, value=-99999, inplace=True)  # clean Nan
    else:
        assert False
    train_df, val_df = train_test_split(df, test_size=ratio_split, shuffle=False)
    return train_df, val_df


def Split_dataframe(df, ratio_split=0.5):
    train_df, val_df = train_test_split(df, train_size=ratio_split, shuffle=False)
    return train_df, val_df


class WPFDataset(paddle.io.Dataset):
    def __init__(self, data,
                 size=None,
                 scale=True,
                 scaler = None):
        """

        Args:
            data: path or dataframe
            scale: transform
        """
        super().__init__()
        assert isinstance(data, str) or isinstance(data, pd.DataFrame) or isinstance(data, list)
        if isinstance(data, str) or isinstance(data, list):
            self.__get_dataframe__(data)
        else:
            self.df = data

        self.scaler = scaler
        if size is None:
            self.input_len = 24 * 6
            self.label_len = 24 * 6
            self.output_len = 24 * 6
        else:
            self.input_len = size[0]
            self.label_len = size[1]
            self.output_len = size[2]

        self.scale = scale
        self.data_x, self.data_y = self.__get_data__()

    def __get_dataframe__(self, file):
        if isinstance(file, str):  # only str
            self.df = pd.read_csv(file, parse_dates=['DATATIME'], infer_datetime_format=True, dayfirst=True,
                                  dtype={'WINDDIRECTION': np.float64, 'HUMIDITY': np.float64,
                                         'PRESSURE': np.float64})
            self.df.sort_values(by='DATATIME', inplace=True, ignore_index=True)  # sorted
            self.df['DATATIME'] = pd.to_datetime(self.df['DATATIME']).astype('int64') / 10 ** 9
            self.df.replace(to_replace=np.nan, value=-99999, inplace=True)  # clean Nan

        elif isinstance(file, list):
            df_list = []
            idx = 1
            for file_path in file:
                df = pd.read_csv(file_path, parse_dates=['DATATIME'], infer_datetime_format=True, dayfirst=True,
                                 dtype={'WINDDIRECTION': np.float64, 'HUMIDITY': np.float64,
                                        'PRESSURE': np.float64}).assign(PLANTID=idx)
                df_list.append(df)
                idx += 1

            self.df = pd.concat(df_list)
            self.df.sort_values(by='DATATIME', inplace=True, ignore_index=True)  # sorted
            self.df['DATATIME'] = pd.to_datetime(self.df['DATATIME']).astype('int64') / 10 ** 9
            self.df.replace(to_replace=np.nan, value=-99999, inplace=True)  # clean Nan

    def __get_data_x_y__(self):
        self.length = len(self.df)

        # 设置边界
        border1 = 0
        border2 = self.length

        data_x = self.df[border1:border2]
        data_y = self.df[border1:border2]

        if self.scale:
            # normalization
            if self.scaler is None:
                scaler = Scaler()
                self.scaler = scaler
                scaler.fit(self.df.values)
                data_x.values[:] = scaler.transform(data_x.values)
                data_y.values[:] = scaler.transform(data_y.values)
            else:
                data_x.values[:] = self.scaler.transform(data_x.values)
                data_y.values[:] = self.scaler.transform(data_y.values)
        else:
            data_x.values = data_x.values
            data_y.values = data_y.values
        return data_x, data_y

    def __get_data__(self):
        data_x, data_y = self.__get_data_x_y__()
        return data_x, data_y

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.input_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.output_len

        x_colunms = ['DATATIME', 'WINDSPEED', 'PREPOWER', 'WINDDIRECTION', 'TEMPERATURE',
                     'HUMIDITY', 'PRESSURE']
        y_colunms = ['ROUND(A.WS,1)', 'ROUND(A.POWER,0)', 'YD15']

        seq_x = self.data_x.loc[:, x_colunms].iloc[s_begin:s_end, :].values
        seq_y = self.data_y.loc[:, y_colunms].iloc[r_begin:r_end, :].values
        return seq_x, seq_y

    def get_scaler(self):
        return self.scaler

    def __len__(self):
        return len(self.data_x) - self.input_len - self.output_len + 1

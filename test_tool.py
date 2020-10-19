import pandas as pd
import numpy as np
import datetime
import glob
import os
import re
import yaml
import codecs

# with open('./drive/Shared drives/AI活用型非接触排泄予測システム研究開発/7_ツール/config.yaml', 'r') as yml:
# with open('config.yaml', 'r') as yml:
#     config = yaml.load(yml)
# コンフィグレーション
# INPUT_DIR_PATH = config['data_merge']['input_dir_path']  # 読み込みデータへのディレクトリパス. ({指定したパス}/{被験者名}/*.csvとなる)
# EXPORT_DIR_PATH = config['data_merge']['output_dir_path']     #出力ファイルパス
# SHARP_MIN = config['data_merge']['sharp_min']    # データの間引き時間(1以上の分指定)
# MOVE_AVE_WINDOW_SIZE = config['data_merge']['move_ave_window_size']    # 移動平均ウィンドウサイズ
# ANOMALY_DETECT_SIZE = config['data_merge']['anomaly_detect_size']     # 異常検知サイズ
INPUT_DIR_PATH = "./drive/Shared drives/AI活用型非接触排泄予測システム研究開発/7_ツール/データマージツール/datas/"
EXPORT_DIR_PATH = "./drive/Shared drives/AI活用型非接触排泄予測システム研究開発/7_ツール/データマージツール/result/"
MOVE_AVE_WINDOW_SIZE = 5    # 移動平均ウィンドウサイズ
ANOMALY_DETECT_SIZE = 5     # 異常検知サイズ
START_TIME = '9:00:00'
FIN_TIME = '19:00:00'

# 以下読み込むデータのヘッダー
# 以下で，読み込むファイルの整合性をチェックする。
MYBEAT_RRI_COLUMNS = ['time', 'RRI', 'temperature', 'Acc X', 'Acc Y', 'Acc Z', 'HF', 'LF/HF', 'LF ratio', 'activity', 'HR']
MYBEAT_HR_COLUMNS = ['time', 'HR', 'temperature', 'Acc X', 'Acc Y', 'Acc Z', 'activity']
FITBIT_COLUMNS = ['time', 'HR']
ECHO_COLUMNS = ['time', 'urination', 'echo']
URINATION_COLUMNS = ['time', 'urination']
TEMP_COLUMNS = ['time', 'temperature', 'humidity']
MYBEAT_3_ANALYSIS_DATA = ['Time', 'RRI', 'temperature', 'Acc_x', 'Acc_y', 'Acc_z', 'HR_Instant', 'Body Motion']
MYBEAT_3_TIME_ANALYSIS_DATA = ['Time', 'SDNN', 'RMSSD', 'CVRR', 'NN50', 'pNN50', 'AC', 'DC', 'HR_Ave']
MYBEAT_3_FREQUENCY_ANALYSIS_DATA = ['Time', 'LF', 'HF', 'VLF', 'ULF', 'LF/HF', 'LF/(LF+HF)', 'LFnorm', 'HFnorm', 'TotalPower']


class MybeatDateException(Exception):
    '''
    MyBeatデータの日付が不正
    '''
    pass


class ExamDataFrame():
    '''
    データ型基底クラス(基本変更不要)
    '''
    def __init__(self, data):
        self.df = data
        exclude_cols = ['time', 'Time']
        self._feature_cols = [col for col in self.df.columns if col not in exclude_cols]

    def data_shaping(self):
        '''
        データを整形する．本クラスを継承するすべてのデータは本関数で
            ①データ行数の調整
            ②外れ値の補間
            ③ヘッダー名修正
        を処理した後でマージされる。

        Args:
            なし
        Returns:
            なし
        Raise:
            MybeatDateException
        '''
        self._add_time_columns()
        # time列をdatetime型へ変換
        self._convert_to_datetime()
        # 重複日付の削除
        self._delete_duplicate()
        self._fix_header_name()
        # 日付を文字列に変換
        self.df.time = self.df.time.astype(str)

    def _add_time_columns(self):
        '''
        日付の入った'Time'列があるデータから
        時刻のみの'time'列を作成する
        '''
        if 'Time' in self.df:
            # 日付を確認する
            start_time = self.df['Time'].str.extract(r'(\d+/\d+/\d+)', expand=False).head(1).values[0]
            end_time = self.df['Time'].str.extract(r'(\d+/\d+/\d+)', expand=False).tail(1).values[0]
            if start_time != end_time:
                # 複数の日付が存在すればエラー
                raise MybeatDateException("failed.\n\n[ERROR]: MybeatData has multiple dates.")
            self.df.insert(0, 'time', self.df['Time'].str.extract(r'(\d+:\d+:\d+)', expand=False))
            self.df = self.df.drop('Time', axis=1)

    def _convert_to_datetime(self):
        '''
        時間列の文字列をフォーマット指定でdatetimeに変換する
        フォーマットが異なる場合，各データクラスでオーバーライドすること。
        '''
        self.df['time'] = pd.to_datetime(self.df['time'], format='%H:%M:%S')

    def _delete_duplicate(self):
        '''
        重複した時間行を削除する
        '''
        self.df = self.df.drop_duplicates(subset='time', keep='last')

    def _fix_header_name(self):
        '''
        ヘッダー名を他のデータと被らないよう変更する
        ヘッダー名が重複するデータクラスでオーバーライドすること、
        '''
        pass

    # def _fill_sec(self):
    #     '''
    #     時間データを1秒間隔に引き延ばす
    #     生データマージ用関数
    #     '''
    #     pass


class HrData(ExamDataFrame):
    '''
    HRを含むデータ型基底クラス(mybeat, fitbitクラス等で継承する)
    HRデータ独自の異常値判定を実装する。
    '''
    def __init__(self, data):
        super().__init__(data)


class NonHrData(ExamDataFrame):
    '''
    HRを含まないデータ型基底クラス(尿意, 温度計クラス等で継承する)
    最もスタンダードな外れ値補完(異常値の1つ上のデータをコピー)を実装する。
    '''
    def __init__(self, data):
        super().__init__(data)


class MybeatData(HrData):
    '''
    myBeatデータクラス
    他と重複するヘッダー名"HR", "temperature"を修正する
    '''
    def __init__(self, data):
        super().__init__(data)

    def _fix_header_name(self):
        '''
        ヘッダー名を他のデータと被らないよう変更する
        References:
            ExamDataFrame
        '''
        self.df = self.df.rename(columns={'HR': 'mybeat_HR'})
        self.df = self.df.rename(columns={'temperature': 'body_temperature'})


class MybeatV3AnalysisData(HrData):
    '''
    Mybeat WHS-3の解析データクラス
    '''
    def __init__(self, data):
        super().__init__(data)


class MybeatV3FrequencyAnalysisData(HrData):
    '''
    Mybeat WHS-3の周期解析データクラス
    '''
    def __init__(self, data):
        super().__init__(data)


    def _fix_header_name(self):
        '''
        ヘッダー名を他のデータと被らないよう変更する
        References:
            ExamDataFrame
        '''
        self.df = self.df.rename(columns={'LF/HF': 'LF_HF'})
        self.df = self.df.rename(columns={'LF/(LF+HF)': 'LF_(LF+HF)'})
        # self.df.time = self.df.time.astype(str)


class MybeatV3TimeAnalysisData(HrData):
    '''
    Mybeat WHS-3の時間解析データクラス
    '''
    def __init__(self, data):
        super().__init__(data)


class FitbitData(HrData):
    '''
    Fitbitデータクラス
    他と重複するヘッダー名"HR"を修正する
    '''
    def __init__(self, data):
        super().__init__(data)

    def _fix_header_name(self):
        '''
        ヘッダー名を他のデータと被らないよう変更する
        References:
            ExamDataFrame
        '''
        self.df = self.df.rename(columns={'HR': 'fitbit_HR'})


class EchoData(NonHrData):
    '''
    尿意・エコーデータクラス
    外れ値補間をオーバーライドする。
    '''
    def __init__(self, data):
        super().__init__(data)

    def _convert_to_datetime(self):
        '''
        time列が[YYYY/mm/dd HH:MM:SS]形式のため，正規表現で抽出し[HH:MM:SS]形式で上書きする。
        その後datetimeに変換する
        '''
        self.df.time = self.df['time'].str.extract(r'(\d+:\d+:\d+)', expand=False)
        self.df.time = pd.to_datetime(self.df['time'], format='%H:%M:%S')


class UrinationData(NonHrData):
    '''
    尿意データクラス
    外れ値補間をオーバーライドする。
    '''
    def __init__(self, data):
        super().__init__(data)

    def _convert_to_datetime(self):
        '''
        time列が[YYYY/mm/dd HH:MM:SS]形式のため，正規表現で抽出し[HH:MM:SS]形式で上書きする。
        その後datetimeに変換する
        '''
        self.df.time = self.df['time'].str.extract(r'(\d+:\d+:\d+)', expand=False)
        self.df.time = pd.to_datetime(self.df['time'], format='%H:%M:%S')


class TempData(NonHrData):
    '''
    温度計データクラス
    '''
    def __init__(self, data):
        super().__init__(data)


def main():
    '''
    被験者名の数だけmerge_datas()を実行する
    '''
    print("Time interval: 1sec")
    file_list = os.listdir(INPUT_DIR_PATH)
    print("Target Person: {}\n".format(file_list))
    num = 1
    try:
        for name in file_list:
            print("<{0}/{1}>".format(num, len(file_list)))
            data_dir_path = INPUT_DIR_PATH + name
            result_file_path = EXPORT_DIR_PATH
            merge_datas(input_dir_path=data_dir_path, output_dir_path=result_file_path, name=name)
            num += 1
        print("\n[FINISH]: All files merged successfully!")
    except MybeatDateException as ex:
        print(ex)


def merge_datas(input_dir_path, output_dir_path, name):
    '''
    各種データを読み込み，1分間隔のデータで出力する
    ※FitBitデータは必ず存在することを前提に実装

    Args:
        input_dir_path:     読み込み対象ディレクトリパス
        output_dir_path:    出力ディレクトリへのパス
        name:           被験者名

    Returns:
        なし
    '''
    datas = {}
    # 読み込むファイル名を指定可能
    all_files = glob.glob(input_dir_path + "/*.csv")

    # csv読み込み
    for filename in all_files:
        df = pd.read_csv(filename, header=0, na_values='-')
        if list(df.columns) == MYBEAT_RRI_COLUMNS:
            datas['mybeat'] = MybeatData(df)
            print("[LOAD]: MyBeat(RRI) data.")
        elif list(df.columns) == MYBEAT_HR_COLUMNS:
            datas['mybeat'] = MybeatData(df)
            print("[LOAD]: MyBeat(HR) data.")
        elif list(df.columns) == FITBIT_COLUMNS:
            datas['fitbit'] = FitbitData(df)
            print("[LOAD]: FitBit data.")
        elif list(df.columns) == URINATION_COLUMNS:
            datas['urination'] = UrinationData(df)
            print("[LOAD]: Urination data.")
        elif list(df.columns) == ECHO_COLUMNS:
            datas['echo'] = EchoData(df)
            print("[LOAD]: Echo data.")
        elif list(df.columns) == TEMP_COLUMNS:
            datas['temp'] = TempData(df)
            print("[LOAD]: Temp data.")
        elif list(df.columns) == MYBEAT_3_ANALYSIS_DATA:
            datas['mybeat_v3_analysis'] = MybeatV3AnalysisData(df)
            print("[LOAD]: Mybeat v3 Analysis data.")
        elif list(df.columns) == MYBEAT_3_FREQUENCY_ANALYSIS_DATA:
            datas['mybeat_v3_frequency_analysis'] = MybeatV3FrequencyAnalysisData(df)
            print("[LOAD]: Mybeat v3 Frequency Analysis data.")
        elif list(df.columns) == MYBEAT_3_TIME_ANALYSIS_DATA:
            datas['mybeat_v3_time_analysis'] = MybeatV3TimeAnalysisData(df)
            print("[LOAD]: Mybeat v3 Time Analysis data.")
        else:
            print("[ERROR]: Invalid file.")
            print(df)
            return

    if len(datas) == 0:
        print("[WARN]: no file loaded.")
        return

    # データの整形
    print("Shaping... ", end='')
    for i in datas.values():
        i.data_shaping()
    print("OK!")

    print("Create Base Column... ", end='')
    # resultの作成
    result = pd.DataFrame(columns=['time'])
    progress_time = pd.to_datetime(START_TIME, format='%H:%M:%S')
    next_time = progress_time + datetime.timedelta(seconds=1)
    end_time = pd.to_datetime(FIN_TIME, format='%H:%M:%S')
    i = 0
    while True:
        # print(progress_time)
        result.loc[i, 'time'] = progress_time
        progress_time = next_time
        next_time = next_time + datetime.timedelta(seconds=1)
        i += 1
        if progress_time > end_time:
            break
    result.time = result.time.astype(str)
    print("OK!")

    if len(datas) == 1:
        # データが一つならそのまま出す
        result = list(datas.values())[0].df
    else:
        # mybeat_v3_analysisを軸にマージ
        print("Merging... ", end='')
        if 'mybeat_v3_analysis' in datas:
            result = pd.merge(result, datas['mybeat_v3_analysis'].df, on='time', how='left')
        # result = datas['mybeat_v3_analysis'].df
        # マージ順番を固定するため，データ名をハードコードしている．
        if 'fitbit' in datas:
            result = pd.merge(result, datas['fitbit'].df, on='time', how='left')
        if 'mybeat' in datas:
            result = pd.merge(result, datas['mybeat'].df, on='time', how='left')
        if 'urination' in datas:
            result = pd.merge(result, datas['urination'].df, on='time', how='left')
        elif 'echo' in datas:
            result = pd.merge(result, datas['echo'].df, on='time', how='left')
        if 'temp' in datas:
            result = pd.merge(result, datas['temp'].df, on='time', how='left')
        if 'mybeat_v3_frequency_analysis' in datas:
            result = pd.merge(result, datas['mybeat_v3_frequency_analysis'].df, on='time', how='left')
        if 'mybeat_v3_time_analysis' in datas:
            result = pd.merge(result, datas['mybeat_v3_time_analysis'].df, on='time', how='left')

    print("OK!")


    # マージ後，増えた残りのNaN(先頭以前と終点以降)を，すべて-1で埋める
    print("Fill in the NaN value")
    print("->urination... ", end='')
    result[['urination',]] = result[['urination',]].fillna(method = 'ffill')
    result['urination'] = result['urination'].fillna(-1)
    print("OK!")

    print("->others... ", end='')
    result = result.interpolate()
    print("OK!")

    output_file = output_dir_path + name + '_1sec'
    os.makedirs(output_dir_path, exist_ok=True)


    # 異常検知
    if ANOMALY_DETECT_SIZE > 1:
        # 出力ディレクトリ作成
        ad_output_dir_path = output_dir_path + name + '_1sec_Anomaly_detection_' + str(ANOMALY_DETECT_SIZE) + '/'
        os.makedirs(ad_output_dir_path, exist_ok=True)
        # 尿意計算
        ad_urination = pd.DataFrame()
        progress_column = 0
        while progress_column < ANOMALY_DETECT_SIZE:
            ad_urination['urination_' + str(progress_column)] = result['urination'].shift(progress_column)
            progress_column += 1
        # 尿意は最大値を取るよう上書き
        ad_urination['urination'] = ad_urination.loc[:, 'urination_0':'urination_' + str(ANOMALY_DETECT_SIZE - 1)].max(axis=1)
        # コピー済み不要な尿意データの削除
        for index in range(ANOMALY_DETECT_SIZE):
            ad_urination = ad_urination.drop(['urination_' + str(index)], axis=1)

        for item in result.iteritems():
            result_ad = pd.DataFrame()
            if item[0] in ['time', 'urination']:
                continue
            result_ad['time'] = result['time']
            progress_column = 0
            while progress_column < ANOMALY_DETECT_SIZE:
                result_ad[item[0] + '_' + str(progress_column)] = item[1].shift(progress_column)
                progress_column += 1
            result_ad['urination'] = ad_urination['urination']

            # ファイル出力
            ad_output_file = ad_output_dir_path + item[0]
            result_ad.to_csv(ad_output_file + '.csv', index=False)
        print("=>Output to '{0}'".format(ad_output_dir_path))


    # 移動平均
    if MOVE_AVE_WINDOW_SIZE > 1:
        result_ma = result.rolling(MOVE_AVE_WINDOW_SIZE, center=True).mean()
        result_ma['urination'] = result['urination'].rolling(MOVE_AVE_WINDOW_SIZE, center=True).max()     # 尿意は最大値
        result_ma.to_csv(output_file + '_Moving_Average.csv', index=False)
        print("=>Output to '{}_Moving_Average.csv'".format(output_file))

        # RRIのみ，求めたウィンドウサイズに対しての偏差を出し，トレンドグラフを作成する
        if 'RRI' in result:
            result = result.assign(RRI_trend_deviation=np.NaN)
            print(result)
            for i in range(len(result['RRI'])):
                result.at[i, 'RRI_trend_deviation'] = result.at[i, 'RRI'] - result_ma.at[i, 'RRI']

    # ファイル出力
    result.to_csv(output_file + '.csv', index=False)
    print("=>Output to '{}.csv'".format(output_file))

if __name__ == '__main__':
    main()

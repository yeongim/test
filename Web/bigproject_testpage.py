# 라이브러리 불러오기
import streamlit as st
import pandas as pd
from keras.models import load_model
import os, glob
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
import zipfile
from matplotlib import font_manager,rc
from PIL import Image
import time

# os.chdir('/Users/lgu01/Python_Personal/big_project')
Emotion_Stat_Dataset = pd.read_excel('./Data/Emotion_Data/Emotion_Stat_Dataset.xlsx')
Emotion_Stat_Dataset['Datetime'] = pd.to_datetime(Emotion_Stat_Dataset['Datetime'])
IoT_Stat_Dataset = pd.read_excel('./Data/IoT_Data/IoT_Stat_Dataset.xlsx')
IoT_Stat_Dataset['등록일시'] = IoT_Stat_Dataset['등록일시'].dt.strftime('%Y-%m-%d %H:%M:%S')
Person_Dataset = pd.read_excel('./Data/Person_Data/Person_Dataset.xlsx')
Info_Change_Reason_Dataset = pd.read_excel('./Data/Person_Data/Info_Change_Reason_Dataset.xlsx')
IoT_Sensor_Info_Dataset = pd.read_excel('./Data/Person_Data/IoT_Sensor_Info.xlsx')
Lone_Person_Dataset = pd.read_csv('./Data/Lone_Person_Data/group_n.zip', compression='zip', header=0, sep=',', quotechar='"')
Lone_Person_Dataset.loc['연령대'] = Lone_Person_Dataset['연령대'].astype('float')
conditions = [
        (Lone_Person_Dataset['연령대'] >= 20) & (Lone_Person_Dataset['연령대'] < 30),
        (Lone_Person_Dataset['연령대'] >= 30) & (Lone_Person_Dataset['연령대'] < 40),
        (Lone_Person_Dataset['연령대'] >= 40) & (Lone_Person_Dataset['연령대'] < 50),
        (Lone_Person_Dataset['연령대'] >= 50) & (Lone_Person_Dataset['연령대'] < 60),
        (Lone_Person_Dataset['연령대'] >= 60) & (Lone_Person_Dataset['연령대'] < 70),
        (Lone_Person_Dataset['연령대'] >= 70) & (Lone_Person_Dataset['연령대'] < 80)
  ]
values = ['20대', '30대', '40대', '50대', '60대', '70대']
# 성별, 날짜 전처리
Lone_Person_Dataset['연령대'] = np.select(conditions, values, default='80대')

font_name = font_manager.FontProperties(fname='./Data/Font/NanumGothic.otf').get_name()
columns_gap = 'large'

########################### ARIMA 모델 함수 ########################################### 박소은 작성 => 이강욱 수정 및 통합
def Lone_Person_Dataset_Loader(group_name, region_name, gender_name, age_name):
    
    df = Lone_Person_Dataset

    # 성별, 날짜 전처리
    df['성별'] = df['성별'].replace({1: '남성', 2: '여성'})
    df['month'] = df['month'].apply(lambda x: str('-'.join(str(x).split('.'))))
    df['month'] = pd.to_datetime(df['month'])
    df['month'] = df['month'].dt.to_period('M')
    x_train = df.loc[(df['month'] != '2023-01') & (df['month'] != '2023-02') & (df['month'] != '2023-03')]
    x_test = df.loc[(df['month'] == '2023-01') | (df['month'] == '2023-02') | (df['month'] == '2023-03')]
    
    df_ts = x_train.groupby(['month', '자치구', '성별', '연령대']).sum().reset_index()

    # 시계열 분석을 위한 데이터 프레임 재구성(피봇테이블)
    df_pivot = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values=group_name)
    frame = pd.DataFrame(df_pivot[region_name, gender_name, age_name])
    df_col = df_pivot[region_name, gender_name, age_name].fillna(0)
    column_name = frame.columns.values[0]
    
    # PeriodIndex를 DatetimeIndex으로 바꾸기 ('M'한달 간격)
    new_index = df_col.index.to_timestamp(freq='M')

    # Datetimeindex 바꾼 데이터프레임
    df_col = pd.DataFrame(df_pivot[region_name, gender_name, age_name].values, index=new_index, columns=[column_name])

    # ARIMA 모델 학습
    model = ARIMA(df_col, order=(2, 1, 1))  # p, d, q
    fitted_model = model.fit()

    # 2023 예측
    start_idx = len(df_col) - 1  # 지난 관측값에서 시작
    end_idx = start_idx + 12  # 12개월치 예측
    forecast = fitted_model.predict(start=start_idx, end=end_idx, typ='levels')
    forecast_index = pd.date_range(start=df_col.index[-1], periods=13, freq='M')[1:]
    forecast = pd.Series(forecast, index=forecast_index)
   
    df_ts2 = x_test.groupby(['month', '자치구', '성별', '연령대']).sum().reset_index()

    # 시계열 분석을 위한 데이터 프레임 재구성(피봇테이블) 
    df_pivot2 = df_ts2.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values=group_name)
    
    # 한글 설정
    import matplotlib.font_manager as fm
    # 폰트 경로 설정
    plt.rc('font', family='sans-serif')
    df_col2 = df_pivot2[region_name, gender_name, age_name].fillna(0)
    from sklearn.metrics import mean_squared_error

    # 예측값과 실제값 사이의 MSE 계산
    mse = mean_squared_error(df_pivot2[region_name, gender_name, age_name], forecast[0:3], squared=False)
    
    # 예측 결과 시계열차트
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df_col.index.strftime('%Y-%m-%d'), df_pivot[region_name, gender_name, age_name], marker='o', color='pink', label='actual')
    for i in range(len(df_col.index)):
        height = df_pivot[region_name, gender_name, age_name][i]
        ax.text(df_col.index[i].strftime('%Y-%m-%d'), height + 0.25, '%.1f' % height, ha='center', va='bottom', size=10)

    ax.plot(forecast.index.strftime('%Y-%m-%d'), forecast, label='predict', marker='o', color='gray')
    for i in range(len(forecast.index)):
        height = forecast[i]
        ax.text(forecast.index[i].strftime('%Y-%m-%d'), height + 1, '%.1f' % height, ha='center', va='bottom', size=10, color='gray')

    # ax.plot(df_col2.index.strftime('%Y-%m-%d'), df_pivot2[region_name, gender_name, age_name], label='미래 실제값', marker='o', color='pink')
    # for i in range(len(df_col2.index)):
    #     height = df_pivot2[region_name, gender_name, age_name][i]
    #     ax.text(df_col2.index[i].strftime('%Y-%m-%d'), height + 0.25, '%.1f' % height, ha='center', va='bottom', size=10)

    plt.xticks(rotation=45)
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Count')
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

    
########################### 예측값 출력 함수 ########################################### 박소은 작성 => 이강욱 수정 및 통합
def pred(group_name, region_name, gender_name, age_name):
#     df = pd.read_csv('./Data/Lone_Person_Data/group_n.csv')
    df = Lone_Person_Dataset

    df['성별'] = df['성별'].replace({1: '남성', 2: '여성'})
    df['month'] = df['month'].apply(lambda x: str('-'.join(str(x).split('.'))))
    df['month'] = pd.to_datetime(df['month'])
    df['month'] = df['month'].dt.to_period('M')
    x_train = df.loc[(df['month'] != '2023-01') & (df['month'] != '2023-02') & (df['month'] != '2023-03')]
    x_test = df.loc[(df['month'] == '2023-01') | (df['month'] == '2023-02') | (df['month'] == '2023-03')]
    
    df_ts = x_train.groupby(['month', '자치구', '성별', '연령대']).sum().reset_index()

    # 시계열 분석을 위한 데이터 프레임 재구성(피봇테이블)
    df_pivot = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values=group_name)
    frame = pd.DataFrame(df_pivot[region_name, gender_name, age_name])
    df_col = df_pivot[region_name, gender_name, age_name].fillna(0)
    column_name = frame.columns.values[0]
    
    # PeriodIndex를 DatetimeIndex으로 바꾸기 ('M'한달 간격)
    new_index = df_col.index.to_timestamp(freq='M')

    # Datetimeindex 바꾼 데이터프레임
    df_col = pd.DataFrame(df_pivot[region_name, gender_name, age_name].values, index=new_index, columns=[column_name])

    # ARIMA 모델 학습
    model = ARIMA(df_col, order=(2, 1, 1))  # p, d, q
    fitted_model = model.fit()

    # 2023 예측
    start_idx = len(df_col) - 1  # 지난 관측값에서 시작
    end_idx = start_idx + 12  # 12개월치 예측
    forecast = fitted_model.predict(start=start_idx, end=end_idx, typ='levels')
    forecast_index = pd.date_range(start=df_col.index[-1], periods=13, freq='M')[1:]
    forecast = pd.Series(forecast, index=forecast_index, name='predict')
   
    df_ts2 = x_test.groupby(['month', '자치구', '성별', '연령대']).sum().reset_index()

    # 시계열 분석을 위한 데이터 프레임 재구성(피봇테이블)
    df_pivot2 = df_ts2.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values=group_name)
    
    # 한글 설정
    import matplotlib.font_manager as fm
    # 폰트 경로 설정
    plt.rc('font', family='sans-serif')
    df_col2 = df_pivot2[region_name, gender_name, age_name].fillna(0)
    
    st.write(forecast)



    
########################### 파이차트(현재) 함수 ########################################### 박소은 작성 => 이강욱 수정 및 통합
def piechart(region, gender, age):
#     df = pd.read_csv('./Data/Lone_Person_Data/group_n.csv')
    df = Lone_Person_Dataset

    # 성별, 날짜 전처리
    df['연령대'] = np.select(conditions, values, default='80대')
    df['성별'] = df['성별'].replace({1: '남성', 2: '여성'})
    df['month'] = df['month'].apply(lambda x: str('-'.join(str(x).split('.'))))
    df['month'] = pd.to_datetime(df['month'])
    df['month'] = df['month'].dt.to_period('M')
    x_train = df.loc[(df['month'] != '2023-01') & (df['month'] != '2023-02') & (df['month'] != '2023-03')]
    x_test = df.loc[(df['month'] == '2023-01') | (df['month'] == '2023-02') | (df['month'] == '2023-03')]

    df_ts = x_train.groupby(['month', '자치구', '성별', '연령대']).sum().reset_index()
    
    # 시계열 분석을 위한 데이터 프레임 재구성(피봇테이블) 자치구, 성별, 연령대별 각 집단 값의 피봇테이블
    df_pivot3_1 = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values='커뮤니케이션이 적은 집단')
    df_pivot3_2 = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values='평일 외출이 적은 집단')
    df_pivot3_3 = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values='휴일 외출이 적은 집단')
    df_pivot3_4 = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values='출근소요시간 및 근무시간이 많은 집단')
    df_pivot3_5 = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values='외출이 매우 적은 집단(전체)')
    df_pivot3_6 = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values='외출이 매우 많은 집단')
    df_pivot3_7 = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values='동영상서비스 이용이 많은 집단')
    df_pivot3_8 = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values='생활서비스 이용이 많은 집단')
    df_pivot3_9 = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values='재정상태에 대한 관심집단')
    df_pivot3_10 = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values='외출-커뮤니케이션이 모두 적은 집단(전체)')
    
    value1 = df_pivot3_1[region, gender, age]['2022-12']
    value2 = df_pivot3_2[region, gender, age]['2022-12']
    value3 = df_pivot3_3[region, gender, age]['2022-12']
    value4 = df_pivot3_4[region, gender, age]['2022-12']
    value5 = df_pivot3_5[region, gender, age]['2022-12']
    value6 = df_pivot3_6[region, gender, age]['2022-12']
    value7 = df_pivot3_7[region, gender, age]['2022-12']
    value8 = df_pivot3_8[region, gender, age]['2022-12']
    value9 = df_pivot3_9[region, gender, age]['2022-12']
    value10 = df_pivot3_10[region, gender, age]['2022-12']
    total = value1+value2+value3+value4+value5+value6+value7+value8+value9+value10

    # 파이차트: 자치구, 성별, 연령을 지정하면 현재(2022-12), 미래(2023-12) 1인 가구 집단의 비중
    ratio = [value1/total, value2/total, value3/total, value4/total, value5/total, value6/total, value7/total, value8/total, value9/total, value10/total]
    labels = ['커뮤니케이션이 적은 집단',
           '평일 외출이 적은 집단', '휴일 외출이 적은 집단', '출근소요시간 및 근무시간이 많은 집단',
           '외출이 매우 적은 집단(전체)', '외출이 매우 많은 집단', '동영상서비스 이용이 많은 집단',
           '생활서비스 이용이 많은 집단', '재정상태에 대한 관심집단', '외출-커뮤니케이션이 모두 적은 집단(전체)']
    
    fig = px.pie(values=ratio, names=labels, hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(font=dict(size=16))
    return fig



########################### 파이차트(미래) 함수 ########################################### 박소은 작성 => 이강욱 수정 및 통합
def piechart_pred(region, gender, age):
#     df = pd.read_csv('./Data/Lone_Person_Data/group_n.csv')
    df = Lone_Person_Dataset

    # 성별, 날짜 전처리
    df['연령대'] = np.select(conditions, values, default='80대')
    df['성별'] = df['성별'].replace({1: '남성', 2: '여성'})
    df['month'] = df['month'].apply(lambda x: str('-'.join(str(x).split('.'))))
    df['month'] = pd.to_datetime(df['month'])
    df['month'] = df['month'].dt.to_period('M')
    x_train = df.loc[(df['month'] != '2023-01') & (df['month'] != '2023-02') & (df['month'] != '2023-03')]
    x_test = df.loc[(df['month'] == '2023-01') | (df['month'] == '2023-02') | (df['month'] == '2023-03')]

    df_ts = x_train.groupby(['month', '자치구', '성별', '연령대']).sum().reset_index()
    
    # 시계열 분석을 위한 데이터 프레임 재구성(피봇테이블) 자치구, 성별, 연령대별 각 집단 값의 피봇테이블
    df_pivot3_1 = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values='커뮤니케이션이 적은 집단')
    df_pivot3_2 = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values='평일 외출이 적은 집단')
    df_pivot3_3 = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values='휴일 외출이 적은 집단')
    df_pivot3_4 = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values='출근소요시간 및 근무시간이 많은 집단')
    df_pivot3_5 = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values='외출이 매우 적은 집단(전체)')
    df_pivot3_6 = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values='외출이 매우 많은 집단')
    df_pivot3_7 = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values='동영상서비스 이용이 많은 집단')
    df_pivot3_8 = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values='생활서비스 이용이 많은 집단')
    df_pivot3_9 = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values='재정상태에 대한 관심집단')
    df_pivot3_10 = df_ts.pivot_table(index='month', columns=['자치구', '성별', '연령대'], values='외출-커뮤니케이션이 모두 적은 집단(전체)')
    
    # 10개 집단에 대한 예측치 forecast1~10개 추출
    forecast_list = []
    for i in range(1, 11):
        df_col = locals()[f'df_pivot3_{i}'][region, gender, age].fillna(0)

        # PeriodIndex를 DatetimeIndex으로 바꾸기 ('M'한달 간격)
        new_index = df_col.index.to_timestamp(freq='M')
        frame = pd.DataFrame(locals()[f'df_pivot3_{i}'][region, gender, age])
        column_name = frame.columns.values[0]

        # Datetimeindex 바꾼 데이터프레임
        df_col = pd.DataFrame(locals()[f'df_pivot3_{i}'][region, gender, age].values, index=new_index, columns=[column_name])

        # 모델 학습
        model = ARIMA(df_col, order=(2, 1, 1))  # p, d, q
        fitted_model = model.fit()

        # 2023 예측
        start_idx = len(df_col) - 1  # 지난 관측값에서 시작
        end_idx = start_idx + 12  # 12개월치 예측
        forecast = fitted_model.predict(start=start_idx, end=end_idx, typ='levels')
        forecast_index = pd.date_range(start=df_col.index[-1], periods=13, freq='M')[1:]
        forecast = pd.Series(forecast, index=forecast_index)
        forecast_list.append(forecast)

    forecast1, forecast2, forecast3, forecast4, forecast5, forecast6, forecast7, forecast8, forecast9, forecast10 = forecast_list
    pred1 = forecast1['2023-12-31']
    pred2 = forecast2['2023-12-31']
    pred3 = forecast3['2023-12-31']
    pred4 = forecast4['2023-12-31']
    pred5 = forecast5['2023-12-31']
    pred6 = forecast6['2023-12-31']
    pred7 = forecast7['2023-12-31']
    pred8 = forecast8['2023-12-31']
    pred9 = forecast9['2023-12-31']
    pred10 = forecast10['2023-12-31']
    pred_total = pred1 + pred2 + pred3 + pred4 + pred5 + pred6 + pred7 + pred8 + pred9 + pred10


    # 파이플롯
    ratio = [pred1/pred_total, pred2/pred_total, pred3/pred_total, pred4/pred_total, pred5/pred_total, pred6/pred_total, pred7/pred_total, pred8/pred_total, pred9/pred_total, pred10/pred_total]
    labels = ['커뮤니케이션이 적은 집단',
           '평일 외출이 적은 집단', '휴일 외출이 적은 집단', '출근소요시간 및 근무시간이 많은 집단',
           '외출이 매우 적은 집단(전체)', '외출이 매우 많은 집단', '동영상서비스 이용이 많은 집단',
           '생활서비스 이용이 많은 집단', '재정상태에 대한 관심집단', '외출-커뮤니케이션이 모두 적은 집단(전체)']

    fig = px.pie(values=ratio, names=labels, hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(font=dict(size=16))
    
    return fig


########################### IoT 추가 함수 ########################################### 이강욱 작성 및 수정
def IoT_add_set_Dataset(Temp_Dataset):
    Original_Temp_Dataset = Temp_Dataset
    Temp_Dataset['등록일시'] = pd.to_datetime(Temp_Dataset['등록일시']).dt.date 
    Temp_Dataset = Temp_Dataset.groupby(['등록일시'], as_index=False)[['전력량1 (Wh)', '조도1 (%)']].sum()
    Temp_Dataset['전력량1 (Wh) 일 평균'] = Temp_Dataset['전력량1 (Wh)'] / 24
    Temp_Dataset['조도1 (%) 일 평균'] = Temp_Dataset['조도1 (%)'] / 24
    
    Target_Type = '정상'
    Current_Status = '정상'
    
    return Original_Temp_Dataset, Temp_Dataset, Target_Type, Current_Status


########################### IoT 위험 감지 함수 ########################################### 이강욱 작성 및 수정
def IoT_Emergency_Detect_Function(t1_Target_Alert_Time, Temp_Dataset):
    
    # st.table(Temp_Dataset.head(10))
    
    Electro_Status, Light_Status = False, False
    Electro_Time, Light_Time = 1, 1
    
    for x in range(len(Temp_Dataset['전력량1 (Wh)'])-1):
        if Temp_Dataset.loc[x, '전력량1 (Wh)'] == Temp_Dataset.loc[x+1, '전력량1 (Wh)']:
            Electro_Time += 1
        else:
            break

    for y in range(len(Temp_Dataset['조도1 (%)'])-1):
        if Temp_Dataset.loc[y, '조도1 (%)'] == Temp_Dataset.loc[y+1, '조도1 (%)']:
            Light_Time += 1
        else:
            break
            
    if t1_Target_Alert_Time <= Electro_Time:
            Electro_Status = True
    if t1_Target_Alert_Time <= Light_Time:
            Light_Status = True
    
    return Electro_Status, Electro_Time, Light_Status, Light_Time
    
    

########################### 디자인 등 사전 작성 ######################################### 박소은, 이강욱 작성 및 수정

    # 버튼 스타일 변경 => 버튼 불필요로 적용된 코드 주석처리됨
    
button_style = """
        <style>
        .stButton button {
            background-color: #5EBAB8;
            color: white;
            padding: 0.5em 1em;
            border-radius: 0.3em;
            border: none;
        }
        </style>
    """


########################### 페이지 코드 함수 ########################################### 이강욱 작성 및 수정

st.set_page_config(layout="wide")

st.markdown(" ## 사회복지사 통계 포털")
tab1, tab2, tab3, tab4 = st.tabs(["IoT 통계", "감정분석 통계", "1인가구 집단 시계열 통계", "대상자 설정 대시보드"])


###########################
with tab1: # IoT 통계
    t1_col1_1, _ , t1_col1_2, _ , t1_col1_3 = st.columns([0.3, 0.05, 0.3, 0.05, 0.3], gap = columns_gap)
    with t1_col1_1:
        st.subheader('대상자 선택')
        tab1_selectbox = st.selectbox('대상자 선택', Person_Dataset['Name'].unique(), key = 'tab1_대상자선택', label_visibility="collapsed")
    
    t1_Serial_Num = Person_Dataset.loc[Person_Dataset.loc[Person_Dataset['Name'] == tab1_selectbox].index, 'IoT_Serial_Num'].reset_index(drop=True)[0]
    IoT_Stat_Dataset_Search_Result_1, IoT_Stat_Dataset_Search_Result_2, t1_Target_Type, IoT_Target_Status = IoT_add_set_Dataset(IoT_Stat_Dataset.loc[IoT_Stat_Dataset['시리얼'] == t1_Serial_Num]) # 함수 별도 추가
    t1_Target_Type = Person_Dataset.loc[Person_Dataset.loc[Person_Dataset['Name'] == tab1_selectbox].index, 'IoT_Type'].reset_index(drop=True)[0]
    t1_Temp_IoT_Sensor_Chosen_Dataset = IoT_Sensor_Info_Dataset.loc[IoT_Sensor_Info_Dataset['IoT_Serial_Num'] == t1_Serial_Num].sort_values('Sensor_Type', ascending=True)
        
    t1_Target_Alert_Time = 0
    if t1_Target_Type == '일반군':
        t1_Target_Alert_Time = 50
    elif t1_Target_Type == '위험군':
        t1_Target_Alert_Time = 36
    elif t1_Target_Type == '고위험군':
        t1_Target_Alert_Time = 24
        
    Electro_Status, Electro_Time, Light_Status, Light_Time = IoT_Emergency_Detect_Function(t1_Target_Alert_Time, 
                                                                IoT_Stat_Dataset.loc[IoT_Stat_Dataset['시리얼'] == t1_Serial_Num]
                                                                .sort_values('등록일시', ascending = False).reset_index(drop=True))
    
    with t1_col1_2:
        st.subheader('현재 상태')
        st.text('- 대상자 시리얼 번호 : ' +  t1_Serial_Num)
        st.text('- 대상자 분류 : ' + t1_Target_Type + ' => ' + str(t1_Target_Alert_Time) + '시간 적용 대상')
    # st.table(t1_Temp_IoT_Sensor_Chosen_Dataset.loc[t1_Temp_IoT_Sensor_Chosen_Dataset['Sensor_Type'] == 'Electro'])
    # st.text(t1_Temp_IoT_Sensor_Chosen_Dataset.loc[t1_Temp_IoT_Sensor_Chosen_Dataset['Sensor_Type'] == 'Electro'].value_counts())
    with t1_col1_3:
        t1_col1A_1, _, t1_col1A_2 = st.columns([0.45, 0.1, 0.45], gap = columns_gap)
        with t1_col1A_1:
            if len(t1_Temp_IoT_Sensor_Chosen_Dataset.loc[t1_Temp_IoT_Sensor_Chosen_Dataset['Sensor_Type'] == 'Electro']) == 0:
                st.image(Image.open('./Data/Asset/Electro_Gray.png'), caption='찾을 수 없음')
            elif Electro_Status == True:
                st.image(Image.open('./Data/Asset/Electro_Red.png'), caption= str(Electro_Time) + ' 시간 이상 변화없음')
            elif Electro_Time >= 15:
                st.image(Image.open('./Data/Asset/Electro_Orange.png'), caption= '15시간 이상 변화없음')
            else:
                st.image(Image.open('./Data/Asset/Electro_Green.png'), caption= '정상 작동 중')
        with t1_col1A_2:
            st.text('')
            if len(t1_Temp_IoT_Sensor_Chosen_Dataset.loc[t1_Temp_IoT_Sensor_Chosen_Dataset['Sensor_Type'] == 'Light']) == 0:
                st.image(Image.open('./Data/Asset/Light_Gray.png'), caption='찾을 수 없음')
            elif Light_Status == True:
                st.image(Image.open('./Data/Asset/Light_Red.png'), caption= str(Light_Time) + ' 시간 이상 변화없음')
            elif Light_Time >= 15:
                st.image(Image.open('./Data/Asset/Light_Orange.png'), caption= '15시간 이상 변화없음')
            else:
                st.image(Image.open('./Data/Asset/Light_Green.png'), caption= '정상 작동 중')
        
    st.subheader('')
    if Electro_Status == False and Light_Status == False:
        st.success('현 대상자의 모든 센서가 정상적으로 작동 중입니다! (전력: '+ str(Electro_Time) +'h / 조도 : ' + str(Light_Time) + 'h)', icon="✅")
    elif Electro_Status == True and Light_Status == False:
        st.error('현 대상자의 전력 센서 측정값이 기준을 초과하였습니다! (전력: '+ str(Electro_Time) +'h / 조도 : ' + str(Light_Time) + 'h)', icon="🚨")
    elif Electro_Status == False and Light_Status == True:
        st.error('현 대상자의 조도 센서 측정값이 기준을 초과하였습니다! (전력: '+ str(Electro_Time) +'h / 조도 : ' + str(Light_Time) + 'h)', icon="🚨")
    elif Electro_Status == True and Light_Status == True:
        st.error('현 대상자의 센서 측정값 모두가 기준을 초과하였습니다! (전력: '+ str(Electro_Time) +'h / 조도 : ' + str(Light_Time) + 'h)', icon="🚨")

    st.subheader('')
    t1_col2_1, _, t1_col2_2= st.columns([0.45, 0.1, 0.45], gap = columns_gap)
    with t1_col2_1:
        st.header('전력센서 통계')
        st.text('최근 50시간 기준 전력량(Wh) 그래프')
        st.line_chart(data = IoT_Stat_Dataset.loc[IoT_Stat_Dataset['시리얼'] == t1_Serial_Num], x = '등록일시', y = '전력량1 (Wh)')
        st.text('최근 7일 전력소비량(Wh) 총량의 일평균 값 그래프')
        st.line_chart(data = IoT_Stat_Dataset_Search_Result_2, x = '등록일시', y = '전력량1 (Wh) 일 평균')
            
    with t1_col2_2:
        st.header('조도센서 통계')
        st.text('최근 50시간 기준 조도(%) 그래프')
        st.line_chart(data = IoT_Stat_Dataset.loc[IoT_Stat_Dataset['시리얼'] == t1_Serial_Num], x = '등록일시', y = '조도1 (%)')
        st.text('최근 7일 조도(%)의 일평균 값 그래프')
        st.line_chart(data = IoT_Stat_Dataset_Search_Result_2, x = '등록일시', y = '조도1 (%) 일 평균')

    with st.expander("센서 통계 데이터 차트 상세보기"):
        st.subheader('센서 통계 상세 : ' + t1_Serial_Num)
        t1_col4_1, t1_col4_2, t1_col4_3 = st.columns([0.33, 0.34, 0.33], gap = columns_gap)
        with t1_col4_1:
            tab1_selectbox_2 = st.selectbox('상세보기 대상자 선택', Person_Dataset['Name'].unique(), key = 'tab1_대상자선택_2')
            t1_Serial_Num_2 = Person_Dataset.loc[Person_Dataset.loc[Person_Dataset['Name'] == tab1_selectbox_2].index, 'IoT_Serial_Num'].reset_index(drop=True)[0]
            st.text(t1_Serial_Num_2)
            IoT_Stat_Dataset_Search_Result_3 = IoT_Stat_Dataset.loc[IoT_Stat_Dataset['시리얼'] == t1_Serial_Num_2]
#             IoT_Stat_Dataset_Search_Result_3['등록일시'] = pd.to_datetime(IoT_Stat_Dataset_Search_Result_3['등록일시'], format='%Y-%m-%d').dt.date 
            IoT_Stat_Dataset_Search_Result_3['등록일시'] = pd.to_datetime(IoT_Stat_Dataset_Search_Result_3['등록일시']).dt.date 
        with t1_col4_2:
            Min_Datetime, Max_Datetime = min(IoT_Stat_Dataset_Search_Result_3['등록일시']), max(IoT_Stat_Dataset_Search_Result_3['등록일시'])
            Datetime_array = st.slider('날짜 범위', value = (Min_Datetime, Max_Datetime))
        with t1_col4_3:
            IoT_Sort_Value = st.selectbox('정렬기준', ['내림차순', '오름차순'])
            if IoT_Sort_Value == '오름차순': IoT_Sort_Value = True
            elif IoT_Sort_Value == '내림차순': IoT_Sort_Value = False
        
        st.table(IoT_Stat_Dataset.loc[(IoT_Stat_Dataset['시리얼'] == t1_Serial_Num_2) & (pd.to_datetime(IoT_Stat_Dataset['등록일시']) >= pd.to_datetime(Datetime_array[0])) 
                                  & (pd.to_datetime(IoT_Stat_Dataset['등록일시']) <= pd.to_datetime(Datetime_array[1]))].set_index('등록일시')[['조도1 (%)', '전력량1 (Wh)']]
                                 .sort_values('등록일시', ascending = IoT_Sort_Value))

###########################
with tab2: # 감정분석 통계
    st.cache_data.clear()
    st.cache_resource.clear()
    t2_col1_1, t2_col1_2, t2_col1_3 = st.columns([0.33, 0.34, 0.33], gap = columns_gap)
    t1_min_datetime, t1_max_datetime = pd.to_datetime(min(Emotion_Stat_Dataset['Datetime'])), pd.to_datetime(max(Emotion_Stat_Dataset['Datetime']))
    with t2_col1_1:
        st.subheader('대상자 선택')
        tab2_selectbox = st.selectbox('대상자 선택', Emotion_Stat_Dataset['User'].unique(), key = 'tab2_대상자선택', label_visibility="collapsed")

    with t2_col1_2:
        st.subheader('기준일 선택')
        tab2_targetdate = pd.to_datetime(st.date_input('대상 날짜 선택',value=datetime.date(2023, 6, 25), min_value = t1_min_datetime, max_value = t1_max_datetime, key = 'tab2_대상날짜선택', label_visibility="collapsed"))
        # tab2_targetdate = time.mktime(tab2_targetdate.timetuple())
        # st.text(type(tab2_targetdate))
        # st.text(type(t1_min_datetime))
        # st.text(type(Emotion_Stat_Dataset.loc[0, 'Datetime']))
        Emotion_Stat_Dataset_Search_Result_1 = Emotion_Stat_Dataset.loc[(Emotion_Stat_Dataset['User'] == tab2_selectbox) & (Emotion_Stat_Dataset['Datetime'] >= t1_min_datetime) & (Emotion_Stat_Dataset['Datetime'] <= tab2_targetdate)]
        Emotion_Stat_Dataset_Search_Result_2 = Emotion_Stat_Dataset_Search_Result_1.groupby(['Datetime'], as_index=False)[['Negative_Count']].sum()
        Emotion_Stat_Dataset_Search_Result_3 = Emotion_Stat_Dataset_Search_Result_2
        Emotion_Stat_Dataset_Search_Result_3['Week'] = pd.to_datetime(Emotion_Stat_Dataset_Search_Result_3['Datetime']).map(lambda x: x.isocalendar()[1])
        Emotion_Stat_Dataset_Search_Result_3 = Emotion_Stat_Dataset_Search_Result_3.groupby(['Week'], as_index=False)[['Negative_Count']].sum()
        
        Emotion_Stat_Dataset_Search_Result_4 = Emotion_Stat_Dataset.loc[Emotion_Stat_Dataset['User'] == tab2_selectbox]
        Emotion_Stat_Dataset_Search_Result_4 = Emotion_Stat_Dataset_Search_Result_4.groupby(['Datetime'], as_index=False)[['Negative_Count']].sum()
        Emotion_Stat_Dataset_Search_Result_4['Week'] = pd.to_datetime(Emotion_Stat_Dataset_Search_Result_4['Datetime']).map(lambda x: x.isocalendar()[1])
        Emotion_Stat_Dataset_Search_Result_4 = Emotion_Stat_Dataset_Search_Result_4.groupby(['Week'], as_index=False)[['Negative_Count']].sum()

        Emotion_Stat_Dataset_Search_Result_5 = Emotion_Stat_Dataset_Search_Result_2.sort_values('Datetime', ascending=False).head(7)
        Last_Week_Negative_Count = Emotion_Stat_Dataset_Search_Result_5['Negative_Count'].sum()
        # st.table(Emotion_Stat_Dataset_Search_Result_5)
        # st.text(Last_Week_Negative_Count)
        
    
    t2_Serial_Num = Person_Dataset.loc[Person_Dataset.loc[Person_Dataset['Name'] == tab2_selectbox].index, 'IoT_Serial_Num'].reset_index(drop=True)[0]
    t2_Target_Type = Person_Dataset.loc[Person_Dataset.loc[Person_Dataset['Name'] == tab2_selectbox].index, 'Emotion_Type'].reset_index(drop=True)[0]
    # Last_Week_Negative_Count = Emotion_Stat_Dataset_Search_Result_4.sort_values('Week', ascending = False).reset_index(drop=True).loc[0, 'Negative_Count']
    
    if Last_Week_Negative_Count < 100 and Last_Week_Negative_Count != '정상(안정)':
        Person_Dataset.loc[Person_Dataset.loc[Person_Dataset['Name'] == tab2_selectbox].index, 'Emotion_Type'] = '정상(안정)'
        Person_Dataset.to_excel('./Data/Person_Data/Person_Dataset.xlsx', index=False)
        Person_Dataset = pd.read_excel('./Data/Person_Data/Person_Dataset.xlsx')
    elif Last_Week_Negative_Count >= 100 and Last_Week_Negative_Count < 130 and Last_Week_Negative_Count != '주의 필요':
        Person_Dataset.loc[Person_Dataset.loc[Person_Dataset['Name'] == tab2_selectbox].index, 'Emotion_Type'] = '주의 필요'
        Person_Dataset.to_excel('./Data/Person_Data/Person_Dataset.xlsx', index=False)
        Person_Dataset = pd.read_excel('./Data/Person_Data/Person_Dataset.xlsx')
    elif Last_Week_Negative_Count >= 130 and Last_Week_Negative_Count < 150 and Last_Week_Negative_Count != '심리 상담 필요':
        Person_Dataset.loc[Person_Dataset.loc[Person_Dataset['Name'] == tab2_selectbox].index, 'Emotion_Type'] = '심리 상담 필요'
        Person_Dataset.to_excel('./Data/Person_Data/Person_Dataset.xlsx', index=False)
        Person_Dataset = pd.read_excel('./Data/Person_Data/Person_Dataset.xlsx')
    elif Last_Week_Negative_Count >= 150 and Last_Week_Negative_Count != '즉시 조치 필요':
        Person_Dataset.loc[Person_Dataset.loc[Person_Dataset['Name'] == tab2_selectbox].index, 'Emotion_Type'] = '즉시 조치 필요'
        Person_Dataset.to_excel('./Data/Person_Data/Person_Dataset.xlsx', index=False)
        Person_Dataset = pd.read_excel('./Data/Person_Data/Person_Dataset.xlsx')
    
    Target_Negative_Count, Target_Negative_Count_Text = 999, '본 메시지가 보일 시 오류 발생한 것입니다.'
    if t2_Target_Type == '정상(안정)':
        Target_Negative_Count, Target_Negative_Count_Text = 0, '100회 미만 적용'
    elif t2_Target_Type == '주의 필요':
        Target_Negative_Count, Target_Negative_Count_Text = 100, '130회 미만 적용'
    elif t2_Target_Type == '심리 상담 필요':
        Target_Negative_Count, Target_Negative_Count_Text = 130, '150회 미만 적용'
    elif t2_Target_Type == '즉시 조치 필요':
        Target_Negative_Count, Target_Negative_Count_Text = 150, '150회 이상 적용'

    with t2_col1_3:
        st.subheader('현재 상태')
        st.text('- 대상자 시리얼 번호 : ' +  t2_Serial_Num)
        st.text('- 대상자 분류 : ' + t2_Target_Type + ' => ' + Target_Negative_Count_Text)
        st.text('- 최근 마지막 주차 ' + str(Last_Week_Negative_Count) + '회 누적')
    
    st.subheader('')
    if t2_Target_Type == '정상(안정)':
        st.success('현 대상자는 정상(안정)입니다!', icon="✅")
    elif t2_Target_Type == '주의 필요':
        st.warning('현 대상자는 주의가 필요합니다!', icon="⚠️")
    elif t2_Target_Type == '심리 상담 필요':
        st.warning('현 대상자는 심리 상담이 필요합니다!', icon="⚠️")
    elif t2_Target_Type == '즉시 조치 필요':
        st.error('현 대상자는 즉시 조치가 필요합니다!', icon="🚨")

    Emotion_Stat_Dataset_Search_Result_4 = None
    
    
    
    st.subheader('')
    t2_col3_1, t2_col3_2, t2_col3_3 = st.columns([0.3, 0.3, 0.3], gap = columns_gap)
    with t2_col3_1:
        st.subheader('시간대별 부정 횟수 집계 그래프')
        st.text('최근 시간대 집계 횟수 : ' + str(Emotion_Stat_Dataset_Search_Result_1.sort_values('End_Time', ascending = False).reset_index(drop=True).loc[0, 'Negative_Count']) + '회')
        st.line_chart(data = Emotion_Stat_Dataset_Search_Result_1.tail(8), x = 'End_Time', y = 'Negative_Count')
        
    with t2_col3_2:
        st.subheader('일별 총 부정 횟수 집계 그래프')
        Emotion_Stat_Dataset_Search_Result_2['Datetime'] = Emotion_Stat_Dataset_Search_Result_2['Datetime'].dt.date
        st.text('최근 마지막 일 집계 횟수 : ' + str(Emotion_Stat_Dataset_Search_Result_2.sort_values('Datetime', ascending = False).reset_index(drop=True).loc[0, 'Negative_Count']) + '회')
        st.line_chart(data = Emotion_Stat_Dataset_Search_Result_2.tail(7), x = 'Datetime', y = 'Negative_Count')
        
    with t2_col3_3:
        st.subheader('주간별 총 부정 횟수 집계 그래프')
        st.text('최근 주차 집계 횟수 : ' + str(Emotion_Stat_Dataset_Search_Result_3.sort_values('Week', ascending = False).reset_index(drop=True).loc[0, 'Negative_Count']) + '회')
        st.bar_chart(data = Emotion_Stat_Dataset_Search_Result_3, x = 'Week', y = 'Negative_Count')

        
    with st.expander("감정분류 데이터 차트 상세보기"):
        st.subheader('상세 차트: ' + tab2_selectbox)
        t2_col4_1, t2_col4_2 = st.columns([0.5, 0.5], gap = columns_gap)
        with t2_col4_1:
            st.text('시간대별 상세 차트')
            st.table(Emotion_Stat_Dataset_Search_Result_1[['Start_Time', 'End_Time', 'Negative_Count']].sort_values('End_Time', ascending=False).head(10))
        
        with t2_col4_2:
            st.text('주차별 상세 차트')
            st.table(Emotion_Stat_Dataset_Search_Result_3.sort_values('Week', ascending=False).head(10))    
        
    Emotion_Stat_Dataset_Search_Result_1 = None
    Emotion_Stat_Dataset_Search_Result_2 = None
    Emotion_Stat_Dataset_Search_Result_3 = None
        
###########################
with tab3: # 1인가구 집단 통계
    st.cache_data.clear()
    st.cache_resource.clear()
    t3_col1_1, t3_col1_2, t3_col1_3, t3_col1_4 = st.columns([0.2,0.2,0.2,0.2]) # 조회할 값 선택
    with t3_col1_1:
        st.subheader('**관심 집단 선택**')
        group = st.selectbox(" ", ['커뮤니케이션이 적은 집단','평일 외출이 적은 집단','휴일 외출이 적은 집단','출근소요시간 및 근무시간이 많은 집단','외출이 매우 적은 집단(전체)','외출이 매우 많은 집단','동영상서비스 이용이 많은 집단','생활서비스 이용이 많은 집단','재정상태에 대한 관심집단','외출-커뮤니케이션이 모두 적은 집단(전체)'])
    with t3_col1_2:
        st.subheader('**성별 선택**')
        gender = st.selectbox(' ', ['남성', '여성'])
    with t3_col1_3:
        st.subheader('**연령대 선택**')
        age = st.selectbox(' ', ['20대', '30대', '40대', '50대', '60대', '70대'])
    with t3_col1_4:
        st.subheader('**자치구 선택**')
        region = st.selectbox(' ', ['강남구', '강동구', '강북구', '강서구', '관악구', '광진구',  '구로구', '금천구', '노원구', '도봉구', '동대문구', '동작구', '마포구', '서대문구', '서초구', '성동구', '성북구', '송파구', '양천구', '영등포구', '용산구', '은평구', '종로구', '중구', '중랑구'])
    st.cache_data.clear()
    st.cache_resource.clear()
    st.header(' ')

#     # 버튼 생성
#     st.markdown(button_style, unsafe_allow_html=True)
#     if st.button('선택'):
#         st.write(region, age, gender, group)
#     else:
#         st.write('버튼을 클릭하세요.')
    
#     st.header(' ')
#     st.header(' ')

    t3_col2_1, _, t3_col2_2 = st.columns([0.6, 0.1, 0.3], gap = columns_gap) # 시계열차트
    with t3_col2_1:
        st.subheader(f'**{group}에 속하는 {region} {age} {gender} 1인가구의 수**')
        Lone_Person_Dataset_Loader(group, region, gender, age)
        st.caption('datetime: 날짜 / count: 1인 가구의 수 / actual: 실제값 / predict: 과거 데이터 기반 예측값')
    with t3_col2_2:
        st.subheader('**미래 예측값**')
        st.header(' ')
        pred(group, region, gender, age)

    st.header(' ')
    st.header(' ')

    
    t3_col3_1, t3_col3_2 = st.columns([0.5, 0.5], gap = columns_gap) # 파이차트
    with t3_col3_1:
        st.subheader(f'**현재 {region} {age} {gender} 1인가구 집단 비중**')
        fig = piechart(region, gender, age)
        st.plotly_chart(fig)  

    with t3_col3_2:
        st.subheader(f'**미래 {region} {age} {gender} 1인가구 집단 비중**')
        fig2 = piechart_pred(region, gender, age)
        st.plotly_chart(fig2) 
        
        
###########################
with tab4: # 대상자 정보 및 수정
    st.cache_data.clear()
    st.cache_resource.clear()
    st.subheader('등록된 정보 보기')
    tab4_selectbox_1 = st.selectbox('대상자 선택', Person_Dataset['Name'].unique(), key = 'tab4_대상자선택_1')
    tab4_Dataset_Index = list(Person_Dataset.loc[Person_Dataset['Name'] == tab4_selectbox_1].index)[0]
    tab4_Dataset = Person_Dataset.loc[tab4_Dataset_Index]
    
    t4_col1_1, t4_col1_2 = st.columns([0.5, 0.5], gap = columns_gap)
    with t4_col1_1:
        t4_Name = st.text_input('성명', Person_Dataset.loc[tab4_Dataset_Index, 'Name'])
    with t4_col1_2:
        t4_Serial_Num = st.text_input('시리얼 번호', Person_Dataset.loc[tab4_Dataset_Index,'IoT_Serial_Num'])
        
    t4_col2_1, t4_col2_2 = st.columns([0.5, 0.5], gap = columns_gap)
    with t4_col2_1:
        t4_Emotion_Type = st.text_input('감정분석 기준 분류', Person_Dataset.loc[tab4_Dataset_Index,'Emotion_Type'])
    with t4_col2_2:
        t4_IoT_Type = st.text_input('IoT센서 기준 분류', Person_Dataset.loc[tab4_Dataset_Index,'IoT_Type'])
        
    t4_col3_1, t4_col3_2 = st.columns([0.5, 0.5], gap = columns_gap)
    with t4_col3_1:
        t4_Phone_Num = st.text_input('전화번호(P.H.)', Person_Dataset.loc[tab4_Dataset_Index,'Phone_Num'])
    with t4_col3_2:
        t4_Home_Address = st.text_input('자택 주소', Person_Dataset.loc[tab4_Dataset_Index,'Home_Address'])
        
    t4_AutoSet_Control = st.checkbox('본 대상자는 AI 또는 설정된 기준에 따라 자동 분류합니다. (해제 시 수동조정 필요)', bool(Person_Dataset.loc[tab4_Dataset_Index,'Non_AutoSet']))
        
    st.subheader("")        
    st.subheader('변경사유 입력')
    st.text('등록된 대상자의 정보를 변경하려 할 경우 필수적으로 사유를 입력해야 합니다!')
    t4_Change_Reason = st.text_input('1111', label_visibility="collapsed")
    
    if t4_Change_Reason != "":
        t4_Change_Reason_Accept_Button_Bool = False
    else:
        t4_Change_Reason_Accept_Button_Bool = True
        
    if st.button('정보 변경', disabled = t4_Change_Reason_Accept_Button_Bool):
        Person_Dataset.loc[tab4_Dataset_Index] = [t4_Name, t4_Serial_Num, t4_Emotion_Type, t4_IoT_Type, t4_Phone_Num, t4_Home_Address, t4_AutoSet_Control]
        Info_Change_Reason_Dict = {'Time' : datetime.datetime.now(), 'Changed_Person' : '관리자', 'Name' : t4_Name, 'Changed_Reason': t4_Change_Reason,'IoT_Serial_Num' : t4_Serial_Num,
                                  'Non_AutoSet': t4_AutoSet_Control, 'Emotion_Type': t4_Emotion_Type, 'IoT_Type' : t4_IoT_Type, 'Phone_Num': t4_Phone_Num, 'Home_Address': t4_Home_Address}
        Person_Dataset.to_excel('./Data/Person_Data/Person_Dataset.xlsx', index=False)
        Person_Dataset = pd.read_excel('./Data/Person_Data/Person_Dataset.xlsx')
        Info_Change_Reason_Dataset.loc[len(Info_Change_Reason_Dataset)] = Info_Change_Reason_Dict
        Info_Change_Reason_Dataset.to_excel('./Data/Person_Data/Info_Change_Reason_Dataset.xlsx', index=False)
        Info_Change_Reason_Dataset = pd.read_excel('./Data/Person_Data/Info_Change_Reason_Dataset.xlsx')
        t4_Change_Reason = None
        st.write('정상적으로 정보가 반영되었습니다!')
    
    st.subheader("")
    with st.expander("최근 정보 변경 이력 상세보기"):
        st.table(Info_Change_Reason_Dataset.loc[Info_Change_Reason_Dataset['Name'] == tab4_selectbox_1].sort_values('Time', ascending=False).head(5))
    st.subheader("")
    with st.expander("IoT 센서 상세보기 및 고장접수"):
        Temp_IoT_Sensor_Chosen_Dataset = IoT_Sensor_Info_Dataset.loc[IoT_Sensor_Info_Dataset['IoT_Serial_Num'] == Person_Dataset.loc[tab4_Dataset_Index,'IoT_Serial_Num']].sort_values('Sensor_Type', ascending=True)
        st.table(Temp_IoT_Sensor_Chosen_Dataset)
        st.caption('IoT_Sensor_Num: 1인가구 개개인에게 부여된 식별코드 / Sensor_Type: 센서 종류, 전력(Electro) 또는 조도(Light) / Sensor_Count: 센서 종류 별 등록순서 / Sensor_Num: 해당 센서의 식별코드')
        st.subheader("")
        st.subheader("고장 접수")
        st.caption('대상자를 선택하시려면 페이지 상단에서 검색 및 선택해주세요.')

        t4_col4_1, t4_col4_2 = st.columns([0.5, 0.5], gap = columns_gap)
        with t4_col4_1:
            t4_Fix_Name = st.text_input('성명', Person_Dataset.loc[tab4_Dataset_Index, 'Name'], key = 'tab4_고장_이름')
        with t4_col4_2:
            t4_Fix_Sensor_Num = st.selectbox('센서 선택', Temp_IoT_Sensor_Chosen_Dataset['Sensor_Num'].unique(), key = 'tab4_고장_센서 선택')
                
        t4_col5_1, t4_col5_2 = st.columns([0.5, 0.5], gap = columns_gap)
        with t4_col5_1:
            t4_Fix_Request_Reason = st.text_input('고장 상세 설명')
        with t4_col5_2:
            st.text('')
            st.text('')
            t4_button_Click = st.button('수리점검 접수 신청', key = 'tab4_고장_접수 신청')
        if t4_button_Click:
            st.success('현 대상자 '+ t4_Fix_Name + '의 ' + t4_Fix_Sensor_Num + '의 수리점검 신청이 완료되었습니다!', icon="✅")

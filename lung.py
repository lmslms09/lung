import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ------------------------------------------------------------------
# 1. 한글 폰트 설정 (Pretendard)
# ------------------------------------------------------------------
# 폴더에 있는 Pretendard 파일명을 확인하고 정확히 매칭해주세요.
# 예시: 'Pretendard-Regular.ttf' 또는 'Pretendard-Medium.ttf' 등
font_filename = 'font.ttf' 

if os.path.exists(font_filename):
    # 폰트 등록
    font_prop = fm.FontProperties(fname=font_filename)
    # Matplotlib의 기본 폰트 패밀리 이름을 Pretendard의 실제 폰트 이름으로 설정
    plt.rc('font', family=font_prop.get_name())
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
else:
    st.warning(f"⚠️ 폴더 내에 '{font_filename}' 파일이 발견되지 않아 기본 폰트를 사용합니다.")

# ------------------------------------------------------------------
# 2. 모델 및 스케일러 로드
# ------------------------------------------------------------------
@st.cache_resource # 웹 앱이 새로고침될 때마다 모델을 다시 로드하지 않도록 캐싱
def load_artifacts():
    model = joblib.load('lung_model.pkl')
    scaler = joblib.load('lung_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("❌ 'lung_model.pkl' 또는 'lung_scaler.pkl' 파일이 같은 폴더에 없습니다. 확인해주세요.")
    st.stop()

# 테스트용 기존 데이터 (실제 데이터프레임 df가 있다면 이를 로드하는 코드로 대체해야 합니다)
# 여기서는 시각화를 위해 임의의 가상 데이터를 생성했습니다.
@st.cache_data
def get_mock_data():
    # 실제 환경에서는 pd.read_csv('your_data.csv') 등으로 대체하세요.
    import numpy as np
    np.random.seed(42)
    mock_df = pd.DataFrame({
        '흡연': np.random.uniform(0, 10, 100),
        '음주량': np.random.uniform(0, 10, 100),
        '주거환경점수': np.random.uniform(0, 100, 100),
        'cluster': np.random.choice([0, 1, 2], 100)
    })
    return mock_df

df = get_mock_data()

# ------------------------------------------------------------------
# 3. 스트림릿 UI 구성
# ------------------------------------------------------------------
st.title("🫁 폐 건강 군집 예측 및 분석 시스템")
st.write("환자의 정보를 입력하면 어떤 군집에 속하는지 예측하고 시각화합니다.")

st.sidebar.header("📋 환자 정보 입력")

# 새로운 환자 데이터 입력 받기
smokes = st.sidebar.number_input("흡연 입력 (값)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)
alcohol = st.sidebar.number_input("음주량 입력 (값)", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
area_q = st.sidebar.number_input("주거환경점수 입력", min_value=0.0, max_value=100.0, value=50.0, step=1.0)

# 예측 버튼
if st.sidebar.button("군집 예측하기"):
    
    # 데이터프레임 변환
    new_patient = pd.DataFrame([[smokes, alcohol, area_q]], columns=['흡연', '음주량', '주거환경점수'])
    
    # 스케일링 및 예측
    new_patient_scaled = scaler.transform(new_patient)
    pred_cluster = model.predict(new_patient_scaled)
    
    # 결과 출력
    st.subheader("🔮 예측 결과")
    st.success(f"이 환자는 **{pred_cluster[0]}번 군집**에 속합니다.")
    
    # ------------------------------------------------------------------
    # 4. 시각화 (새로운 값의 위치 출력)
    # ------------------------------------------------------------------
    st.subheader("📊 군집 분포 내 환자 위치")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 기존 데이터 산점도
    scatter = ax.scatter(df['흡연'], df['음주량'], c=df['cluster'], cmap='viridis', alpha=0.5)
    
    # 새 환자 표시 (X 표시)
    ax.scatter(smokes, alcohol, c='red', s=300, marker='X', edgecolors='black', label='현재 환자')
    
    # 축 이름 및 레이블 설정 (폰트 프로퍼티 적용)
    ax.set_xlabel('흡연', fontproperties=font_prop if 'font_prop' in locals() else None, fontsize=12)
    ax.set_ylabel('음주량', fontproperties=font_prop if 'font_prop' in locals() else None, fontsize=12)
    ax.set_title('흡연 및 음주량에 따른 군집 분포', fontproperties=font_prop if 'font_prop' in locals() else None, fontsize=14)
    
    # 범례 추가
    ax.legend(prop=font_prop if 'font_prop' in locals() else None)
    
    # 칼라바 추가 (군집 표시용)
    cbar = fig.colorbar(scatter, ax=ax, ticks=[0, 1, 2])
    cbar.set_label('군집 (Cluster)', fontproperties=font_prop if 'font_prop' in locals() else None)
    
    # 스트림릿에 그래프 출력
    st.pyplot(fig)
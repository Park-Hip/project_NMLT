import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Language translations
TRANSLATIONS = {
    "en": {
        "page_title": "Flood Probability Predictor",
        "title": "🌊 Flood Probability Prediction",
        "intro": "Enter the environmental and infrastructure factors below to predict flood probability.",
        "input_features": "📊 Input Features",
        "predict_button": "🔮 Predict Flood Probability",
        "result_title": "📈 Prediction Result",
        "flood_probability": "Flood Probability",
        "low_risk": "🟢 **Low Risk** - The area has low flood probability.",
        "moderate_risk": "🟡 **Moderate Risk** - The area has moderate flood probability.",
        "high_risk": "🔴 **High Risk** - The area has high flood probability!",
        "about": "ℹ️ About",
        "about_text": """
        This application predicts flood probability based on various 
        environmental and infrastructure factors using a Linear Regression model.
        
        **Features Used:**
        - Environmental factors (Monsoon, Climate, etc.)
        - Infrastructure quality (Dams, Drainage, etc.)
        - Human factors (Urbanization, Population, etc.)
        - Administrative factors (Planning, Political, etc.)
        
        **Model:** Linear Regression
        """,
        "developed_for": "**Developed for:** Nhập Môn Lập Trình - Đồ Án Cuối Kì",
        "language": "🌐 Language",
        "features": {
            "MonsoonIntensity": "Intensity of monsoon (0-15)",
            "TopographyDrainage": "Quality of topography drainage (0-15)",
            "RiverManagement": "Quality of river management (0-15)",
            "Deforestation": "Level of deforestation (0-15)",
            "Urbanization": "Level of urbanization (0-15)",
            "ClimateChange": "Impact of climate change (0-15)",
            "DamsQuality": "Quality of dams (0-15)",
            "Siltation": "Level of siltation (0-15)",
            "AgriculturalPractices": "Impact of agricultural practices (0-15)",
            "Encroachments": "Level of encroachments (0-15)",
            "IneffectiveDisasterPreparedness": "Level of ineffective disaster preparedness (0-15)",
            "DrainageSystems": "Quality of drainage systems (0-15)",
            "CoastalVulnerability": "Level of coastal vulnerability (0-15)",
            "Landslides": "Risk of landslides (0-15)",
            "Watersheds": "Watershed conditions (0-15)",
            "DeterioratingInfrastructure": "Level of deteriorating infrastructure (0-15)",
            "PopulationScore": "Population density score (0-15)",
            "WetlandLoss": "Level of wetland loss (0-15)",
            "InadequatePlanning": "Level of inadequate planning (0-15)",
            "PoliticalFactors": "Impact of political factors (0-15)"
        },
        "feature_names": {
            "MonsoonIntensity": "Monsoon Intensity",
            "TopographyDrainage": "Topography Drainage",
            "RiverManagement": "River Management",
            "Deforestation": "Deforestation",
            "Urbanization": "Urbanization",
            "ClimateChange": "Climate Change",
            "DamsQuality": "Dams Quality",
            "Siltation": "Siltation",
            "AgriculturalPractices": "Agricultural Practices",
            "Encroachments": "Encroachments",
            "IneffectiveDisasterPreparedness": "Ineffective Disaster Preparedness",
            "DrainageSystems": "Drainage Systems",
            "CoastalVulnerability": "Coastal Vulnerability",
            "Landslides": "Landslides",
            "Watersheds": "Watersheds",
            "DeterioratingInfrastructure": "Deteriorating Infrastructure",
            "PopulationScore": "Population Score",
            "WetlandLoss": "Wetland Loss",
            "InadequatePlanning": "Inadequate Planning",
            "PoliticalFactors": "Political Factors"
        }
    },
    "vi": {
        "page_title": "Dự Đoán Xác Suất Lũ Lụt",
        "title": "🌊 Dự Đoán Xác Suất Lũ Lụt",
        "intro": "Nhập các yếu tố môi trường và cơ sở hạ tầng bên dưới để dự đoán xác suất lũ lụt.",
        "input_features": "📊 Các Thông Số Đầu Vào",
        "predict_button": "🔮 Dự Đoán Xác Suất Lũ Lụt",
        "result_title": "📈 Kết Quả Dự Đoán",
        "flood_probability": "Xác Suất Lũ Lụt",
        "low_risk": "🟢 **Rủi ro thấp** - Khu vực có xác suất lũ lụt thấp.",
        "moderate_risk": "🟡 **Rủi ro trung bình** - Khu vực có xác suất lũ lụt trung bình.",
        "high_risk": "🔴 **Rủi ro cao** - Khu vực có xác suất lũ lụt cao!",
        "about": "ℹ️ Giới Thiệu",
        "about_text": """
        Ứng dụng này dự đoán xác suất lũ lụt dựa trên các yếu tố 
        môi trường và cơ sở hạ tầng sử dụng mô hình Hồi quy Tuyến tính.
        
        **Các yếu tố sử dụng:**
        - Yếu tố môi trường (Gió mùa, Khí hậu, v.v.)
        - Chất lượng cơ sở hạ tầng (Đập, Thoát nước, v.v.)
        - Yếu tố con người (Đô thị hóa, Dân số, v.v.)
        - Yếu tố hành chính (Quy hoạch, Chính trị, v.v.)
        
        **Mô hình:** Hồi quy Tuyến tính
        """,
        "developed_for": "**Phát triển cho:** Nhập Môn Lập Trình - Đồ Án Cuối Kì",
        "language": "🌐 Ngôn ngữ",
        "features": {
            "MonsoonIntensity": "Cường độ gió mùa (0-15)",
            "TopographyDrainage": "Thoát nước địa hình (0-15)",
            "RiverManagement": "Quản lý sông ngòi (0-15)",
            "Deforestation": "Mức độ phá rừng (0-15)",
            "Urbanization": "Mức độ đô thị hóa (0-15)",
            "ClimateChange": "Tác động biến đổi khí hậu (0-15)",
            "DamsQuality": "Chất lượng đập (0-15)",
            "Siltation": "Mức độ bồi lắng (0-15)",
            "AgriculturalPractices": "Tác động canh tác nông nghiệp (0-15)",
            "Encroachments": "Mức độ lấn chiếm (0-15)",
            "IneffectiveDisasterPreparedness": "Chuẩn bị thiên tai kém (0-15)",
            "DrainageSystems": "Chất lượng hệ thống thoát nước (0-15)",
            "CoastalVulnerability": "Mức độ dễ tổn thương ven biển (0-15)",
            "Landslides": "Nguy cơ sạt lở (0-15)",
            "Watersheds": "Điều kiện lưu vực (0-15)",
            "DeterioratingInfrastructure": "Cơ sở hạ tầng xuống cấp (0-15)",
            "PopulationScore": "Điểm mật độ dân số (0-15)",
            "WetlandLoss": "Mức độ mất đất ngập nước (0-15)",
            "InadequatePlanning": "Quy hoạch không đầy đủ (0-15)",
            "PoliticalFactors": "Tác động yếu tố chính trị (0-15)"
        },
        "feature_names": {
            "MonsoonIntensity": "Cường Độ Gió Mùa",
            "TopographyDrainage": "Thoát Nước Địa Hình",
            "RiverManagement": "Quản Lý Sông Ngòi",
            "Deforestation": "Phá Rừng",
            "Urbanization": "Đô Thị Hóa",
            "ClimateChange": "Biến Đổi Khí Hậu",
            "DamsQuality": "Chất Lượng Đập",
            "Siltation": "Bồi Lắng",
            "AgriculturalPractices": "Canh Tác Nông Nghiệp",
            "Encroachments": "Lấn Chiếm",
            "IneffectiveDisasterPreparedness": "Chuẩn Bị Thiên Tai Kém",
            "DrainageSystems": "Hệ Thống Thoát Nước",
            "CoastalVulnerability": "Tổn Thương Ven Biển",
            "Landslides": "Sạt Lở Đất",
            "Watersheds": "Lưu Vực",
            "DeterioratingInfrastructure": "Hạ Tầng Xuống Cấp",
            "PopulationScore": "Điểm Dân Số",
            "WetlandLoss": "Mất Đất Ngập Nước",
            "InadequatePlanning": "Quy Hoạch Kém",
            "PoliticalFactors": "Yếu Tố Chính Trị"
        }
    }
}

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("flood_model.pkl")

# Data engineering function (same as in notebook)
def data_engineer(df):
    df = df.copy()
    df['Water_Capacity'] = df['DrainageSystems'] + df['DamsQuality'] + df['RiverManagement']
    df['Water_Load'] = df['MonsoonIntensity'] + df['ClimateChange'] + df['Siltation']
    df['Hydrological_Balance'] = df['Water_Capacity'] - df['Water_Load']
    human_factors = [
        'Deforestation', 'Urbanization', 'AgriculturalPractices',
        'Encroachments', 'PopulationScore', 'WetlandLoss'
    ]
    df['Anthropogenic_Pressure'] = df[human_factors].mean(axis=1)
    df['Admin_Gridlock'] = df['PoliticalFactors'] * df['InadequatePlanning']
    return df

# Preprocessing pipeline (same as in notebook)
def preprocess_pipeline(X):
    X = data_engineer(X)
    scaler = StandardScaler()
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Load training data for fitting scaler/imputer
@st.cache_resource
def load_training_data():
    df = pd.read_csv("data/flood_data.csv")
    X = df.drop("FloodProbability", axis=1)
    return X

# Preprocess with fitted scaler from training data
def preprocess_with_training_fit(X_new, X_train):
    # Apply data engineering to both
    X_train_eng = data_engineer(X_train)
    X_new_eng = data_engineer(X_new)
    
    # Fit imputer and scaler on training data
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    scaler = StandardScaler()
    
    X_train_imputed = imputer.fit_transform(X_train_eng)
    scaler.fit(X_train_imputed)
    
    # Transform new data
    X_new_imputed = imputer.transform(X_new_eng)
    X_new_scaled = scaler.transform(X_new_imputed)
    
    return X_new_scaled

# Main app
def main():
    st.set_page_config(
        page_title="Flood Probability Predictor",
        page_icon="🌊",
        layout="wide"
    )
    
    # Language selector in sidebar
    with st.sidebar:
        lang = st.selectbox(
            "🌐 Language / Ngôn ngữ",
            options=["en", "vi"],
            format_func=lambda x: "English" if x == "en" else "Tiếng Việt"
        )
    
    t = TRANSLATIONS[lang]
    
    st.title(t["title"])
    st.markdown("---")
    st.write(t["intro"])
    
    # Load model and training data
    model = load_model()
    X_train = load_training_data()
    
    # Feature descriptions from translations
    feature_info = t["features"]
    feature_names = t["feature_names"]
    
    # Create input form with columns
    st.subheader(t["input_features"])
    
    col1, col2, col3, col4 = st.columns(4)
    
    features = list(feature_info.keys())
    input_values = {}
    
    # Distribute features across columns
    for i, feature in enumerate(features):
        if i % 4 == 0:
            with col1:
                input_values[feature] = st.slider(
                    feature_names[feature],
                    min_value=0,
                    max_value=15,
                    value=5,
                    help=feature_info[feature],
                    key=feature
                )
        elif i % 4 == 1:
            with col2:
                input_values[feature] = st.slider(
                    feature_names[feature],
                    min_value=0,
                    max_value=15,
                    value=5,
                    help=feature_info[feature],
                    key=feature
                )
        elif i % 4 == 2:
            with col3:
                input_values[feature] = st.slider(
                    feature_names[feature],
                    min_value=0,
                    max_value=15,
                    value=5,
                    help=feature_info[feature],
                    key=feature
                )
        else:
            with col4:
                input_values[feature] = st.slider(
                    feature_names[feature],
                    min_value=0,
                    max_value=15,
                    value=5,
                    help=feature_info[feature],
                    key=feature
                )
    
    st.markdown("---")
    
    # Predict button
    if st.button(t["predict_button"], type="primary", use_container_width=True):
        # Create DataFrame from input
        input_df = pd.DataFrame([input_values])
        
        # Preprocess input
        input_processed = preprocess_with_training_fit(input_df, X_train)
        
        # Make prediction
        prediction = model.predict(input_processed)[0]
        
        # Ensure probability is between 0 and 1
        prediction = max(0, min(1, prediction))
        
        # Display result
        st.markdown("---")
        st.subheader(t["result_title"])
        
        col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
        
        with col_result2:
            # Display prediction with color coding
            if prediction < 0.3:
                st.success(f"### {t['flood_probability']}: {prediction:.2%}")
                st.write(t["low_risk"])
            elif prediction < 0.6:
                st.warning(f"### {t['flood_probability']}: {prediction:.2%}")
                st.write(t["moderate_risk"])
            else:
                st.error(f"### {t['flood_probability']}: {prediction:.2%}")
                st.write(t["high_risk"])
            
            # Progress bar for visualization
            st.progress(prediction)
    
    # Sidebar info
    with st.sidebar:
        st.header(t["about"])
        st.write(t["about_text"])
        
        st.markdown("---")
        st.write(t["developed_for"])

if __name__ == "__main__":
    main()

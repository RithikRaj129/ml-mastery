# streamlit_app.py
# Run with: streamlit run streamlit_app.py
# Deploy free: streamlit share (push to GitHub, connect at share.streamlit.io)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title='Cancer Classifier Dashboard',
    page_icon=' ',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── Load model (cached - only loads once) ─────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('model_pipeline.joblib')

@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df['diagnosis'] = df['target'].map({0:'Malignant', 1:'Benign'})
    return df, data

model = load_model()
df, data = load_data()

# ── Sidebar ────────────────────────────────────────────────────────
st.sidebar.title('  Cancer Classifier')
st.sidebar.markdown('Real-time breast cancer malignancy prediction')
page = st.sidebar.radio('Navigation', ['  Predict', '  Data Explorer', '  Model Performance'])

# ═══════════════════════════════════════════════════════════
if page == '  Predict':
    st.title('Breast Cancer Malignancy Prediction')
    st.markdown('Enter cell nucleus measurements to predict malignancy.')

    st.subheader('  Input Features')
    col1, col2, col3 = st.columns(3)

    # Create sliders for key features
    with col1:
        mean_radius    = st.slider('Mean Radius',    6.0, 28.0, 14.0, 0.1)
        mean_texture   = st.slider('Mean Texture',   9.0, 40.0, 19.0, 0.1)
        mean_perimeter = st.slider('Mean Perimeter', 43.0, 190.0, 92.0, 0.5)
        mean_area      = st.slider('Mean Area',      143.0, 2502.0, 655.0, 1.0)
        mean_smooth    = st.slider('Mean Smoothness', 0.05, 0.16, 0.10, 0.001)

    with col2:
        mean_compact   = st.slider('Mean Compactness', 0.02, 0.35, 0.10, 0.001)
        mean_concave   = st.slider('Mean Concavity',   0.0,  0.43, 0.09, 0.001)
        mean_cp        = st.slider('Mean Concave Pts', 0.0,  0.20, 0.05, 0.001)
        mean_sym       = st.slider('Mean Symmetry',    0.10, 0.30, 0.18, 0.001)
        mean_frac      = st.slider('Mean Fractal Dim', 0.05, 0.10, 0.06, 0.001)

    with col3:
        st.markdown('**Use default values for remaining 20 features**')
        use_sample = st.selectbox('Or use example patient:', ['Custom', 'Patient A (Benign)', 'Patient B (Malignant)'])
        if use_sample == 'Patient A (Benign)':
            sample_idx = np.where(data.target == 1)[0][0]
            features = data.data[sample_idx]
        elif use_sample == 'Patient B (Malignant)':
            sample_idx = np.where(data.target == 0)[0][0]
            features = data.data[sample_idx]
        else:
            # Fill remaining 20 features with dataset means
            features = data.data.mean(axis=0).copy()
            features[0]  = mean_radius
            features[1]  = mean_texture
            features[2]  = mean_perimeter
            features[3]  = mean_area
            features[4]  = mean_smooth
            features[5]  = mean_compact
            features[6]  = mean_concave
            features[7]  = mean_cp
            features[8]  = mean_sym
            features[9]  = mean_frac

    # ── Make prediction ───────────────────────────────────────────
    X = features.reshape(1, -1)
    prediction  = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    pred_label  = 'Benign' if prediction == 1 else 'Malignant'
    confidence  = max(probability)

    # ── Display result ────────────────────────────────────────────
    st.markdown('---')
    st.subheader('  Prediction Result')
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        color = ' ' if pred_label=='Benign' else ' '
        st.metric('Diagnosis', f'{color} {pred_label}')

    with res_col2:
        st.metric('Confidence', f'{confidence:.1%}')

    with res_col3:
        st.metric('P(Malignant)', f'{probability[0]:.1%}')

    # Probability gauge
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=probability[0]*100,
        number={'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#E53935' if pred_label=='Malignant' else '#43A047'},
            'steps': [
                {'range': [0,50],  'color': '#E8F5E9'},
                {'range': [50,100],'color': '#FFEBEE'},
            ],
            'threshold': {'value': 50, 'line': {'color':'black','width':3}},
        },
        title={'text': 'Malignancy Probability'},
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

    if pred_label == 'Malignant':
        st.warning('  High malignancy probability. Recommend immediate specialist review.')
    else:
        st.success('  Low malignancy probability. Continue routine monitoring.')

    st.caption('This tool is for research purposes only. Not a medical device.')

# ═══════════════════════════════════════════════════════════
elif page == '  Data Explorer':
    st.title('  Dataset Explorer')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Samples', len(df))
    col2.metric('Features', len(data.feature_names))
    col3.metric('Malignant', (df.target==0).sum())
    col4.metric('Benign', (df.target==1).sum())

    # Feature distribution
    selected_feature = st.selectbox('Feature to explore:', data.feature_names)
    fig = px.histogram(df, x=selected_feature, color='diagnosis',
                        barmode='overlay', nbins=40,
                        color_discrete_map={'Malignant':'#E53935','Benign':'#1976D2'},
                        title=f'{selected_feature} Distribution by Diagnosis')
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader('Correlation Heatmap (Top 10 Features)')
    top10 = df[list(data.feature_names[:10])].corr()
    fig = px.imshow(top10, color_continuous_scale='RdYlBu_r', zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
elif page == '  Model Performance':
    st.title('  Model Performance')

    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    import seaborn as sns
    import matplotlib.pyplot as plt

    with st.spinner('Computing metrics...'):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, data.data, data.target, cv=cv, scoring='roc_auc')

    col1, col2, col3 = st.columns(3)
    col1.metric('CV AUC (mean)', f'{scores.mean():.4f}')
    col2.metric('CV AUC (std)', f'±{scores.std():.4f}')
    col3.metric('CV AUC (min)', f'{scores.min():.4f}')

    st.success(f'Model achieves {scores.mean():.1%} AUC on 5-fold cross-validation')

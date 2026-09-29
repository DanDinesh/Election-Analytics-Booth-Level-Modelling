import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title='Election Analytics 2026', page_icon='📊', layout='wide')

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR = BASE_DIR / 'models'
DATABASE_DIR = BASE_DIR / 'database'
REPORTS_DIR = BASE_DIR / 'reports'

st.title('📊 Election Analytics and Booth-Level Vote Share Modelling')
st.caption('Historical 2016→2021 model evaluation + descriptive analysis of the supplied 2026 booth dataset')
st.info('This dashboard is for historical modelling and descriptive analytics. The 2026 section should not be treated as a guaranteed prediction of a future election result.')

@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if path.exists() else None

model_results = load_csv(RESULTS_DIR / 'model_comparison.csv')
predictions = load_csv(RESULTS_DIR / 'historical_predictions.csv')
training_data = load_csv(RESULTS_DIR / 'historical_training_dataset.csv')
booth2026 = load_csv(RESULTS_DIR / 'booth_2026_analysis.csv')
summary2026 = load_csv(RESULTS_DIR / '2026_party_summary.csv')
dl_history = load_csv(RESULTS_DIR / 'deep_learning_history.csv')

st.sidebar.header('Navigation')
page = st.sidebar.radio('Go to', [
    'Project Overview',
    'Model Comparison',
    'Historical Predictions',
    'Deep Learning',
    '2026 Booth Analysis',
    'Project Files'
])

if page == 'Project Overview':
    st.header('Project Overview')
    c1, c2, c3, c4 = st.columns(4)

    if training_data is not None:
        c1.metric('Historical matched rows', f'{len(training_data):,}')
        c2.metric('Unique booths', f"{training_data['ps_no'].nunique():,}" if 'ps_no' in training_data else 'N/A')
    else:
        c1.metric('Historical matched rows', 'N/A')
        c2.metric('Unique booths', 'N/A')

    if model_results is not None and not model_results.empty:
        row = model_results.sort_values('RMSE').iloc[0]
        c3.metric('Lowest historical RMSE', f"{row['RMSE']:.4f}")
        c4.metric('Model', str(row['Model']))
    else:
        c3.metric('Lowest historical RMSE', 'N/A')
        c4.metric('Model', 'N/A')

    st.subheader('Workflow')
    st.markdown('**Raw Data → Cleaning → Booth/Party Aggregation → Feature Engineering → Grouped Train/Test Split → ML & DL Models → Evaluation → Dashboard**')
    st.subheader('Models Used')
    st.write('Linear Regression, Random Forest, Gradient Boosting, XGBoost, and MLP Deep Learning.')
    st.subheader('Evaluation Metrics')
    st.write('MAE, RMSE, and R².')

elif page == 'Model Comparison':
    st.header('Historical Model Comparison')
    if model_results is None or model_results.empty:
        st.warning('model_comparison.csv was not found.')
    else:
        st.dataframe(model_results, use_container_width=True)
        ordered = model_results.sort_values('RMSE')
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(ordered['Model'], ordered['RMSE'])
        ax.set_ylabel('RMSE')
        ax.set_title('Historical 2016→2021 RMSE by Model')
        ax.tick_params(axis='x', rotation=25)
        fig.tight_layout()
        st.pyplot(fig)
        row = ordered.iloc[0]
        st.success(f"Lowest historical RMSE in this backtest: {row['Model']} (MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}, R²={row['R2']:.4f})")
        st.caption('R² is a regression statistic, not election accuracy.')

elif page == 'Historical Predictions':
    st.header('Historical Predictions')
    if predictions is None or predictions.empty:
        st.warning('historical_predictions.csv was not found.')
    else:
        st.dataframe(predictions.head(100), use_container_width=True)
        model_cols = [c for c in predictions.columns if c != 'actual_2021']
        selected_model = st.selectbox('Select model prediction', model_cols)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(predictions['actual_2021'], predictions[selected_model], alpha=0.65)
        ax.plot([0, 1], [0, 1], linestyle='--')
        ax.set_xlabel('Actual 2021 Vote Share')
        ax.set_ylabel('Predicted 2021 Vote Share')
        ax.set_title(f'Actual vs Predicted — {selected_model}')
        fig.tight_layout()
        st.pyplot(fig)

elif page == 'Deep Learning':
    st.header('Deep Learning — MLP')
    st.markdown('''
**Architecture**
- Dense(64, ReLU)
- Dropout(0.20)
- Dense(32, ReLU)
- Dropout(0.15)
- Dense(16, ReLU)
- Dense(1, Sigmoid)

**Training**
- Optimizer: Adam
- Learning rate: 0.001
- Loss: Mean Squared Error
- Batch size: 32
- Early stopping enabled
''')
    if dl_history is not None and not dl_history.empty:
        loss_cols = [c for c in ['loss', 'val_loss'] if c in dl_history.columns]
        mae_cols = [c for c in ['mae', 'val_mae'] if c in dl_history.columns]
        if loss_cols:
            st.subheader('Training Loss')
            st.line_chart(dl_history[loss_cols])
        if mae_cols:
            st.subheader('Training MAE')
            st.line_chart(dl_history[mae_cols])
    else:
        st.warning('deep_learning_history.csv was not found.')

elif page == '2026 Booth Analysis':
    st.header('2026 Supplied Booth Data — Descriptive Analysis')
    st.warning('This section summarizes values already present in the supplied 2026 dataset. It is not a future-winner prediction.')

    if summary2026 is not None and not summary2026.empty:
        st.subheader('Supplied Party Summary')
        show_summary = summary2026.copy()
        if 'share' in show_summary.columns:
            show_summary['share_percent'] = show_summary['share'] * 100
        st.dataframe(show_summary, use_container_width=True)
        if {'party', 'share'}.issubset(show_summary.columns):
            st.bar_chart(show_summary.set_index('party')[['share']] * 100)

    if booth2026 is None or booth2026.empty:
        st.warning('booth_2026_analysis.csv was not found.')
    else:
        st.subheader('Booth-Level Explorer')
        min_booth = int(booth2026['booth_no'].min())
        max_booth = int(booth2026['booth_no'].max())
        default_high = min(max_booth, min_booth + 50)
        booth_range = st.slider('Booth number range', min_value=min_booth, max_value=max_booth, value=(min_booth, default_high))
        filtered = booth2026[booth2026['booth_no'].between(booth_range[0], booth_range[1])].copy()
        cols = [c for c in ['booth_no', 'booth_name', 'polled', 'inc_share', 'admk_share', 'tvk_share', 'other_share'] if c in filtered.columns]
        st.dataframe(filtered[cols], use_container_width=True)
        share_cols = [c for c in ['inc_share', 'admk_share', 'tvk_share'] if c in filtered.columns]
        if share_cols:
            st.line_chart(filtered.set_index('booth_no')[share_cols] * 100)
        st.download_button('Download filtered booth data', filtered.to_csv(index=False).encode('utf-8'), file_name='filtered_booth_2026.csv', mime='text/csv')

elif page == 'Project Files':
    st.header('Project Files')
    folders = {'Results': RESULTS_DIR, 'Models': MODELS_DIR, 'Database': DATABASE_DIR, 'Reports': REPORTS_DIR}
    for name, folder in folders.items():
        st.subheader(name)
        if folder.exists():
            files = sorted([p.name for p in folder.iterdir() if p.is_file()])
            if files:
                for f in files:
                    st.write('•', f)
            else:
                st.write('No files found.')
        else:
            st.write('Folder not found.')

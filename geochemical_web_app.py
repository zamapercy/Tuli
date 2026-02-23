import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import io
from datetime import datetime

# Configure Streamlit
st.set_page_config(
    page_title="Geochemical Data Plotter",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧪 Geochemical Data Plotter")
st.markdown("Interactive visualization tool for geochemical data analysis")

# File upload section
uploaded_file = st.file_uploader("Upload your geochemical Excel file (.xls or .xlsx)", type=['xls', 'xlsx'])

# Load data
@st.cache_data
def load_data(file):
    try:
        excel_file = pd.ExcelFile(file)
        sheet_names = excel_file.sheet_names
        
        # Load first sheet
        df = pd.read_excel(file, sheet_name=0)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
        
        # Convert numeric columns
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        return df, numeric_cols, sheet_names
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Check if file is uploaded
if uploaded_file is None:
    st.info("👆 Please upload your Excel file to get started")
    st.stop()

# Load data and check
df, numeric_cols, sheet_names = load_data(uploaded_file)

if df is None:
    st.error("Failed to load data. Please check your file format.")
    st.stop()

# Display data info in sidebar
with st.sidebar:
    st.markdown("### 📊 Data Information")
    st.metric("Total Rows", len(df))
    st.metric("Total Columns", len(df.columns))
    st.metric("Numeric Variables", len(numeric_cols))
    
    with st.expander("View Column Names"):
        st.write(sorted(numeric_cols))
    
    st.markdown("---")
    st.markdown("### 🎨 Plot Configuration")

# Main content
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("### Scatter Plot")
    
    # Column selection
    col_x, col_y = st.columns(2)
    with col_x:
        x_var = st.selectbox("X Variable", sorted(numeric_cols), key="x_scatter")
    with col_y:
        y_var = st.selectbox("Y Variable", sorted(numeric_cols), index=1 if len(numeric_cols) > 1 else 0, key="y_scatter")
    
    # Create scatter plot
    data_scatter = df[[x_var, y_var]].dropna()
    
    if len(data_scatter) > 0:
        fig_scatter = px.scatter(
            x=data_scatter[x_var],
            y=data_scatter[y_var],
            title=f"{y_var} vs {x_var}",
            labels={x_var: x_var, y_var: y_var},
            trendline="ols",
            opacity=0.7
        )
        
        fig_scatter.update_traces(marker=dict(size=8, color='#1f77b4'))
        fig_scatter.update_layout(
            height=500,
            font=dict(size=11),
            showlegend=False,
            hovermode='closest'
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Statistics
        with st.expander("📈 Statistics"):
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.metric(f"{x_var} Mean", f"{data_scatter[x_var].mean():.2f}")
                st.metric(f"{x_var} Std", f"{data_scatter[x_var].std():.2f}")
            with stats_col2:
                st.metric(f"{y_var} Mean", f"{data_scatter[y_var].mean():.2f}")
                st.metric(f"{y_var} Std", f"{data_scatter[y_var].std():.2f}")
    else:
        st.warning("No valid data for selected columns")

with col2:
    st.markdown("### Ternary Plot")
    
    # Ternary plot variables
    ternary_vars = st.multiselect(
        "Select 3 Variables for Ternary Plot",
        sorted(numeric_cols),
        max_selections=3,
        key="ternary_select"
    )
    
    if len(ternary_vars) == 3:
        data_ternary = df[ternary_vars].dropna()
        
        if len(data_ternary) > 0:
            # Normalize to proportions
            total = data_ternary.sum(axis=1)
            data_ternary_norm = data_ternary.div(total, axis=0) * 100
            
            fig_ternary = go.Figure(data=[
                go.Scatterternary(
                    a=data_ternary_norm[ternary_vars[0]],
                    b=data_ternary_norm[ternary_vars[1]],
                    c=data_ternary_norm[ternary_vars[2]],
                    mode='markers',
                    marker=dict(size=8, color='#ff7f0e', opacity=0.7),
                    text=[f"{ternary_vars[0]}: {a:.1f}%<br>{ternary_vars[1]}: {b:.1f}%<br>{ternary_vars[2]}: {c:.1f}%" 
                          for a, b, c in zip(data_ternary_norm[ternary_vars[0]], data_ternary_norm[ternary_vars[1]], data_ternary_norm[ternary_vars[2]])],
                    hovertemplate='%{text}<extra></extra>'
                )
            ])
            
            fig_ternary.update_layout(
                ternary=dict(
                    aaxis=dict(title=ternary_vars[0]),
                    baxis=dict(title=ternary_vars[1]),
                    caxis=dict(title=ternary_vars[2])
                ),
                height=500,
                font=dict(size=10),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig_ternary, use_container_width=True)
        else:
            st.warning("No valid data for selected variables")
    else:
        st.info("Select exactly 3 variables for ternary plot")

# Advanced analysis
st.markdown("---")
st.markdown("### 📉 Advanced Analysis")

tab1, tab2, tab3 = st.tabs(["Correlation Matrix", "Data Table", "Batch Export"])

with tab1:
    # Correlation heatmap
    corr_data = df[numeric_cols].corr()
    fig_corr = px.imshow(
        corr_data,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Correlation Matrix",
        zmin=-1,
        zmax=1,
        labels=dict(color="Correlation")
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    # Data table
    st.markdown("**Numeric Data Summary**")
    display_cols = st.multiselect("Select columns to display", numeric_cols, default=numeric_cols[:5])
    if display_cols:
        st.dataframe(df[display_cols].describe(), use_container_width=True)

with tab3:
    st.markdown("**Generate Multiple Plots**")
    st.info("Create and download multiple scatter plots at once")
    
    batch_x = st.selectbox("X Variable (Batch)", sorted(numeric_cols), key="batch_x")
    batch_y_list = st.multiselect("Y Variables (Batch)", sorted(numeric_cols), key="batch_y")
    
    if st.button("Generate Plots"):
        if batch_y_list:
            plot_files = []
            progress_bar = st.progress(0)
            
            for i, batch_y in enumerate(batch_y_list):
                data = df[[batch_x, batch_y]].dropna()
                
                if len(data) > 0:
                    fig = px.scatter(
                        x=data[batch_x],
                        y=data[batch_y],
                        title=f"{batch_y} vs {batch_x}",
                        trendline="ols"
                    )
                    fig.update_traces(marker=dict(size=6, opacity=0.6))
                    
                    # Save to HTML bytes
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"plot_{batch_x}_vs_{batch_y}_{timestamp}.html"
                    html_bytes = fig.to_html().encode('utf-8')
                    plot_files.append((filename, html_bytes))
                
                progress_bar.progress((i + 1) / len(batch_y_list))
            
            st.success(f"✓ Generated {len(plot_files)} plots")
            
            for filename, html_bytes in plot_files:
                st.download_button(
                    label=f"📥 {filename}",
                    data=html_bytes,
                    file_name=filename,
                    mime="text/html",
                    key=filename
                )

st.markdown("---")
st.markdown("<small>Geochemical Data Plotter • Built with Streamlit</small>", unsafe_allow_html=True)

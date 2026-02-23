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

# Geochemical Standard Plots
st.markdown("---")
st.markdown("### 🧪 Geochemical Analysis Plots")

# Check if required columns exist
has_ni = any('ni' == col or col.startswith('ni_') or col.endswith('_ni') for col in numeric_cols)
has_mgo = any('mgo' == col or 'mgo' in col for col in numeric_cols)
has_cu = any('cu' == col or col.startswith('cu_') or col.endswith('_cu') for col in numeric_cols)
has_la = any('la' == col or col.startswith('la_') or col.endswith('_la') for col in numeric_cols)
has_nb = any('nb' == col or col.startswith('nb_') or col.endswith('_nb') for col in numeric_cols)
has_nd = any('nd' == col or col.startswith('nd_') or col.endswith('_nd') for col in numeric_cols)
has_sio2 = any('sio2' in col or 'sio₂' in col for col in numeric_cols)

# Find depth column (check numeric columns first, then all columns)
depth_col = None
for col in numeric_cols:
    col_lower = col.lower()
    if any(keyword in col_lower for keyword in ['depth', 'depth_m', 'depth(m)', 'depth_meters', 'depth_ft']):
        depth_col = col
        break

if not depth_col:
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['depth', 'depth_m', 'depth(m)', 'depth_meters', 'depth_ft']):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].notna().sum() > 0:
                    depth_col = col
                    numeric_cols.append(col)
                    break
            except:
                pass

has_depth = depth_col is not None

# Find borehole column (usually categorical)
borehole_col = None
for col in df.columns:
    col_lower = col.lower()
    if any(keyword in col_lower for keyword in ['borehole', 'hole', 'bore', 'drillhole', 'dh', 'sample_id', 'location']):
        borehole_col = col
        break

geo_tab1, geo_tab2 = st.tabs(["Ni vs MgO", "Depth Profiles"])

with geo_tab1:
    st.markdown("#### Ni vs MgO Plot")
    if has_ni and has_mgo:
        ni_col = [col for col in numeric_cols if 'ni' == col or col.startswith('ni_') or col.endswith('_ni')][0]
        mgo_col = [col for col in numeric_cols if 'mgo' in col][0]
        
        data_ni_mgo = df[[ni_col, mgo_col]].dropna()
        
        if borehole_col and borehole_col in df.columns:
            # Create copy with borehole info
            plot_df = df[[mgo_col, ni_col, borehole_col]].dropna()
            
            # Color by borehole with distinct colors
            fig_ni_mgo = px.scatter(
                plot_df,
                x=mgo_col,
                y=ni_col,
                color=borehole_col,
                title=f"Ni vs MgO (colored by {borehole_col})",
                labels={ni_col: "Ni (ppm)", mgo_col: "MgO (%)"},
                opacity=0.8,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_ni_mgo.update_traces(marker=dict(size=10, line=dict(width=0.5, color='white')))
        else:
            fig_ni_mgo = px.scatter(
                x=data_ni_mgo[mgo_col],
                y=data_ni_mgo[ni_col],
                title="Ni vs MgO",
                labels={mgo_col: "MgO (%)", ni_col: "Ni (ppm)"},
                opacity=0.7
            )
            fig_ni_mgo.update_traces(marker=dict(size=8, color='#e74c3c'))
        
        fig_ni_mgo.update_layout(height=500, hovermode='closest', legend=dict(title=borehole_col if borehole_col else ""))
        st.plotly_chart(fig_ni_mgo, use_container_width=True)
    else:
        st.warning("⚠️ Ni and/or MgO columns not found in the dataset")
        st.info(f"Available columns: {', '.join(sorted(numeric_cols)[:10])}...")

with geo_tab2:
    st.markdown("#### Depth Profiles by Borehole")
    
    if not has_depth:
        st.warning("⚠️ Depth column not found in the dataset")
        st.info(f"Available columns: {', '.join(sorted(df.columns))}")
        st.info("Looking for columns containing: depth, depth_m, depth(m), depth_meters, or depth_ft")
    else:
        
        if borehole_col:
            selected_borehole = st.selectbox(
                "Select Borehole",
                options=sorted(df[borehole_col].dropna().unique()),
                key="borehole_select"
            )
            df_filtered = df[df[borehole_col] == selected_borehole].copy()
        else:
            st.info("No borehole column detected - showing all data")
            df_filtered = df.copy()
        
        # Create depth profile plots
        depth_plots = []
        plot_titles = []
        plot_colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#27ae60']
        
        if has_mgo:
            mgo_col = [col for col in numeric_cols if 'mgo' in col][0]
            depth_plots.append((mgo_col, "MgO (%)", plot_colors[0]))
        
        if has_ni:
            ni_col = [col for col in numeric_cols if 'ni' == col or col.startswith('ni_') or col.endswith('_ni')][0]
            depth_plots.append((ni_col, "Ni (ppm)", plot_colors[1]))
        
        if has_cu:
            cu_col = [col for col in numeric_cols if 'cu' == col or col.startswith('cu_') or col.endswith('_cu')][0]
            depth_plots.append((cu_col, "Cu (ppm)", plot_colors[2]))
        
        if has_sio2:
            sio2_col = [col for col in numeric_cols if 'sio2' in col or 'sio₂' in col][0]
            depth_plots.append((sio2_col, "SiO₂ (%)", plot_colors[4]))
        
        # La/Nb or La/Nd ratio
        if has_la and has_nb:
            la_col = [col for col in numeric_cols if 'la' == col or col.startswith('la_') or col.endswith('_la')][0]
            nb_col = [col for col in numeric_cols if 'nb' == col or col.startswith('nb_') or col.endswith('_nb')][0]
            df_filtered['la_nb_ratio'] = df_filtered[la_col] / df_filtered[nb_col]
            depth_plots.append(('la_nb_ratio', "La/Nb", plot_colors[3]))
        elif has_la and has_nd:
            la_col = [col for col in numeric_cols if 'la' == col or col.startswith('la_') or col.endswith('_la')][0]
            nd_col = [col for col in numeric_cols if 'nd' == col or col.startswith('nd_') or col.endswith('_nd')][0]
            df_filtered['la_nd_ratio'] = df_filtered[la_col] / df_filtered[nd_col]
            depth_plots.append(('la_nd_ratio', "La/Nd", plot_colors[3]))
        
        # Create plots in a grid
        if len(depth_plots) > 0:
            cols_per_row = 2
            for i in range(0, len(depth_plots), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, plot_info in enumerate(depth_plots[i:i+cols_per_row]):
                    var_col, label, color = plot_info
                    with cols[j]:
                        data_plot = df_filtered[[depth_col, var_col]].dropna()
                        
                        if len(data_plot) > 0:
                            fig = px.scatter(
                                x=data_plot[var_col],
                                y=data_plot[depth_col],
                                title=f"{label} vs Depth",
                                labels={var_col: label, depth_col: "Depth (m)"},
                                opacity=0.8
                            )
                            
                            # Invert y-axis for depth (deeper = lower)
                            fig.update_yaxis(autorange="reversed")
                            fig.update_traces(marker=dict(size=8, color=color, line=dict(width=0.5, color='white')))
                            fig.update_layout(height=400, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"No data for {label}")
        else:
            st.warning("⚠️ Required columns for depth profiles not found")

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

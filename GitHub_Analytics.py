# Import python packages
import streamlit as st
from snowflake.snowpark.context import get_active_session
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


# Get the current credentials
session = get_active_session()


# Page configuration
st.set_page_config(
    page_title="Ericsson - CI/CD Pipeline Tracker",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ericsson Brand Colors
ERICSSON_BLUE = "#0058A3"
ERICSSON_DARK_BLUE = "#002561"
ERICSSON_LIGHT_BLUE = "#0082C8"
ERICSSON_GRAY = "#707070"
ERICSSON_LIGHT_GRAY = "#F4F4F4"
ERICSSON_WHITE = "#FFFFFF"

# Custom CSS for Ericsson branding
st.markdown("""
<style>
    /* Import Ericsson-style fonts */
    @import url('https://fonts.googleapis.com/css2?family=Helvetica+Neue:wght@300;400;500;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif;
    }
    
    /* Main header with Ericsson branding */
    .main-header {
        font-size: 2.5rem;
        font-weight: 300;
        color: #0058A3;
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 4px solid #0058A3;
        margin-bottom: 2rem;
        letter-spacing: 0.5px;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    }
    
    /* Metric containers */
    .metric-container {
        background-color: #F4F4F4;
        padding: 1.5rem;
        border-radius: 4px;
        border-left: 4px solid #0058A3;
        box-shadow: 0 1px 3px rgba(0,88,163,0.1);
    }
    
    /* Section headers with Ericsson styling */
    .section-header {
        font-size: 1.5rem;
        font-weight: 400;
        color: #002561;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #0058A3;
        padding-left: 0.75rem;
        letter-spacing: 0.3px;
    }
    
    /* Expander styling */
    .stExpander {
        background-color: white;
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        margin-bottom: 1rem;
        border-left: 3px solid #0058A3;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #F4F4F4;
        border-right: 2px solid #0058A3;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: #F4F4F4;
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #0058A3;
        font-weight: 500;
    }
    
    /* Metric delta */
    [data-testid="stMetricDelta"] {
        color: #002561;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #0058A3;
        color: white;
        border: none;
        border-radius: 3px;
        font-weight: 500;
        transition: background-color 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #002561;
    }
    
    /* Links */
    a {
        color: #0058A3;
        text-decoration: none;
    }
    
    a:hover {
        color: #002561;
        text-decoration: underline;
    }
    
    /* Footer */
    .ericsson-footer {
        text-align: center;
        padding: 2rem;
        color: #707070;
        font-size: 0.9rem;
        border-top: 2px solid #E0E0E0;
        margin-top: 3rem;
        background-color: #F4F4F4;
    }
    
    /* Headers in general */
    h1, h2, h3 {
        color: #002561;
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    
    /* Status badges */
    .status-success {
        background-color: #66BB6A;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 500;
        font-size: 0.875rem;
    }
    .status-failed {
        background-color: #EF5350;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 500;
        font-size: 0.875rem;
    }
    .status-running {
        background-color: #42A5F5;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 500;
        font-size: 0.875rem;
    }
    .status-canceled {
        background-color: #9E9E9E;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 500;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# Status icons
STATUS_ICONS = {
    'success': '✅',
    'failed': '❌',
    'running': '🔄',
    'canceled': '⚠️',
    'pending': '⏳',
    'manual': '👤',
    'skipped': '⏭️'
}

STAGE_ICONS = {
    'build': '🔨',
    'test': '🧪',
    'staging': '📦',
    'production': '🚀',
    'cleanup': '🧹',
    'review': '👁️',
    'prepare': '⚙️'
}

@st.cache_data
def load_data():
    """Load and preprocess the GitLab CI/CD data"""
    # For Snowflake deployment
    data = session.sql("SELECT * FROM DATABSE.SCHEMA.TABLE")
    df = data.to_pandas()
    
    
    # Convert date columns to datetime
    date_columns = ['JOB_CREATEDAT', 'JOB_STARTEDAT', 'JOB_FINISHEDAT']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Add calculated columns
    df['AGE_HOURS'] = (datetime.now() - df['JOB_CREATEDAT']).dt.total_seconds() / 3600
    df['IS_RECENT'] = df['JOB_CREATEDAT'] >= (datetime.now() - timedelta(days=7))
    df['DURATION_MINUTES'] = df['JOB_DURATION'] / 60
    df['DURATION_HOURS'] = df['JOB_DURATION'] / 3600
    
    # Add success flag
    df['IS_SUCCESS'] = df['PIPELINE_STATUS'] == 'success'
    df['IS_FAILED'] = df['PIPELINE_STATUS'] == 'failed'
    
    return df

def display_metrics(df):
    """Display key CI/CD metrics in columns"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="📊 Total Pipelines",
            value=f"{len(df):,}",
            delta=f"+{len(df[df['IS_RECENT']])} (7d)"
        )
    
    with col2:
        success_rate = (df['IS_SUCCESS'].sum() / len(df) * 100) if len(df) > 0 else 0
        recent_success = (df[df['IS_RECENT']]['IS_SUCCESS'].sum() / len(df[df['IS_RECENT']]) * 100) if len(df[df['IS_RECENT']]) > 0 else 0
        delta_success = recent_success - success_rate
        st.metric(
            label="✅ Success Rate",
            value=f"{success_rate:.1f}%",
            delta=f"{delta_success:+.1f}% (7d)"
        )
    
    with col3:
        failed_jobs = len(df[df['IS_FAILED']])
        st.metric(
            label="❌ Failed Pipelines",
            value=f"{failed_jobs:,}",
            delta=f"{len(df[df['IS_RECENT'] & df['IS_FAILED']])} (7d)",
            delta_color="inverse"
        )
    
    with col4:
        avg_duration = df['DURATION_MINUTES'].mean()
        recent_avg = df[df['IS_RECENT']]['DURATION_MINUTES'].mean()
        st.metric(
            label="⏱️ Avg Duration",
            value=f"{avg_duration:.1f} min",
            delta=f"{recent_avg - avg_duration:+.1f} min (7d)"
        )
    
    with col5:
        running = len(df[df['PIPELINE_STATUS'] == 'running'])
        st.metric(
            label="🔄 Running Now",
            value=f"{running:,}",
            delta=f"{(running/len(df)*100):.1f}%"
        )

def display_pipeline_overview(df):
    """Display pipeline status overview"""
    st.markdown('<div class="section-header">📊 Pipeline Status Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Pipeline status distribution
        status_counts = df['PIPELINE_STATUS'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        # Ericsson color palette
        status_colors = {
            'success': '#66BB6A',
            'failed': '#EF5350',
            'running': '#42A5F5',
            'canceled': '#9E9E9E'
        }
        colors = [status_colors.get(status, '#0058A3') for status in status_counts['Status']]
        
        fig = px.pie(
            status_counts,
            values='Count',
            names='Status',
            title='Pipeline Status Distribution',
            hole=0.4,
            color_discrete_sequence=colors
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            font=dict(family="Helvetica Neue, Arial", size=12, color='#002561'),
            title_font=dict(size=16, color='#002561', family="Helvetica Neue, Arial")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Job status breakdown
        job_status_counts = df['JOB_STATUS'].value_counts().head(8).reset_index()
        job_status_counts.columns = ['Job Status', 'Count']
        
        fig = px.bar(
            job_status_counts,
            x='Count',
            y='Job Status',
            orientation='h',
            title='Job Status Breakdown',
            color='Count',
            color_continuous_scale=[[0, '#E3F2FD'], [0.5, '#0082C8'], [1, '#0058A3']]
        )
        fig.update_layout(
            showlegend=False,
            yaxis={'categoryorder':'total ascending'},
            font=dict(family="Helvetica Neue, Arial", size=12, color='#002561'),
            title_font=dict(size=16, color='#002561', family="Helvetica Neue, Arial"),
            plot_bgcolor='#FAFAFA',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)

def display_stage_analysis(df):
    """Display pipeline stage analysis"""
    st.markdown('<div class="section-header">🔨 Stage Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Jobs by stage
        stage_counts = df['JOB_STAGE'].value_counts().reset_index()
        stage_counts.columns = ['Stage', 'Count']
        
        ericsson_colors = ['#0058A3', '#0082C8', '#002561', '#5B9BD5', '#2E75B6', '#4F81BD', '#0070C0']
        
        fig = px.bar(
            stage_counts,
            x='Stage',
            y='Count',
            title='Jobs by Pipeline Stage',
            color='Stage',
            color_discrete_sequence=ericsson_colors
        )
        fig.update_layout(
            showlegend=False,
            font=dict(family="Helvetica Neue, Arial", size=12, color='#002561'),
            title_font=dict(size=16, color='#002561', family="Helvetica Neue, Arial"),
            plot_bgcolor='#FAFAFA',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Success rate by stage
        stage_success = df.groupby('JOB_STAGE').agg({
            'IS_SUCCESS': 'mean',
            'JOB_ID': 'count'
        }).reset_index()
        stage_success.columns = ['Stage', 'Success Rate', 'Total Jobs']
        stage_success['Success Rate'] = stage_success['Success Rate'] * 100
        stage_success = stage_success.sort_values('Success Rate', ascending=True)
        
        fig = px.bar(
            stage_success,
            x='Success Rate',
            y='Stage',
            orientation='h',
            title='Success Rate by Stage (%)',
            color='Success Rate',
            color_continuous_scale=[[0, '#EF5350'], [0.5, '#FFA726'], [1, '#66BB6A']],
            text='Success Rate'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            showlegend=False,
            yaxis={'categoryorder':'total ascending'},
            font=dict(family="Helvetica Neue, Arial", size=12, color='#002561'),
            title_font=dict(size=16, color='#002561', family="Helvetica Neue, Arial"),
            plot_bgcolor='#FAFAFA',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)

def display_duration_analysis(df):
    """Display job duration analysis"""
    st.markdown('<div class="section-header">⏱️ Duration Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Average duration by stage
        stage_duration = df.groupby('JOB_STAGE')['DURATION_MINUTES'].mean().reset_index()
        stage_duration.columns = ['Stage', 'Avg Duration (min)']
        stage_duration = stage_duration.sort_values('Avg Duration (min)', ascending=False)
        
        fig = px.bar(
            stage_duration,
            x='Avg Duration (min)',
            y='Stage',
            orientation='h',
            title='Average Job Duration by Stage',
            color='Avg Duration (min)',
            color_continuous_scale=[[0, '#E3F2FD'], [0.5, '#0082C8'], [1, '#0058A3']],
            text='Avg Duration (min)'
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(
            showlegend=False,
            yaxis={'categoryorder':'total ascending'},
            font=dict(family="Helvetica Neue, Arial", size=12, color='#002561'),
            title_font=dict(size=16, color='#002561', family="Helvetica Neue, Arial"),
            plot_bgcolor='#FAFAFA',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Duration distribution box plot
        fig = px.box(
            df[df['DURATION_MINUTES'] < df['DURATION_MINUTES'].quantile(0.95)],
            y='DURATION_MINUTES',
            x='JOB_STAGE',
            title='Job Duration Distribution by Stage (95th percentile)',
            color='JOB_STAGE',
            color_discrete_sequence=['#0058A3', '#0082C8', '#002561', '#5B9BD5', '#2E75B6', '#4F81BD', '#0070C0']
        )
        fig.update_layout(
            showlegend=False,
            font=dict(family="Helvetica Neue, Arial", size=12, color='#002561'),
            title_font=dict(size=16, color='#002561', family="Helvetica Neue, Arial"),
            plot_bgcolor='#FAFAFA',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)

def display_timeline_trends(df):
    """Display timeline and trends"""
    st.markdown('<div class="section-header">📈 Timeline & Trends</div>', unsafe_allow_html=True)
    
    # Daily pipeline activity
    df_sorted = df.sort_values('JOB_CREATEDAT')
    df_sorted['DATE'] = df_sorted['JOB_CREATEDAT'].dt.date
    
    daily_stats = df_sorted.groupby('DATE').agg({
        'JOB_ID': 'count',
        'IS_SUCCESS': 'sum',
        'IS_FAILED': 'sum'
    }).reset_index()
    daily_stats.columns = ['Date', 'Total', 'Success', 'Failed']
    daily_stats['Success Rate'] = (daily_stats['Success'] / daily_stats['Total'] * 100).round(1)
    
    # Take last 30 days for clarity
    daily_stats = daily_stats.tail(30)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_stats['Date'],
        y=daily_stats['Success'],
        mode='lines+markers',
        name='Successful',
        line=dict(color='#66BB6A', width=2),
        marker=dict(size=6)
    ))
    fig.add_trace(go.Scatter(
        x=daily_stats['Date'],
        y=daily_stats['Failed'],
        mode='lines+markers',
        name='Failed',
        line=dict(color='#EF5350', width=2),
        marker=dict(size=6)
    ))
    fig.add_trace(go.Scatter(
        x=daily_stats['Date'],
        y=daily_stats['Total'],
        mode='lines',
        name='Total',
        line=dict(color='#0058A3', width=3, dash='dash')
    ))
    
    fig.update_layout(
        title='Daily Pipeline Activity (Last 30 Days)',
        xaxis_title='Date',
        yaxis_title='Number of Pipelines',
        hovermode='x unified',
        height=400,
        font=dict(family="Helvetica Neue, Arial", size=12, color='#002561'),
        title_font=dict(size=16, color='#002561', family="Helvetica Neue, Arial"),
        plot_bgcolor='#FAFAFA',
        paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Success rate trend
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_stats['Date'],
            y=daily_stats['Success Rate'],
            mode='lines+markers',
            name='Success Rate',
            line=dict(color='#0058A3', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(0, 88, 163, 0.1)'
        ))
        fig.update_layout(
            title='Daily Success Rate Trend (%)',
            xaxis_title='Date',
            yaxis_title='Success Rate (%)',
            height=350,
            font=dict(family="Helvetica Neue, Arial", size=12, color='#002561'),
            title_font=dict(size=16, color='#002561', family="Helvetica Neue, Arial"),
            plot_bgcolor='#FAFAFA',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average duration trend
        duration_trend = df_sorted.groupby('DATE')['DURATION_MINUTES'].mean().reset_index()
        duration_trend.columns = ['Date', 'Avg Duration']
        duration_trend = duration_trend.tail(30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=duration_trend['Date'],
            y=duration_trend['Avg Duration'],
            mode='lines+markers',
            name='Avg Duration',
            line=dict(color='#0082C8', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(0, 130, 200, 0.1)'
        ))
        fig.update_layout(
            title='Average Duration Trend (minutes)',
            xaxis_title='Date',
            yaxis_title='Duration (minutes)',
            height=350,
            font=dict(family="Helvetica Neue, Arial", size=12, color='#002561'),
            title_font=dict(size=16, color='#002561', family="Helvetica Neue, Arial"),
            plot_bgcolor='#FAFAFA',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)

def display_project_breakdown(df):
    """Display project-wise breakdown"""
    st.markdown('<div class="section-header">📦 Project Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Top projects by job count
        project_counts = df['PROJECT_NAME'].value_counts().head(15).reset_index()
        project_counts.columns = ['Project', 'Jobs']
        
        # Add success rate
        project_success = df.groupby('PROJECT_NAME')['IS_SUCCESS'].mean().reset_index()
        project_success.columns = ['Project', 'Success Rate']
        project_success['Success Rate'] = project_success['Success Rate'] * 100
        
        project_stats = project_counts.merge(project_success, on='Project')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=project_stats['Project'],
            x=project_stats['Jobs'],
            orientation='h',
            name='Total Jobs',
            marker=dict(
                color=project_stats['Success Rate'],
                colorscale=[[0, '#EF5350'], [0.5, '#FFA726'], [1, '#66BB6A']],
                colorbar=dict(title="Success<br>Rate (%)")
            ),
            text=project_stats['Jobs'],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Jobs: %{x}<br>Success: %{marker.color:.1f}%<extra></extra>'
        ))
        fig.update_layout(
            title='Top 15 Projects by Job Count (colored by success rate)',
            xaxis_title='Number of Jobs',
            yaxis_title='Project',
            yaxis={'categoryorder':'total ascending'},
            height=500,
            font=dict(family="Helvetica Neue, Arial", size=12, color='#002561'),
            title_font=dict(size=16, color='#002561', family="Helvetica Neue, Arial"),
            plot_bgcolor='#FAFAFA',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Project Statistics")
        st.metric("Total Projects", df['PROJECT_NAME'].nunique())
        st.metric("Avg Jobs/Project", f"{len(df)/df['PROJECT_NAME'].nunique():.0f}")
        
        # Most active project
        most_active = df['PROJECT_NAME'].value_counts().index[0]
        most_active_count = df['PROJECT_NAME'].value_counts().iloc[0]
        st.metric("Most Active", most_active, delta=f"{most_active_count} jobs")
        
        # Best performing project (with min 50 jobs)
        project_perf = df.groupby('PROJECT_NAME').agg({
            'IS_SUCCESS': 'mean',
            'JOB_ID': 'count'
        }).reset_index()
        project_perf = project_perf[project_perf['JOB_ID'] >= 50]
        if len(project_perf) > 0:
            best_project = project_perf.loc[project_perf['IS_SUCCESS'].idxmax()]
            st.metric(
                "Best Success Rate",
                best_project['PROJECT_NAME'],
                delta=f"{best_project['IS_SUCCESS']*100:.1f}%"
            )

def display_recent_activity(df):
    """Display recent pipeline activity"""
    st.markdown('<div class="section-header">🔄 Recent Pipeline Activity (Last 7 Days)</div>', unsafe_allow_html=True)
    
    recent_df = df[df['IS_RECENT']].sort_values('JOB_CREATEDAT', ascending=False)
    
    if len(recent_df) == 0:
        st.info("No pipeline activity in the last 7 days.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔄 Recent Jobs", len(recent_df))
    with col2:
        recent_success_rate = (recent_df['IS_SUCCESS'].sum() / len(recent_df) * 100)
        st.metric("✅ Success Rate", f"{recent_success_rate:.1f}%")
    with col3:
        st.metric("❌ Failed", len(recent_df[recent_df['IS_FAILED']]))
    with col4:
        st.metric("⏱️ Avg Duration", f"{recent_df['DURATION_MINUTES'].mean():.1f} min")
    
    # Recent jobs table
    display_cols = ['PROJECT_NAME', 'JOB_NAME', 'PIPELINE_STATUS', 'JOB_STAGE', 
                   'DURATION_MINUTES', 'JOB_CREATEDAT']
    recent_display = recent_df[display_cols].head(100).copy()
    
    # Format for display
    recent_display['JOB_CREATEDAT'] = recent_display['JOB_CREATEDAT'].dt.strftime('%Y-%m-%d %H:%M')
    recent_display['DURATION_MINUTES'] = recent_display['DURATION_MINUTES'].round(1)
    recent_display['PIPELINE_STATUS'] = recent_display['PIPELINE_STATUS'].apply(
        lambda x: f"{STATUS_ICONS.get(x, '📌')} {x}"
    )
    recent_display['JOB_STAGE'] = recent_display['JOB_STAGE'].apply(
        lambda x: f"{STAGE_ICONS.get(x, '📌')} {x}"
    )
    
    st.dataframe(
        recent_display,
        use_container_width=True,
        height=400,
        hide_index=True,
        column_config={
            "PROJECT_NAME": "Project",
            "JOB_NAME": "Job",
            "PIPELINE_STATUS": "Status",
            "JOB_STAGE": "Stage",
            "DURATION_MINUTES": st.column_config.NumberColumn("Duration (min)", format="%.1f"),
            "JOB_CREATEDAT": "Created"
        }
    )

def display_failed_jobs(df):
    """Display failed jobs analysis"""
    st.markdown('<div class="section-header">❌ Failed Jobs Analysis</div>', unsafe_allow_html=True)
    
    failed_df = df[df['IS_FAILED']].copy()
    
    if len(failed_df) == 0:
        st.success("🎉 No failed jobs! All pipelines are running smoothly.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Failed jobs by stage
        failed_by_stage = failed_df['JOB_STAGE'].value_counts().reset_index()
        failed_by_stage.columns = ['Stage', 'Failed Jobs']
        
        fig = px.bar(
            failed_by_stage,
            x='Failed Jobs',
            y='Stage',
            orientation='h',
            title='Failed Jobs by Stage',
            color='Failed Jobs',
            color_continuous_scale=[[0, '#FFCDD2'], [1, '#EF5350']]
        )
        fig.update_layout(
            showlegend=False,
            yaxis={'categoryorder':'total ascending'},
            font=dict(family="Helvetica Neue, Arial", size=12, color='#002561'),
            title_font=dict(size=16, color='#002561', family="Helvetica Neue, Arial"),
            plot_bgcolor='#FAFAFA',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top projects with failures
        failed_projects = failed_df['PROJECT_NAME'].value_counts().head(10).reset_index()
        failed_projects.columns = ['Project', 'Failures']
        
        fig = px.bar(
            failed_projects,
            x='Failures',
            y='Project',
            orientation='h',
            title='Top 10 Projects with Failures',
            color='Failures',
            color_continuous_scale=[[0, '#FFCDD2'], [1, '#EF5350']]
        )
        fig.update_layout(
            showlegend=False,
            yaxis={'categoryorder':'total ascending'},
            font=dict(family="Helvetica Neue, Arial", size=12, color='#002561'),
            title_font=dict(size=16, color='#002561', family="Helvetica Neue, Arial"),
            plot_bgcolor='#FAFAFA',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent failures
    with st.expander("🔍 Recent Failed Jobs (Last 50)", expanded=False):
        recent_failures = failed_df.sort_values('JOB_CREATEDAT', ascending=False).head(50)
        display_cols = ['PROJECT_NAME', 'JOB_NAME', 'JOB_STAGE', 'DURATION_MINUTES', 'JOB_CREATEDAT']
        failures_display = recent_failures[display_cols].copy()
        failures_display['JOB_CREATEDAT'] = failures_display['JOB_CREATEDAT'].dt.strftime('%Y-%m-%d %H:%M')
        failures_display['DURATION_MINUTES'] = failures_display['DURATION_MINUTES'].round(1)
        
        st.dataframe(
            failures_display,
            use_container_width=True,
            height=300,
            hide_index=True
        )

def main():
    """Main application"""
    
    # Ericsson Branded Header
    st.markdown("""
    <div style="background-color: #0058A3; padding: 1.5rem 2rem; margin: -1rem -1rem 2rem -1rem; border-bottom: 3px solid #002561;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1 style="color: white; margin: 0; font-weight: 300; font-size: 2.5rem; letter-spacing: 2px;">ERICSSON</h1>
                <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.95; font-weight: 300;">CI/CD Pipeline Development Tracker</p>
            </div>
            <div style="text-align: right;">
                <p style="color: white; margin: 0; font-size: 0.9rem; opacity: 0.9;">Empowering DevOps Excellence</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading pipeline data...'):
        df = load_data()
    
    # Sidebar branding and filters
    st.sidebar.markdown("""
    <div style="background-color: #0058A3; padding: 1rem; margin: -1rem -1rem 1rem -1rem; text-align: center; border-bottom: 2px solid #002561;">
        <h2 style="color: white; margin: 0; font-weight: 300; letter-spacing: 1px; font-size: 1.5rem;">ERICSSON</h2>
        <p style="color: white; margin: 0.25rem 0 0 0; font-size: 0.8rem; opacity: 0.9;">CI/CD Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.header("🔍 Filters")
    
    # Project filter
    all_projects = ['All'] + sorted(df['PROJECT_NAME'].unique().tolist())
    selected_project = st.sidebar.selectbox('Project', all_projects)
    
    # Pipeline status filter
    all_statuses = ['All'] + sorted(df['PIPELINE_STATUS'].unique().tolist())
    selected_status = st.sidebar.selectbox('Pipeline Status', all_statuses)
    
    # Stage filter
    all_stages = ['All'] + sorted(df['JOB_STAGE'].unique().tolist())
    selected_stage = st.sidebar.selectbox('Job Stage', all_stages)
    
    # Date range filter
    st.sidebar.markdown("---")
    date_range = st.sidebar.date_input(
        "Job Created Date Range",
        value=(df['JOB_CREATEDAT'].min().date(), df['JOB_CREATEDAT'].max().date()),
        min_value=df['JOB_CREATEDAT'].min().date(),
        max_value=df['JOB_CREATEDAT'].max().date()
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_project != 'All':
        filtered_df = filtered_df[filtered_df['PROJECT_NAME'] == selected_project]
    
    if selected_status != 'All':
        filtered_df = filtered_df[filtered_df['PIPELINE_STATUS'] == selected_status]
    
    if selected_stage != 'All':
        filtered_df = filtered_df[filtered_df['JOB_STAGE'] == selected_stage]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['JOB_CREATEDAT'].dt.date >= start_date) & 
            (filtered_df['JOB_CREATEDAT'].dt.date <= end_date)
        ]
    
    # Display information
    st.sidebar.markdown("---")
    st.sidebar.info(f"📊 Showing {len(filtered_df):,} of {len(df):,} jobs")
    
    # Main content
    if len(filtered_df) == 0:
        st.warning("No pipeline jobs match the selected filters.")
        return
    
    # Display sections
    display_metrics(filtered_df)
    
    st.markdown("---")
    
    display_pipeline_overview(filtered_df)
    
    st.markdown("---")
    
    display_stage_analysis(filtered_df)
    
    st.markdown("---")
    
    display_duration_analysis(filtered_df)
    
    st.markdown("---")
    
    display_timeline_trends(filtered_df)
    
    st.markdown("---")
    
    display_project_breakdown(filtered_df)
    
    st.markdown("---")
    
    display_recent_activity(filtered_df)
    
    st.markdown("---")
    
    display_failed_jobs(filtered_df)
    
    # Ericsson Footer
    st.markdown("---")
    st.markdown(f"""
    <div class="ericsson-footer">
        <div style="border-top: 3px solid #0058A3; padding-top: 1.5rem; margin-top: 1rem;">
            <p style="margin: 0; font-size: 1rem; color: #002561; font-weight: 500;">
                <strong>ERICSSON</strong> | CI/CD Analytics Platform
            </p>
            <p style="margin: 0.5rem 0; color: #707070; font-size: 0.9rem;">
                Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
                Total jobs: {len(df):,}
            </p>
            <p style="margin: 0.5rem 0 0 0; color: #707070; font-size: 0.85rem;">
                © {datetime.now().year} Ericsson. Empowering an intelligent, sustainable, and connected world.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

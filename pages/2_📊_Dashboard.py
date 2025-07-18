import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from datetime import datetime, timedelta

# st.set_page_config(
#     page_title="Analytics Dashboard",
#     page_icon="📊",
#     layout="wide"
# )

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-left: 4px solid #007bff;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("📊 Analytics Dashboard")
    
    # Check prerequisites
    if not st.session_state.get('db_initialized', False):
        st.error("🚨 Database not initialized. Please restart the application.")
        return
    
    # Get data
    try:
        df = st.session_state.db.get_analysis_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    if df.empty:
        show_empty_state()
        return
    
    # Main dashboard
    show_filters(df)
    
    # Get filtered data
    filtered_df = apply_filters(df)
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        return
    
    # Dashboard sections
    show_key_metrics(filtered_df)
    show_charts_grid(filtered_df)
    show_detailed_analysis(filtered_df)
    show_data_table(filtered_df)

def show_empty_state():
    """Show empty state when no data is available"""
    
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>📊 No Data Available</h2>
        <p>Upload and analyze some tickets to see your dashboard!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📁 Upload Data Now", type="primary", use_container_width=True):
            st.switch_page("pages/1_📁_Upload_Data.py")

def show_filters(df):
    """Show filter controls"""
    
    st.markdown("### 🎛️ Filters & Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Month filter
        months = ['All'] + sorted(df['month'].unique().tolist(), reverse=True)
        selected_month = st.selectbox("📅 Select Month:", months)
        st.session_state.filter_month = selected_month
    
    with col2:
        # Category filter
        categories = ['All'] + sorted(df['ticket_category'].unique().tolist())
        selected_category = st.selectbox("🏷️ Category:", categories)
        st.session_state.filter_category = selected_category
    
    with col3:
        # Priority filter
        priority_range = st.slider(
            "⭐ Priority Score Range:",
            min_value=1,
            max_value=10,
            value=(1, 10)
        )
        st.session_state.filter_priority = priority_range
    
    with col4:
        # Date range
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            min_date = df['created_at'].min().date()
            max_date = df['created_at'].max().date()
            
            date_range = st.date_input(
                "📆 Date Range:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            st.session_state.filter_date_range = date_range

def apply_filters(df):
    """Apply selected filters to the data"""
    
    filtered_df = df.copy()
    
    # Month filter
    if st.session_state.get('filter_month') and st.session_state.filter_month != 'All':
        filtered_df = filtered_df[filtered_df['month'] == st.session_state.filter_month]
    
    # Category filter
    if st.session_state.get('filter_category') and st.session_state.filter_category != 'All':
        filtered_df = filtered_df[filtered_df['ticket_category'] == st.session_state.filter_category]
    
    # Priority filter
    if st.session_state.get('filter_priority'):
        min_priority, max_priority = st.session_state.filter_priority
        filtered_df = filtered_df[
            (filtered_df['priority_score'] >= min_priority) & 
            (filtered_df['priority_score'] <= max_priority)
        ]
    
    # Date range filter
    if (st.session_state.get('filter_date_range') and 
        len(st.session_state.filter_date_range) == 2 and
        'created_at' in filtered_df.columns):
        
        start_date, end_date = st.session_state.filter_date_range
        filtered_df['created_at'] = pd.to_datetime(filtered_df['created_at'])
        filtered_df = filtered_df[
            (filtered_df['created_at'].dt.date >= start_date) & 
            (filtered_df['created_at'].dt.date <= end_date)
        ]
    
    return filtered_df

def show_key_metrics(df):
    """Show key performance metrics"""
    
    st.markdown("### 📈 Key Performance Metrics")
    
    # Calculate metrics
    total_tickets = len(df)
    avg_resolution_time = df['resolution_time_hours'].mean()
    avg_satisfaction = df['customer_satisfaction'].mean()
    high_priority_count = len(df[df['priority_score'] >= 7])
    avg_priority = df['priority_score'].mean()
    
    # Calculate trends (if multiple months)
    months = df['month'].nunique()
    trend_indicators = calculate_trends(df) if months > 1 else {}
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        delta = trend_indicators.get('tickets_trend', None)
        st.metric(
            "Total Tickets", 
            f"{total_tickets:,}",
            delta=delta
        )
    
    with col2:
        delta = trend_indicators.get('resolution_trend', None)
        st.metric(
            "Avg Resolution Time", 
            f"{avg_resolution_time:.1f}h",
            delta=f"{delta:.1f}h" if delta else None
        )
    
    with col3:
        delta = trend_indicators.get('satisfaction_trend', None)
        st.metric(
            "Avg Satisfaction", 
            f"{avg_satisfaction:.1f}/5 ⭐",
            delta=f"{delta:.1f}" if delta else None
        )
    
    with col4:
        delta = trend_indicators.get('priority_trend', None)
        st.metric(
            "High Priority Issues", 
            f"{high_priority_count}",
            delta=delta
        )
    
    with col5:
        st.metric(
            "Avg Priority Score", 
            f"{avg_priority:.1f}/10"
        )

def calculate_trends(df):
    """Calculate month-over-month trends"""
    try:
        monthly_stats = df.groupby('month').agg({
            'ticket_id': 'count',
            'resolution_time_hours': 'mean',
            'customer_satisfaction': 'mean',
            'priority_score': lambda x: (x >= 7).sum()
        }).reset_index()
        
        if len(monthly_stats) >= 2:
            latest = monthly_stats.iloc[-1]
            previous = monthly_stats.iloc[-2]
            
            return {
                'tickets_trend': int(latest['ticket_id'] - previous['ticket_id']),
                'resolution_trend': latest['resolution_time_hours'] - previous['resolution_time_hours'],
                'satisfaction_trend': latest['customer_satisfaction'] - previous['customer_satisfaction'],
                'priority_trend': int(latest['priority_score'] - previous['priority_score'])
            }
    except:
        pass
    
    return {}

def show_charts_grid(df):
    """Show main charts in a grid layout"""
    
    st.markdown("### 📊 Visual Analytics")
    
    # Row 1: Category and Priority charts
    col1, col2 = st.columns(2)
    
    with col1:
        show_category_chart(df)
    
    with col2:
        show_priority_distribution(df)
    
    # Row 2: Time-based and satisfaction charts
    col1, col2 = st.columns(2)
    
    with col1:
        show_resolution_time_chart(df)
    
    with col2:
        show_satisfaction_analysis(df)
    
    # Row 3: Full-width charts
    show_monthly_trends(df)
    show_correlation_analysis(df)

def show_category_chart(df):
    """Show ticket category distribution"""
    
    category_counts = df['ticket_category'].value_counts()
    
    fig = px.bar(
        x=category_counts.values,
        y=category_counts.index,
        orientation='h',
        title="📋 Ticket Categories",
        labels={'x': 'Number of Tickets', 'y': 'Category'},
        color=category_counts.values,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_priority_distribution(df):
    """Show priority score distribution"""
    
    priority_counts = df['priority_score'].value_counts().sort_index()
    
    fig = px.pie(
        values=priority_counts.values,
        names=[f"Priority {i}" for i in priority_counts.index],
        title="⭐ Priority Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)

def show_resolution_time_chart(df):
    """Show resolution time analysis"""
    
    fig = px.histogram(
        df,
        x='resolution_time_hours',
        nbins=20,
        title="⏱️ Resolution Time Distribution",
        labels={'resolution_time_hours': 'Resolution Time (Hours)', 'count': 'Number of Tickets'},
        color_discrete_sequence=['#636EFA']
    )
    
    # Add average line
    avg_time = df['resolution_time_hours'].mean()
    fig.add_vline(
        x=avg_time,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Avg: {avg_time:.1f}h"
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_satisfaction_analysis(df):
    """Show customer satisfaction analysis"""
    
    satisfaction_counts = df['customer_satisfaction'].value_counts().sort_index()
    
    fig = px.bar(
        x=[f"{i} Star{'s' if i != 1 else ''}" for i in satisfaction_counts.index],
        y=satisfaction_counts.values,
        title="😊 Customer Satisfaction",
        labels={'x': 'Rating', 'y': 'Number of Tickets'},
        color=satisfaction_counts.values,
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def show_monthly_trends(df):
    """Show monthly trends if multiple months available"""
    
    if df['month'].nunique() > 1:
        monthly_data = df.groupby('month').agg({
            'ticket_id': 'count',
            'priority_score': 'mean',
            'customer_satisfaction': 'mean',
            'resolution_time_hours': 'mean'
        }).reset_index()
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Ticket Volume', 'Average Priority Score', 
                          'Customer Satisfaction', 'Resolution Time'),
            vertical_spacing=0.1
        )
        
        # Ticket volume
        fig.add_trace(
            go.Scatter(
                x=monthly_data['month'],
                y=monthly_data['ticket_id'],
                mode='lines+markers',
                name='Ticket Count',
                line=dict(color='#636EFA', width=3)
            ),
            row=1, col=1
        )
        
        # Priority score
        fig.add_trace(
            go.Scatter(
                x=monthly_data['month'],
                y=monthly_data['priority_score'],
                mode='lines+markers',
                name='Avg Priority',
                line=dict(color='#EF553B', width=3)
            ),
            row=1, col=2
        )
        
        # Satisfaction
        fig.add_trace(
            go.Scatter(
                x=monthly_data['month'],
                y=monthly_data['customer_satisfaction'],
                mode='lines+markers',
                name='Avg Satisfaction',
                line=dict(color='#00CC96', width=3)
            ),
            row=2, col=1
        )
        
        # Resolution time
        fig.add_trace(
            go.Scatter(
                x=monthly_data['month'],
                y=monthly_data['resolution_time_hours'],
                mode='lines+markers',
                name='Avg Resolution Time',
                line=dict(color='#AB63FA', width=3)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="📈 Monthly Trends Analysis",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_correlation_analysis(df):
    """Show correlation between different metrics"""
    
    st.markdown("### 🔗 Correlation Analysis")
    
    # Create scatter plot matrix
    numeric_cols = ['priority_score', 'resolution_time_hours', 'customer_satisfaction']
    
    if all(col in df.columns for col in numeric_cols):
        fig = px.scatter_matrix(
            df,
            dimensions=numeric_cols,
            color='ticket_category',
            title="🔍 Metrics Correlation Matrix",
            height=600
        )
        
        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, use_container_width=True)

def show_detailed_analysis(df):
    """Show detailed analysis insights"""
    
    st.markdown("### 🔍 Detailed Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Top Issues by Priority")
        high_priority = df[df['priority_score'] >= 7].nlargest(10, 'priority_score')
        
        if not high_priority.empty:
            for _, row in high_priority.iterrows():
                with st.expander(f"🔥 Priority {row['priority_score']} - {row['ticket_category']}"):
                    st.write(f"**Ticket ID:** {row['ticket_id']}")
                    st.write(f"**Month:** {row['month']}")
                    st.write(f"**Resolution Time:** {row['resolution_time_hours']:.1f} hours")
                    st.write(f"**Customer Satisfaction:** {row['customer_satisfaction']}/5 ⭐")
                    
                    # Show SDK issues if available
                    if pd.notna(row['sdk_issues']):
                        try:
                            sdk_issues = json.loads(row['sdk_issues']) if isinstance(row['sdk_issues'], str) else row['sdk_issues']
                            if sdk_issues:
                                st.write("**SDK Issues:**")
                                for issue in sdk_issues[:3]:  # Show top 3 issues
                                    st.write(f"• {issue}")
                        except:
                            pass
        else:
            st.info("No high-priority tickets found with current filters.")
    
    with col2:
        st.markdown("#### 📊 Performance Summary")
        
        # Category performance
        cat_performance = df.groupby('ticket_category').agg({
            'customer_satisfaction': 'mean',
            'resolution_time_hours': 'mean',
            'priority_score': 'mean',
            'ticket_id': 'count'
        }).round(2)
        
        cat_performance.columns = ['Avg Satisfaction', 'Avg Resolution (h)', 'Avg Priority', 'Count']
        cat_performance = cat_performance.sort_values('Avg Satisfaction', ascending=False)
        
        st.dataframe(cat_performance, use_container_width=True)

def show_data_table(df):
    """Show detailed data table"""
    
    st.markdown("### 📋 Detailed Data Table")
    
    # Prepare display columns
    display_cols = [
        'ticket_id', 'month', 'ticket_category', 'priority_score',
        'resolution_time_hours', 'customer_satisfaction'
    ]
    
    # Filter available columns
    available_cols = [col for col in display_cols if col in df.columns]
    
    if available_cols:
        display_df = df[available_cols].copy()
        
        # Format numeric columns
        if 'resolution_time_hours' in display_df.columns:
            display_df['resolution_time_hours'] = display_df['resolution_time_hours'].round(2)
        
        if 'customer_satisfaction' in display_df.columns:
            display_df['customer_satisfaction'] = display_df['customer_satisfaction'].round(1)
        
        # Show data with pagination
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Export option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"ticket_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()

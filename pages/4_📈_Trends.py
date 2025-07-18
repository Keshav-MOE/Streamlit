# At the top of pages/4_📈_Trends.py, replace:
import scipy.stats as stats

# With:
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("⚠️ Scipy not available - some advanced forecasting features disabled")
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from datetime import datetime, timedelta
from collections import Counter

# REMOVE THIS LINE - it causes the error:
# st.set_page_config(...)

# Try to import scipy, make it optional
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Custom CSS
st.markdown("""
<style>
    .trend-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    .forecast-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .alert-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("📈 Historical Trends & Forecasting")
    
    # Check prerequisites
    if not st.session_state.get('db_initialized', False):
        st.error("🚨 Database not initialized. Please restart the application.")
        return
    
    # Get data
    try:
        df = st.session_state.db.get_analysis_data()
        monthly_summaries = st.session_state.db.get_monthly_summaries()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    if df.empty:
        show_empty_state()
        return
    
    # Check if we have multiple months for trend analysis
    if df['month'].nunique() < 2:
        show_single_month_view(df)
        return
    
    # Main trend analysis
    show_trend_overview(df, monthly_summaries)
    show_volume_trends(df)
    show_quality_trends(df)
    show_issue_evolution(df)
    show_forecasting(df)
    show_comparative_analysis(df)
    show_actionable_insights(df)

def show_empty_state():
    """Show empty state when no data is available"""
    
    st.markdown("""
    <div class="trend-card">
        <h2>📈 No Historical Data Available</h2>
        <p>Upload data for multiple months to see trend analysis!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📁 Upload Data Now", type="primary", use_container_width=True):
            st.switch_page("pages/1_📁_Upload_Data.py")

def show_single_month_view(df):
    """Show view for single month data"""
    
    st.info("📊 Only one month of data available. Upload more months to see trends!")
    
    # Show basic single-month analytics
    month_name = df['month'].iloc[0] if not df.empty else "Unknown"
    
    st.markdown(f"### 📅 {month_name} Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tickets", len(df))
    
    with col2:
        avg_priority = df['priority_score'].mean() if 'priority_score' in df.columns else 0
        st.metric("Avg Priority", f"{avg_priority:.1f}/10")
    
    with col3:
        avg_satisfaction = df['customer_satisfaction'].mean() if 'customer_satisfaction' in df.columns else 0
        st.metric("Avg Satisfaction", f"{avg_satisfaction:.1f}/5")
    
    with col4:
        avg_resolution = df['resolution_time_hours'].mean() if 'resolution_time_hours' in df.columns else 0
        st.metric("Avg Resolution", f"{avg_resolution:.1f}h")
    
    # Show basic charts
    show_single_month_charts(df)

def show_single_month_charts(df):
    """Show charts for single month analysis"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution
        if 'ticket_category' in df.columns:
            category_counts = df['ticket_category'].value_counts()
            if not category_counts.empty:
                fig = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="📊 Category Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No category data available")
        else:
            st.info("No category column found")
    
    with col2:
        # Priority vs Satisfaction
        if all(col in df.columns for col in ['priority_score', 'customer_satisfaction', 'ticket_category']):
            fig = px.scatter(
                df,
                x='priority_score',
                y='customer_satisfaction',
                title="⭐ Priority vs Satisfaction",
                color='ticket_category'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for scatter plot")

def show_trend_overview(df, monthly_summaries):
    """Show high-level trend overview"""
    
    st.markdown("### 🎯 Trend Overview")
    
    # Calculate overall trends
    monthly_stats = calculate_monthly_statistics(df)
    
    if monthly_stats.empty:
        st.warning("No monthly statistics available")
        return
    
    # Display trend metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate month-over-month changes
    latest_month = monthly_stats.iloc[-1]
    previous_month = monthly_stats.iloc[-2] if len(monthly_stats) > 1 else latest_month
    
    ticket_change = latest_month['ticket_count'] - previous_month['ticket_count']
    priority_change = latest_month['avg_priority'] - previous_month['avg_priority']
    satisfaction_change = latest_month['avg_satisfaction'] - previous_month['avg_satisfaction']
    resolution_change = latest_month['avg_resolution_time'] - previous_month['avg_resolution_time']
    
    with col1:
        st.markdown(f"""
        <div class="trend-card">
            <h3>🎫 Ticket Volume</h3>
            <h2>{latest_month['ticket_count']}</h2>
            <p>{'📈 +' if ticket_change > 0 else '📉 '}{ticket_change} vs last month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        priority_icon = "📈" if priority_change > 0 else "📉"
        priority_color = "#ff4444" if priority_change > 0 else "#44ff44"
        st.markdown(f"""
        <div class="trend-card">
            <h3>⭐ Avg Priority</h3>
            <h2>{latest_month['avg_priority']:.1f}</h2>
            <p style="color: {priority_color};">{priority_icon} {priority_change:+.1f} vs last month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        satisfaction_icon = "📈" if satisfaction_change > 0 else "📉"
        satisfaction_color = "#44ff44" if satisfaction_change > 0 else "#ff4444"
        st.markdown(f"""
        <div class="trend-card">
            <h3>😊 Satisfaction</h3>
            <h2>{latest_month['avg_satisfaction']:.1f}/5</h2>
            <p style="color: {satisfaction_color};">{satisfaction_icon} {satisfaction_change:+.1f} vs last month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        resolution_icon = "📈" if resolution_change > 0 else "📉"
        resolution_color = "#ff4444" if resolution_change > 0 else "#44ff44"
        st.markdown(f"""
        <div class="trend-card">
            <h3>⏱️ Resolution Time</h3>
            <h2>{latest_month['avg_resolution_time']:.1f}h</h2>
            <p style="color: {resolution_color};">{resolution_icon} {resolution_change:+.1f}h vs last month</p>
        </div>
        """, unsafe_allow_html=True)

def calculate_monthly_statistics(df):
    """Calculate monthly statistics for trend analysis"""
    
    if df.empty:
        return pd.DataFrame()
    
    # Define required columns with defaults
    required_columns = {
        'ticket_id': 'ticket_id',
        'priority_score': 'priority_score', 
        'customer_satisfaction': 'customer_satisfaction',
        'resolution_time_hours': 'resolution_time_hours'
    }
    
    # Check which columns exist
    available_columns = {k: v for k, v in required_columns.items() if v in df.columns}
    
    if not available_columns:
        st.warning("Required columns not found for monthly statistics")
        return pd.DataFrame()
    
    try:
        agg_dict = {}
        
        # Add aggregations for available columns
        if 'ticket_id' in available_columns.values():
            agg_dict[df.columns[df.columns.isin(['ticket_id'])].tolist()[0]] = 'count'
        
        for col_key, col_name in available_columns.items():
            if col_name in df.columns and col_key != 'ticket_id':
                agg_dict[col_name] = 'mean'
        
        monthly_stats = df.groupby('month').agg(agg_dict).reset_index()
        
        # Rename columns to standard names
        new_column_names = {'month': 'month'}
        for old_col in monthly_stats.columns:
            if old_col == 'month':
                continue
            elif 'ticket_id' in old_col or old_col == 'ticket_id':
                new_column_names[old_col] = 'ticket_count'
            elif 'priority' in old_col:
                new_column_names[old_col] = 'avg_priority'
            elif 'satisfaction' in old_col:
                new_column_names[old_col] = 'avg_satisfaction'
            elif 'resolution' in old_col:
                new_column_names[old_col] = 'avg_resolution_time'
        
        monthly_stats = monthly_stats.rename(columns=new_column_names)
        
        # Fill missing columns with defaults
        default_values = {
            'ticket_count': 0,
            'avg_priority': 5.0,
            'avg_satisfaction': 3.0,
            'avg_resolution_time': 0.0
        }
        
        for col, default_val in default_values.items():
            if col not in monthly_stats.columns:
                monthly_stats[col] = default_val
        
        # Sort by month
        monthly_stats = monthly_stats.sort_values('month')
        
        return monthly_stats
        
    except Exception as e:
        st.error(f"Error calculating monthly statistics: {e}")
        return pd.DataFrame()

def show_volume_trends(df):
    """Show ticket volume trends"""
    
    st.markdown("### 📊 Volume Trends")
    
    monthly_stats = calculate_monthly_statistics(df)
    
    if monthly_stats.empty:
        st.warning("No volume trend data available")
        return
    
    # Simple volume chart
    fig = px.line(
        monthly_stats,
        x='month',
        y='ticket_count',
        title='📈 Monthly Ticket Volume',
        markers=True
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show insights
    if len(monthly_stats) >= 2:
        latest_volume = monthly_stats['ticket_count'].iloc[-1]
        previous_volume = monthly_stats['ticket_count'].iloc[-2]
        growth_rate = ((latest_volume - previous_volume) / previous_volume * 100) if previous_volume > 0 else 0
        
        if growth_rate > 10:
            st.markdown(f"""
            <div class="alert-box">
                <strong>⚠️ High Growth Alert:</strong> Ticket volume increased by {growth_rate:.1f}% last month. 
                Consider scaling support resources.
            </div>
            """, unsafe_allow_html=True)
        elif growth_rate < -10:
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ Volume Reduction:</strong> Ticket volume decreased by {abs(growth_rate):.1f}% last month. 
                Recent improvements are showing positive results.
            </div>
            """, unsafe_allow_html=True)

def show_quality_trends(df):
    """Show quality trends (satisfaction, resolution time)"""
    
    st.markdown("### 🎯 Quality Trends")
    
    monthly_stats = calculate_monthly_statistics(df)
    
    if monthly_stats.empty:
        st.warning("No quality trend data available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Satisfaction trend
        fig = px.line(
            monthly_stats,
            x='month',
            y='avg_satisfaction',
            title='😊 Customer Satisfaction Trend',
            markers=True
        )
        fig.add_hline(y=4.0, line_dash="dash", line_color="green", annotation_text="Target: 4.0")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Resolution time trend
        fig = px.line(
            monthly_stats,
            x='month',
            y='avg_resolution_time',
            title='⏱️ Resolution Time Trend',
            markers=True
        )
        fig.add_hline(y=24, line_dash="dash", line_color="orange", annotation_text="Target: 24h")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_issue_evolution(df):
    """Show how issues evolve over time"""
    
    st.markdown("### 🔄 Issue Evolution")
    
    if 'month' not in df.columns:
        st.warning("Month column not found for issue evolution")
        return
    
    # Get all months
    all_months = sorted(df['month'].unique())
    
    if len(all_months) < 2:
        st.info("Need at least 2 months for issue evolution analysis")
        return
    
    # Extract and track top issues over time
    try:
        all_sdk_issues = extract_sdk_issues_from_df(df)
        flat_issues = [item for sublist in all_sdk_issues for item in sublist if item]
        
        if not flat_issues:
            st.info("No SDK issues found for evolution analysis")
            return
        
        top_5_issues = [issue for issue, count in Counter(flat_issues).most_common(5)]
        
        # Track these issues over time
        issue_evolution = {}
        for month in all_months:
            month_df = df[df['month'] == month]
            month_issues = extract_sdk_issues_from_df(month_df)
            month_flat = [item for sublist in month_issues for item in sublist if item]
            month_counter = Counter(month_flat)
            
            issue_evolution[month] = {issue: month_counter.get(issue, 0) for issue in top_5_issues}
        
        # Create evolution chart
        fig = go.Figure()
        
        for issue in top_5_issues:
            months = list(issue_evolution.keys())
            counts = [issue_evolution[month][issue] for month in months]
            
            fig.add_trace(go.Scatter(
                x=months,
                y=counts,
                mode='lines+markers',
                name=issue[:30] + '...' if len(issue) > 30 else issue,
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title='📈 Top Issues Evolution Over Time',
            xaxis_title='Month',
            yaxis_title='Issue Frequency',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Could not analyze issue evolution: {e}")

def show_forecasting(df):
    """Show forecasting and predictions"""
    
    st.markdown("### 🔮 Forecasting & Predictions")
    
    monthly_stats = calculate_monthly_statistics(df)
    
    if len(monthly_stats) < 3:
        st.info("Need at least 3 months of data for reliable forecasting.")
        return
    
    if not SCIPY_AVAILABLE:
        st.warning("⚠️ Advanced forecasting requires scipy package. Showing basic trend analysis instead.")
        show_simple_trends(monthly_stats)
        return
    
    try:
        # Simple linear regression for forecasting
        months_numeric = range(len(monthly_stats))
        
        # Forecast ticket volume
        ticket_slope, ticket_intercept, r_value, p_value, std_err = stats.linregress(
            months_numeric, monthly_stats['ticket_count']
        )
        
        # Forecast satisfaction
        satisfaction_slope, satisfaction_intercept, _, _, _ = stats.linregress(
            months_numeric, monthly_stats['avg_satisfaction']
        )
        
        # Generate forecasts for next 3 months
        future_months = ['Next Month', 'Month +2', 'Month +3']
        next_months_numeric = range(len(monthly_stats), len(monthly_stats) + 3)
        
        forecast_tickets = [max(0, ticket_slope * x + ticket_intercept) for x in next_months_numeric]
        forecast_satisfaction = [max(1, min(5, satisfaction_slope * x + satisfaction_intercept)) for x in next_months_numeric]
        
        # Create forecast visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Ticket Volume Forecast', 'Satisfaction Forecast')
        )
        
        # Historical data
        fig.add_trace(
            go.Scatter(
                x=monthly_stats['month'],
                y=monthly_stats['ticket_count'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Forecast
        fig.add_trace(
            go.Scatter(
                x=future_months,
                y=forecast_tickets,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # Satisfaction historical
        fig.add_trace(
            go.Scatter(
                x=monthly_stats['month'],
                y=monthly_stats['avg_satisfaction'],
                mode='lines+markers',
                name='Historical Satisfaction',
                line=dict(color='green'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Satisfaction forecast
        fig.add_trace(
            go.Scatter(
                x=future_months,
                y=forecast_satisfaction,
                mode='lines+markers',
                name='Satisfaction Forecast',
                line=dict(color='orange', dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=500, title_text="📊 3-Month Forecasting")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show forecast summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="forecast-box">
                <h4>📈 Next Month Prediction</h4>
                <p><strong>Tickets:</strong> {forecast_tickets[0]:.0f}</p>
                <p><strong>Satisfaction:</strong> {forecast_satisfaction[0]:.1f}/5</p>
                <p><strong>Confidence:</strong> {abs(r_value)*100:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.warning(f"Forecasting error: {e}. Showing simple trends instead.")
        show_simple_trends(monthly_stats)

def show_simple_trends(monthly_stats):
    """Show simple trend analysis without scipy"""
    
    if monthly_stats.empty or len(monthly_stats) < 2:
        return
    
    # Calculate simple growth rates
    latest_tickets = monthly_stats['ticket_count'].iloc[-1]
    previous_tickets = monthly_stats['ticket_count'].iloc[-2] if len(monthly_stats) > 1 else latest_tickets
    
    growth_rate = ((latest_tickets - previous_tickets) / previous_tickets * 100) if previous_tickets > 0 else 0
    
    st.markdown(f"""
    <div class="forecast-box">
        <h4>📊 Simple Trend Analysis</h4>
        <p><strong>Current Volume:</strong> {latest_tickets:.0f} tickets</p>
        <p><strong>Growth Rate:</strong> {growth_rate:+.1f}%</p>
        <p><strong>Trend:</strong> {'📈 Increasing' if growth_rate > 0 else '📉 Decreasing'}</p>
    </div>
    """, unsafe_allow_html=True)

def show_comparative_analysis(df):
    """Show comparative analysis between different time periods"""
    
    st.markdown("### 🔍 Comparative Analysis")
    
    if 'month' not in df.columns:
        st.warning("Month column not found")
        return
    
    all_months = sorted(df['month'].unique())
    
    if len(all_months)  < 2:
        st.info("Need at least 2 months for comparative analysis.")
        return
    
    # Period selection
    col1, col2 = st.columns(2)
    
    with col1:
        period_1 = st.selectbox("Select Period 1:", all_months, index=0)
    
    with col2:
        period_2 = st.selectbox("Select Period 2:", all_months, index=len(all_months)-1)
    
    if period_1 == period_2:
        st.warning("Please select different periods for comparison.")
        return
    
    # Get data for both periods
    period_1_data = df[df['month'] == period_1]
    period_2_data = df[df['month'] == period_2]
    
    if period_1_data.empty or period_2_data.empty:
        st.warning("No data found for selected periods")
        return
    
    # Calculate comparison metrics
    comparison_data = {
        'Metric': [],
        period_1: [],
        period_2: [],
        'Change': [],
        'Change %': []
    }
    
    # Total tickets
    p1_tickets = len(period_1_data)
    p2_tickets = len(period_2_data)
    comparison_data['Metric'].append('Total Tickets')
    comparison_data[period_1].append(p1_tickets)
    comparison_data[period_2].append(p2_tickets)
    comparison_data['Change'].append(p2_tickets - p1_tickets)
    comparison_data['Change %'].append(((p2_tickets - p1_tickets) / p1_tickets * 100) if p1_tickets > 0 else 0)
    
    # Add other metrics if columns exist
    metrics_map = {
        'Avg Priority Score': 'priority_score',
        'Avg Satisfaction': 'customer_satisfaction', 
        'Avg Resolution Time': 'resolution_time_hours'
    }
    
    for metric_name, col_name in metrics_map.items():
        if col_name in df.columns:
            p1_val = period_1_data[col_name].mean()
            p2_val = period_2_data[col_name].mean()
            
            comparison_data['Metric'].append(metric_name)
            comparison_data[period_1].append(round(p1_val, 2))
            comparison_data[period_2].append(round(p2_val, 2))
            comparison_data['Change'].append(round(p2_val - p1_val, 2))
            comparison_data['Change %'].append(round(((p2_val - p1_val) / p1_val * 100) if p1_val > 0 else 0, 1))
    
    # Display comparison
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

def show_actionable_insights(df):
    """Show actionable insights based on trend analysis"""
    
    st.markdown("### 💡 Actionable Insights")
    
    if df.empty:
        st.info("No data available for insights")
        return
    
    monthly_stats = calculate_monthly_statistics(df)
    
    if monthly_stats.empty:
        st.info("Cannot generate insights without monthly statistics")
        return
    
    # Generate insights
    insights = []
    recommendations = []
    
    # Volume trend analysis
    if len(monthly_stats) >= 2:
        latest_volume = monthly_stats['ticket_count'].iloc[-1]
        previous_volume = monthly_stats['ticket_count'].iloc[-2]
        
        if latest_volume > previous_volume * 1.2:
            insights.append("📈 Ticket volume is increasing significantly")
            recommendations.append("Scale support team or implement automation")
        elif latest_volume < previous_volume * 0.8:
            insights.append("📉 Ticket volume is decreasing")
            recommendations.append("Document successful strategies for replication")
    
    # Satisfaction analysis
    if 'avg_satisfaction' in monthly_stats.columns:
        avg_satisfaction = monthly_stats['avg_satisfaction'].mean()
        if avg_satisfaction < 3:
            insights.append("😟 Customer satisfaction is below acceptable levels")
            recommendations.append("Implement customer feedback program immediately")
        elif avg_satisfaction > 4:
            insights.append("😊 Customer satisfaction is excellent")
            recommendations.append("Share best practices across all teams")
    
    # Display insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 Key Insights")
        if insights:
            for insight in insights:
                st.markdown(f"• {insight}")
        else:
            st.markdown("• No significant trends detected")
    
    with col2:
        st.markdown("#### 🎯 Recommendations")
        if recommendations:
            for rec in recommendations:
                st.markdown(f"• {rec}")
        else:
            st.markdown("• Continue monitoring performance")

def extract_sdk_issues_from_df(df):
    """Extract SDK issues from dataframe"""
    
    sdk_issues = []
    for _, row in df.iterrows():
        if pd.notna(row.get('sdk_issues')):
            try:
                issues = json.loads(row['sdk_issues']) if isinstance(row['sdk_issues'], str) else row['sdk_issues']
                if isinstance(issues, list):
                    sdk_issues.append(issues)
                else:
                    sdk_issues.append([])
            except (json.JSONDecodeError, TypeError):
                sdk_issues.append([])
        else:
            sdk_issues.append([])
    
    return sdk_issues

if __name__ == "__main__":
    main()

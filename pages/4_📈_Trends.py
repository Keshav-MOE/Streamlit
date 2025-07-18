# At the top of pages/4_📈_Trends.py, replace:
import scipy.stats as stats

# With:
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("⚠️ Scipy not available - some advanced forecasting features disabled")
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
import scipy.stats as stats

st.set_page_config(
    page_title="Trends Analysis",
    page_icon="📈",
    layout="wide"
)

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
    month_name = df['month'].iloc[0]
    
    st.markdown(f"### 📅 {month_name} Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tickets", len(df))
    
    with col2:
        avg_priority = df['priority_score'].mean()
        st.metric("Avg Priority", f"{avg_priority:.1f}/10")
    
    with col3:
        avg_satisfaction = df['customer_satisfaction'].mean()
        st.metric("Avg Satisfaction", f"{avg_satisfaction:.1f}/5")
    
    with col4:
        avg_resolution = df['resolution_time_hours'].mean()
        st.metric("Avg Resolution", f"{avg_resolution:.1f}h")
    
    # Show basic charts
    show_single_month_charts(df)

def show_single_month_charts(df):
    """Show charts for single month analysis"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution
        category_counts = df['ticket_category'].value_counts()
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="📊 Category Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Priority vs Satisfaction
        fig = px.scatter(
            df,
            x='priority_score',
            y='customer_satisfaction',
            title="⭐ Priority vs Satisfaction",
            color='ticket_category'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_trend_overview(df, monthly_summaries):
    """Show high-level trend overview"""
    
    st.markdown("### 🎯 Trend Overview")
    
    # Calculate overall trends
    monthly_stats = calculate_monthly_statistics(df)
    
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
    
    monthly_stats = df.groupby('month').agg({
        'ticket_id': 'count',
        'priority_score': 'mean',
        'customer_satisfaction': 'mean',
        'resolution_time_hours': 'mean'
    }).reset_index()
    
    monthly_stats.columns = ['month', 'ticket_count', 'avg_priority', 'avg_satisfaction', 'avg_resolution_time']
    
    # Sort by month (assuming format like "January_2024")
    monthly_stats = monthly_stats.sort_values('month')
    
    return monthly_stats

def show_volume_trends(df):
    """Show ticket volume trends"""
    
    st.markdown("### 📊 Volume Trends")
    
    monthly_stats = calculate_monthly_statistics(df)
    
    # Create comprehensive volume analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Monthly Ticket Volume', 'Volume by Category', 
                       'Weekly Patterns', 'Growth Rate'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Monthly volume
    fig.add_trace(
        go.Scatter(
            x=monthly_stats['month'],
            y=monthly_stats['ticket_count'],
            mode='lines+markers',
            name='Ticket Count',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Volume by category over time
    category_trends = df.groupby(['month', 'ticket_category']).size().reset_index(name='count')
    top_categories = df['ticket_category'].value_counts().head(5).index
    
    for category in top_categories:
        category_data = category_trends[category_trends['ticket_category'] == category]
        fig.add_trace(
            go.Scatter(
                x=category_data['month'],
                y=category_data['count'],
                mode='lines',
                name=category,
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Calculate growth rate
    monthly_stats['growth_rate'] = monthly_stats['ticket_count'].pct_change() * 100
    
    fig.add_trace(
        go.Bar(
            x=monthly_stats['month'],
            y=monthly_stats['growth_rate'],
            name='Growth Rate %',
            marker_color=['red' if x < 0 else 'green' for x in monthly_stats['growth_rate']],
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="📈 Comprehensive Volume Analysis")
    st.plotly_chart(fig, use_container_width=True)
    
    # Show insights
    latest_growth = monthly_stats['growth_rate'].iloc[-1]
    avg_growth = monthly_stats['growth_rate'].mean()
    
    if pd.notna(latest_growth):
        if latest_growth > 10:
            st.markdown(f"""
            <div class="alert-box">
                <strong>⚠️ High Growth Alert:</strong> Ticket volume increased by {latest_growth:.1f}% last month. 
                Consider scaling support resources.
            </div>
            """, unsafe_allow_html=True)
        elif latest_growth < -10:
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ Volume Reduction:</strong> Ticket volume decreased by {abs(latest_growth):.1f}% last month. 
                Recent improvements are showing positive results.
            </div>
            """, unsafe_allow_html=True)

def show_quality_trends(df):
    """Show quality trends (satisfaction, resolution time)"""
    
    st.markdown("### 🎯 Quality Trends")
    
    monthly_stats = calculate_monthly_statistics(df)
    
    # Create quality metrics chart
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Customer Satisfaction Trend', 'Resolution Time Trend'),
        specs=[[{"secondary_y": True}, {"secondary_y": True}]]
    )
    
    # Satisfaction trend
    fig.add_trace(
        go.Scatter(
            x=monthly_stats['month'],
            y=monthly_stats['avg_satisfaction'],
            mode='lines+markers',
            name='Satisfaction',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Add satisfaction target line
    fig.add_hline(
        y=4.0, 
        line_dash="dash", 
        line_color="green",
        annotation_text="Target: 4.0",
        row=1, col=1
    )
    
    # Resolution time trend
    fig.add_trace(
        go.Scatter(
            x=monthly_stats['month'],
            y=monthly_stats['avg_resolution_time'],
            mode='lines+markers',
            name='Resolution Time',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=8)
        ),
        row=1, col=2
    )
    
    # Add resolution time target
    target_resolution = 24  # 24 hours target
    fig.add_hline(
        y=target_resolution,
        line_dash="dash",
        line_color="orange", 
        annotation_text=f"Target: {target_resolution}h",
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Quality insights
    current_satisfaction = monthly_stats['avg_satisfaction'].iloc[-1]
    current_resolution = monthly_stats['avg_resolution_time'].iloc[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if current_satisfaction >= 4.0:
            st.markdown("""
            <div class="success-box">
                <strong>✅ Satisfaction Target Met:</strong> Current satisfaction score meets our 4.0+ target.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-box">
                <strong>⚠️ Satisfaction Below Target:</strong> Current score ({current_satisfaction:.1f}) is below 4.0 target.
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if current_resolution <= 24:
            st.markdown("""
            <div class="success-box">
                <strong>✅ Resolution Target Met:</strong> Resolution time meets 24-hour target.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-box">
                <strong>⚠️ Resolution Time Above Target:</strong> Current time ({current_resolution:.1f}h) exceeds 24h target.
            </div>
            """, unsafe_allow_html=True)

def show_issue_evolution(df):
    """Show how issues evolve over time"""
    
    st.markdown("### 🔄 Issue Evolution")
    
    # Extract and track top issues over time
    all_months = sorted(df['month'].unique())
    
    # Get top issues overall
    all_sdk_issues = extract_sdk_issues_from_df(df)
    flat_issues = [item for sublist in all_sdk_issues for item in sublist if item]
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
    
    # Issue lifecycle analysis
    st.markdown("#### 🔄 Issue Lifecycle Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # New issues (appeared in latest month)
        if len(all_months) >= 2:
            latest_month = all_months[-1]
            previous_month = all_months[-2]
            
            latest_issues = set(issue_evolution[latest_month].keys())
            previous_issues = set(issue_evolution[previous_month].keys())
            
            new_issues = latest_issues - previous_issues
            resolved_issues = previous_issues - latest_issues
            
            st.markdown(f"""
            **🆕 New Issues ({len(new_issues)}):**
            {chr(10).join([f"• {issue[:40]}..." if len(issue) > 40 else f"• {issue}" for issue in list(new_issues)[:5]])}
            """)
    
    with col2:
        if len(all_months) >= 2:
            st.markdown(f"""
            **✅ Resolved Issues ({len(resolved_issues)}):**
            {chr(10).join([f"• {issue[:40]}..." if len(issue) > 40 else f"• {issue}" for issue in list(resolved_issues)[:5]])}
            """)
    
    with col3:
        # Persistent issues
        persistent_issues = []
        for issue in top_5_issues:
            months_present = sum(1 for month in all_months if issue_evolution[month][issue] > 0)
            if months_present >= len(all_months) * 0.8:  # Present in 80% of months
                persistent_issues.append(issue)
        
        st.markdown(f"""
        **⚠️ Persistent Issues ({len(persistent_issues)}):**
        {chr(10).join([f"• {issue[:40]}..." if len(issue) > 40 else f"• {issue}" for issue in persistent_issues[:5]])}
        """)

def show_forecasting(df):
    """Show forecasting and predictions"""
    
    st.markdown("### 🔮 Forecasting & Predictions")
    
    monthly_stats = calculate_monthly_statistics(df)
    
    if len(monthly_stats) < 3:
        st.info("Need at least 3 months of data for reliable forecasting.")
        return
    
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
    
    forecast_tickets = [ticket_slope * x + ticket_intercept for x in next_months_numeric]
    forecast_satisfaction = [satisfaction_slope * x + satisfaction_intercept for x in next_months_numeric]
    
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
    
    with col2:
        trend_direction = "increasing" if ticket_slope > 0 else "decreasing"
        trend_color = "#ff4444" if ticket_slope > 0 else "#44ff44"
        
        st.markdown(f"""
        <div class="forecast-box">
            <h4>📊 Trend Analysis</h4>
            <p><strong>Volume:</strong> <span style="color: {trend_color};">{trend_direction}</span></p>
            <p><strong>Rate:</strong> {abs(ticket_slope):.1f} tickets/month</p>
            <p><strong>R²:</strong> {r_value**2:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if forecast_tickets[0] > monthly_stats['ticket_count'].iloc[-1] * 1.2:
            alert_type = "alert-box"
            alert_icon = "⚠️"
            alert_text = "High volume expected"
        else:
            alert_type = "success-box"
            alert_icon = "✅"
            alert_text = "Volume manageable"
        
        st.markdown(f"""
        <div class="{alert_type}">
            <h4>{alert_icon} Capacity Planning</h4>
            <p>{alert_text}</p>
            <p><strong>Recommended action:</strong> {'Scale team' if 'High' in alert_text else 'Maintain capacity'}</p>
        </div>
        """, unsafe_allow_html=True)

def show_comparative_analysis(df):
    """Show comparative analysis between different time periods"""
    
    st.markdown("### 🔍 Comparative Analysis")
    
    all_months = sorted(df['month'].unique())
    
    if len(all_months) < 2:
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
    
    # Calculate comparison metrics
    comparison = {
        'Metric': ['Total Tickets', 'Avg Priority Score', 'Avg Satisfaction', 'Avg Resolution Time'],
        period_1: [
            len(period_1_data),
            period_1_data['priority_score'].mean(),
            period_1_data['customer_satisfaction'].mean(),
            period_1_data['resolution_time_hours'].mean()
        ],
        period_2: [
            len(period_2_data),
            period_2_data['priority_score'].mean(),
            period_2_data['customer_satisfaction'].mean(),
            period_2_data['resolution_time_hours'].mean()
        ]
    }
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df['Change'] = comparison_df[period_2] - comparison_df[period_1]
    comparison_df['Change %'] = (comparison_df['Change'] / comparison_df[period_1] * 100).round(1)
    
    # Display comparison table
    st.markdown("#### 📊 Period Comparison")
    
    # Format the dataframe for display
    display_df = comparison_df.copy()
    display_df[period_1] = display_df[period_1].round(2)
    display_df[period_2] = display_df[period_2].round(2)
    display_df['Change'] = display_df['Change'].round(2)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Visual comparison
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=period_1,
        x=comparison_df['Metric'],
        y=comparison_df[period_1],
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name=period_2,
        x=comparison_df['Metric'],
        y=comparison_df[period_2],
        marker_color='darkblue'
    ))
    
    fig.update_layout(
        title=f'📊 {period_1} vs {period_2} Comparison',
        xaxis_title='Metrics',
        yaxis_title='Values',
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights from comparison
    st.markdown("#### 🎯 Key Changes")
    
    insights = []
    
    # Ticket volume change
    ticket_change = comparison_df[comparison_df['Metric'] == 'Total Tickets']['Change %'].iloc[0]
    if abs(ticket_change) > 10:
        direction = "increased" if ticket_change > 0 else "decreased" 
        insights.append(f"📊 Ticket volume {direction} by {abs(ticket_change):.1f}%")
    
    # Priority change
    priority_change = comparison_df[comparison_df['Metric'] == 'Avg Priority Score']['Change'].iloc[0]
    if abs(priority_change) > 0.5:
        direction = "increased" if priority_change > 0 else "decreased"
        insights.append(f"⭐ Average priority {direction} by {abs(priority_change):.1f} points")
    
    # Satisfaction change
    satisfaction_change = comparison_df[comparison_df['Metric'] == 'Avg Satisfaction']['Change'].iloc[0]
    if abs(satisfaction_change) > 0.2:
        direction = "improved" if satisfaction_change > 0 else "declined"
        insights.append(f"😊 Customer satisfaction {direction} by {abs(satisfaction_change):.1f} points")
    
    for insight in insights:
        st.markdown(f"• {insight}")

def show_actionable_insights(df):
    """Show actionable insights based on trend analysis"""
    
    st.markdown("### 💡 Actionable Insights")
    
    monthly_stats = calculate_monthly_statistics(df)
    
    # Generate insights based on trends
    insights = []
    recommendations = []
    
    # Volume trend analysis
    if len(monthly_stats) >= 3:
        recent_volumes = monthly_stats['ticket_count'].tail(3)
        volume_trend = recent_volumes.pct_change().mean()
        
        if volume_trend > 0.1:  # 10% increase trend
            insights.append("📈 Ticket volume is trending upward consistently")
            recommendations.append("Consider scaling support team or implementing self-service solutions")
        elif volume_trend < -0.1:  # 10% decrease trend
            insights.append("📉 Ticket volume is decreasing - improvements are working")
            recommendations.append("Document and replicate successful strategies")
    
    # Satisfaction trend analysis
    satisfaction_trend = monthly_stats['avg_satisfaction'].diff().mean()
    if satisfaction_trend < -0.1:
        insights.append("😟 Customer satisfaction is declining over time")
        recommendations.append("Urgently review support processes and implement customer feedback loop")
    elif satisfaction_trend > 0.1:
        insights.append("😊 Customer satisfaction is improving consistently")
        recommendations.append("Continue current strategies and share best practices")
    
    # Resolution time analysis
    resolution_trend = monthly_stats['avg_resolution_time'].diff().mean()
    if resolution_trend > 2:  # 2+ hours increase per month
        insights.append("⏰ Resolution times are increasing")
        recommendations.append("Investigate bottlenecks and consider automation or additional training")
    
    # Priority analysis
    priority_trend = monthly_stats['avg_priority'].diff().mean()
    if priority_trend > 0.3:
        insights.append("🔥 Issue severity is increasing")
        recommendations.append("Focus on root cause analysis and preventive measures")
    
    # Display insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 Key Insights")
        if insights:
            for insight in insights:
                st.markdown(f"• {insight}")
        else:
            st.markdown("• Trends are stable with no significant changes")
    
    with col2:
        st.markdown("#### 🎯 Recommendations")
        if recommendations:
            for rec in recommendations:
                st.markdown(f"• {rec}")
        else:
            st.markdown("• Continue monitoring current performance")
    
    # Strategic recommendations based on overall patterns
    st.markdown("#### 🚀 Strategic Recommendations")
    
    # Calculate overall health score
    latest_data = monthly_stats.iloc[-1]
    health_factors = {
        'satisfaction': min(latest_data['avg_satisfaction'] / 5 * 100, 100),
        'priority': max(0, 100 - (latest_data['avg_priority'] / 10 * 100)),
        'resolution': max(0, 100 - min(latest_data['avg_resolution_time'] / 48 * 100, 100))
    }
    
    overall_health = sum(health_factors.values()) / len(health_factors)
    
    if overall_health >= 80:
        st.markdown("""
        <div class="success-box">
            <h4>✅ Excellent Performance</h4>
            <p>Your support metrics are performing well. Focus on:</p>
            <ul>
                <li>Maintaining current quality standards</li>
                <li>Sharing best practices across teams</li>
                <li>Exploring proactive support opportunities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif overall_health >= 60:
        st.markdown("""
        <div class="forecast-box">
            <h4>📊 Good Performance with Room for Improvement</h4>
            <p>Performance is solid but can be enhanced. Consider:</p>
            <ul>
                <li>Identifying top 3 improvement areas</li>
                <li>Implementing targeted training programs</li>
                <li>Enhancing documentation and self-service</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-box">
            <h4>⚠️ Performance Needs Attention</h4>
            <p>Multiple metrics need improvement. Priority actions:</p>
            <ul>
                <li>Conduct comprehensive process review</li>
                <li>Implement immediate corrective measures</li>
                <li>Set up weekly performance monitoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

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
            except:
                sdk_issues.append([])
        else:
            sdk_issues.append([])
    
    return sdk_issues

if __name__ == "__main__":
    main()

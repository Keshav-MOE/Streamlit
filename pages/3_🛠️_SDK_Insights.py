import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import re
from collections import Counter
# Remove these unused imports:
# import nltk
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import io

st.set_page_config(
    page_title="SDK Insights",
    page_icon="🛠️",
    layout="wide"
)

# Custom CSS (keep your existing CSS)
st.markdown("""
<style>
    .insight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .improvement-item {
        background-color: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #28a745;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .priority-high {
        border-left-color: #dc3545;
        background-color: #f8d7da;
    }
    .priority-medium {
        border-left-color: #ffc107;
        background-color: #fff3cd;
    }
    .priority-low {
        border-left-color: #28a745;
        background-color: #d4edda;
    }
    .sdk-metric {
        text-align: center;
        padding: 1rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🛠️ SDK Improvement Insights")
    
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
    
    # Main sections
    show_sdk_overview(df)
    show_issue_analysis(df)
    show_improvement_recommendations(df)
    show_impact_analysis(df)
    show_priority_matrix(df)
    show_detailed_insights(df)

def show_empty_state():
    """Show empty state when no data is available"""
    
    st.markdown("""
    <div class="insight-card">
        <h2>🛠️ No SDK Data Available</h2>
        <p>Upload and analyze some tickets to see SDK improvement insights!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📁 Upload Data Now", type="primary", use_container_width=True):
            st.switch_page("pages/1_📁_Upload_Data.py")

def show_sdk_overview(df):
    """Show SDK-related overview metrics"""
    
    st.markdown("### 🎯 SDK Overview")
    
    # Extract and analyze SDK issues
    all_sdk_issues = extract_sdk_issues(df)
    all_improvements = extract_improvement_suggestions(df)
    
    # Calculate SDK metrics
    total_sdk_tickets = len([issues for issues in all_sdk_issues if issues])
    unique_issues = len(set([item for sublist in all_sdk_issues for item in sublist]))
    avg_priority_sdk = df[df['sdk_issues'].notna()]['priority_score'].mean() if not df.empty else 0
    high_impact_issues = len([issues for issues in all_sdk_issues if any('critical' in str(issue).lower() or 'urgent' in str(issue).lower() for issue in issues)])
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="sdk-metric">
            <h2 style="color: #1f77b4; margin: 0;">{total_sdk_tickets}</h2>
            <p style="margin: 0;">SDK-Related Tickets</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="sdk-metric">
            <h2 style="color: #ff7f0e; margin: 0;">{unique_issues}</h2>
            <p style="margin: 0;">Unique SDK Issues</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="sdk-metric">
            <h2 style="color: #2ca02c; margin: 0;">{avg_priority_sdk:.1f}</h2>
            <p style="margin: 0;">Avg Priority Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="sdk-metric">
            <h2 style="color: #d62728; margin: 0;">{high_impact_issues}</h2>
            <p style="margin: 0;">High Impact Issues</p>
        </div>
        """, unsafe_allow_html=True)

def show_issue_analysis(df):
    """Show detailed issue analysis"""
    
    st.markdown("### 🔍 Issue Analysis")
    
    # Extract all SDK issues
    all_sdk_issues = extract_sdk_issues(df)
    flat_issues = [item for sublist in all_sdk_issues for item in sublist if item]
    
    if not flat_issues:
        st.info("No SDK issues found in the data.")
        return
    
    # Count issues
    issue_counts = Counter(flat_issues)
    top_issues = dict(issue_counts.most_common(15))
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top issues bar chart
        if top_issues:
            fig = px.bar(
                x=list(top_issues.values()),
                y=list(top_issues.keys()),
                orientation='h',
                title="🔥 Top SDK Issues",
                labels={'x': 'Frequency', 'y': 'Issue Type'},
                color=list(top_issues.values()),
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No issues data to display")
    
    with col2:
        # Issue categories pie chart
        categorized_issues = categorize_issues(flat_issues)
        
        if categorized_issues:
            fig = px.pie(
                values=list(categorized_issues.values()),
                names=list(categorized_issues.keys()),
                title="📊 Issue Categories",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textinfo='percent+label')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorized issues to display")
    
    # Show issue trends over time
    if df['month'].nunique() > 1:
        show_issue_trends(df, flat_issues)

def categorize_issues(issues):
    """Categorize SDK issues into main areas"""
    
    categories = {
        'Integration':     ['integration', 'setup', 'installation', 'config'],
        'Documentation':   ['doc', 'guide', 'example', 'tutorial', 'help'],
        'API':            ['api', 'endpoint', 'response', 'request', 'method'],
        'Performance':    ['slow', 'performance', 'speed', 'timeout', 'latency'],
        'Authentication': ['auth', 'login', 'token', 'permission', 'access'],
        'Error Handling': ['error', 'exception', 'crash', 'failure', 'bug'],
        'Compatibility':  ['version', 'compatibility', 'browser', 'platform'],
        'Features':       ['feature', 'functionality', 'missing', 'request']
    }
    
    categorized = {cat: 0 for cat in categories.keys()}
    categorized['Other'] = 0
    
    for issue in issues:
        issue_lower = str(issue).lower()
        matched = False
        
        for category, keywords in categories.items():
            if any(keyword in issue_lower for keyword in keywords):
                categorized[category] += 1
                matched = True
                break
        
        if not matched:
            categorized['Other'] += 1
    
    # Remove empty categories
    return {k: v for k, v in categorized.items() if v > 0}

def show_issue_trends(df, issues):
    """Show how issues trend over months"""
    
    st.markdown("#### 📈 Issue Trends Over Time")
    
    # Get top 5 issues for trending
    top_issues = Counter(issues).most_common(5)
    
    if not top_issues:
        st.info("No trending data available")
        return
    
    monthly_trends = {}
    
    for month in df['month'].unique():
        month_df = df[df['month'] == month]
        month_issues = extract_sdk_issues(month_df)
        month_flat = [item for sublist in month_issues for item in sublist]
        
        monthly_trends[month] = Counter(month_flat)
    
    # Create trend data
    trend_data = []
    for issue, _ in top_issues:
        for month in sorted(df['month'].unique()):
            count = monthly_trends[month].get(issue, 0)
            trend_data.append({'Month': month, 'Issue': issue, 'Count': count})
    
    if trend_data:
        trend_df = pd.DataFrame(trend_data)
        
        fig = px.line(
            trend_df,
            x='Month',
            y='Count',
            color='Issue',
            title='📈 Top Issues Trend',
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_improvement_recommendations(df):
    """Show improvement recommendations"""
    
    st.markdown("### 💡 Improvement Recommendations")
    
    # Extract improvement suggestions
    all_improvements = extract_improvement_suggestions(df)
    flat_improvements = [item for sublist in all_improvements for item in sublist if item]
    
    if not flat_improvements:
        st.info("No improvement suggestions found in the data.")
        return
    
    # Count and prioritize improvements
    improvement_counts = Counter(flat_improvements)
    top_improvements = improvement_counts.most_common(10)
    
    # Categorize by priority (based on frequency and keywords)
    categorized_improvements = []
    
    for improvement, count in top_improvements:
        priority = calculate_improvement_priority(improvement, count, len(df))
        impact = estimate_impact(improvement)
        
        categorized_improvements.append({
            'improvement': improvement,
            'frequency': count,
            'priority': priority,
            'impact': impact,
            'percentage': (count / len(df)) * 100 if len(df) > 0 else 0
        })
    
    # Sort by priority and display
    categorized_improvements.sort(key=lambda x: (x['priority'] == 'High', x['frequency']), reverse=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🎯 Prioritized Improvements")
        
        for i, item in enumerate(categorized_improvements[:8]):
            priority_class = f"priority-{item['priority'].lower()}"
            
            st.markdown(f"""
            <div class="improvement-item {priority_class}">
                <h4 style="margin: 0 0 0.5rem 0;">#{i+1} {item['improvement']}</h4>
                <p style="margin: 0;">
                    <strong>Frequency:</strong> {item['frequency']} tickets ({item['percentage']:.1f}%) |
                    <strong>Priority:</strong> {item['priority']} |
                    <strong>Impact:</strong> {item['impact']}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Priority distribution
        priority_dist = {}
        for item in categorized_improvements:
            priority_dist[item['priority']] = priority_dist.get(item['priority'], 0) + 1
        
        if priority_dist:
            fig = px.pie(
                values=list(priority_dist.values()),
                names=list(priority_dist.keys()),
                title="🎯 Priority Distribution",
                color_discrete_map={
                    'High': '#dc3545',
                    'Medium': '#ffc107', 
                    'Low': '#28a745'
                }
            )
            st.plotly_chart(fig, use_container_width=True)

def calculate_improvement_priority(improvement, frequency, total_tickets):
    """Calculate improvement priority based on various factors"""
    
    # High priority keywords
    high_priority_keywords = [
        'critical', 'urgent', 'crash', 'security', 'performance',
        'auth', 'authentication', 'api', 'integration'
    ]
    
    # Medium priority keywords  
    medium_priority_keywords = [
        'documentation', 'example', 'guide', 'error', 'bug',
        'feature', 'usability', 'interface'
    ]
    
    improvement_lower = str(improvement).lower()
    frequency_ratio = frequency / total_tickets if total_tickets > 0 else 0
    
    # High priority conditions
    if (any(keyword in improvement_lower for keyword in high_priority_keywords) or
        frequency_ratio > 0.1):
        return 'High'
    
    # Medium priority conditions
    elif (any(keyword in improvement_lower for keyword in medium_priority_keywords) or
          frequency_ratio > 0.05):
        return 'Medium'
    
    # Default to low
    else:
        return 'Low'

def estimate_impact(improvement):
    """Estimate the impact of an improvement"""
    
    high_impact_keywords = ['api', 'integration', 'authentication', 'performance', 'security']
    medium_impact_keywords = ['documentation', 'example', 'usability', 'interface']
    
    improvement_lower = str(improvement).lower()
    
    if any(keyword in improvement_lower for keyword in high_impact_keywords):
        return 'High'
    elif any(keyword in improvement_lower for keyword in medium_impact_keywords):
        return 'Medium'
    else:
        return 'Low'

def show_impact_analysis(df):
    """Show potential impact analysis"""
    
    st.markdown("### 🎯 Impact Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Resolution Time vs Priority")
        
        # Scatter plot of resolution time vs priority colored by satisfaction
        if 'priority_score' in df.columns and 'resolution_time_hours' in df.columns:
            fig = px.scatter(
                df,
                x='priority_score',
                y='resolution_time_hours',
                color='customer_satisfaction',
                size='customer_satisfaction',
                title='Resolution Time vs Priority',
                labels={
                    'priority_score': 'Priority Score',
                    'resolution_time_hours': 'Resolution Time (Hours)',
                    'customer_satisfaction': 'Satisfaction'
                },
                color_continuous_scale='RdYlGn'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for resolution time analysis")
    
    with col2:
        st.markdown("#### 💰 Potential Cost Savings")
        
        # Calculate potential savings
        if 'resolution_time_hours' in df.columns and not df.empty:
            avg_resolution_time = df['resolution_time_hours'].mean()
            high_priority_tickets = len(df[df['priority_score'] >= 7]) if 'priority_score' in df.columns else 0
            
            # Assuming $50/hour for support cost
            hourly_cost = 50
            current_cost = avg_resolution_time * len(df) * hourly_cost
            
            # Potential 20% reduction with improvements
            potential_savings = current_cost * 0.2
            
            st.markdown(f"""
            <div class="insight-card">
                <h3>💰 Potential Monthly Savings</h3>
                <h2>${potential_savings:,.0f}</h2>
                <p>Based on 20% reduction in resolution time</p>
                <hr>
                <p><strong>Current Cost:</strong> ${current_cost:,.0f}/month</p>
                <p><strong>Avg Resolution Time:</strong> {avg_resolution_time:.1f} hours</p>
                <p><strong>High Priority Tickets:</strong> {high_priority_tickets}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Insufficient data for cost analysis")

def show_priority_matrix(df):
    """Show improvement priority matrix"""
    
    st.markdown("### 📋 Development Priority Matrix")
    
    # Create priority matrix data
    improvements = extract_improvement_suggestions(df)
    flat_improvements = [item for sublist in improvements for item in sublist if item]
    
    if not flat_improvements:
        st.info("No improvement data available for priority matrix")
        return
    
    improvement_counts = Counter(flat_improvements)
    
    matrix_data = []
    for improvement, count in improvement_counts.most_common(20):
        frequency_score = min(count / len(df) * 100, 10) if len(df) > 0 else 0  # Max 10
        complexity_score = estimate_complexity(improvement)  # 1-10 scale
        impact_score = estimate_impact_score(improvement)  # 1-10 scale
        priority_score = (frequency_score + impact_score) / max(complexity_score, 1)  # Avoid division by zero
        
        matrix_data.append({
            'Improvement': improvement[:50] + '...' if len(improvement) > 50 else improvement,
            'Frequency': count,
            'Impact': impact_score,
            'Complexity': complexity_score,
            'Priority Score': priority_score
        })
    
    if matrix_data:
        # Sort by priority score
        matrix_data.sort(key=lambda x: x['Priority Score'], reverse=True)
        
        # Create bubble chart
        matrix_df = pd.DataFrame(matrix_data)
        
        fig = px.scatter(
            matrix_df,
            x='Complexity',
            y='Impact',
            size='Frequency',
            hover_name='Improvement',
            title='🎯 Priority Matrix: Impact vs Complexity',
            labels={
                'Complexity': 'Implementation Complexity (1-10)',
                'Impact': 'Business Impact (1-10)',
                'Frequency': 'Frequency'
            },
            color='Priority Score',
            color_continuous_scale='RdYlGn'
        )
        
        # Add quadrant lines
        fig.add_hline(y=5, line_dash="dash", line_color="gray", annotation_text="Medium Impact")
        fig.add_vline(x=5, line_dash="dash", line_color="gray", annotation_text="Medium Complexity")
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show top priorities table
        st.markdown("#### 🏆 Top Development Priorities")
        priority_df = matrix_df[['Improvement', 'Frequency', 'Impact', 'Complexity', 'Priority Score']].head(10)
        st.dataframe(priority_df, use_container_width=True, hide_index=True)

def estimate_complexity(improvement):
    """Estimate implementation complexity (1-10 scale)"""
    
    high_complexity_keywords = ['api', 'integration', 'architecture', 'security', 'performance']
    medium_complexity_keywords = ['feature', 'interface', 'validation', 'workflow']
    low_complexity_keywords = ['documentation', 'example', 'message', 'text', 'copy']
    
    improvement_lower = str(improvement).lower()
    
    if any(keyword in improvement_lower for keyword in high_complexity_keywords):
        return 8  # High complexity
    elif any(keyword in improvement_lower for keyword in medium_complexity_keywords):
        return 5  # Medium complexity
    elif any(keyword in improvement_lower for keyword in low_complexity_keywords):
        return 2  # Low complexity
    else:
        return 5  # Default medium

def estimate_impact_score(improvement):
    """Estimate business impact score (1-10 scale)"""
    
    high_impact_keywords = ['critical', 'security', 'performance', 'integration', 'api']
    medium_impact_keywords = ['usability', 'feature', 'workflow', 'interface']
    
    improvement_lower = str(improvement).lower()
    
    if any(keyword in improvement_lower for keyword in high_impact_keywords):
        return 9
    elif any(keyword in improvement_lower for keyword in medium_impact_keywords):
        return 6
    else:
        return 3

def show_detailed_insights(df):
    """Show detailed insights and actionable items"""
    
    st.markdown("### 📝 Detailed Insights & Action Items")
    
    # Generate insights
    insights = generate_insights(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Key Insights")
        if insights['key_insights']:
            for i, insight in enumerate(insights['key_insights'], 1):
                st.markdown(f"**{i}.** {insight}")
        else:
            st.info("No specific insights generated yet.")
    
    with col2:
        st.markdown("#### ✅ Recommended Actions")
        if insights['recommended_actions']:
            for i, action in enumerate(insights['recommended_actions'], 1):
                st.markdown(f"**{i}.** {action}")
        else:
            st.info("No specific actions recommended yet.")
    
    # Export functionality
    st.markdown("#### 📤 Export Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Dashboard Data", use_container_width=True):
            export_dashboard_data(df)
    
    with col2:
        if st.button("📋 Generate Report", use_container_width=True):
            generate_insights_report(df, insights)
    
    with col3:
        if st.button("📧 Email Summary", use_container_width=True):
            st.info("Email functionality would be implemented here")

def generate_insights(df):
    """Generate key insights from the data"""
    
    insights = {
        'key_insights': [],
        'recommended_actions': []
    }
    
    if df.empty:
        return insights
    
    # Analyze data for insights
    total_tickets = len(df)
    avg_priority = df['priority_score'].mean() if 'priority_score' in df.columns else 0
    avg_satisfaction = df['customer_satisfaction'].mean() if 'customer_satisfaction' in df.columns else 0
    
    # Generate insights based on data
    if avg_priority > 6:
        insights['key_insights'].append(
            f"High average priority score ({avg_priority:.1f}/10) indicates serious SDK issues requiring immediate attention."
        )
        insights['recommended_actions'].append(
            "Prioritize fixing high-impact SDK issues to reduce support burden."
        )
    
    if avg_satisfaction < 3.5:
        insights['key_insights'].append(
            f"Low customer satisfaction ({avg_satisfaction:.1f}/5) suggests significant user experience problems."
        )
        insights['recommended_actions'].append(
            "Implement customer feedback loop and improve SDK documentation."
        )
    
    # Add more insights based on common issues
    all_issues = extract_sdk_issues(df)
    flat_issues = [item for sublist in all_issues for item in sublist if item]
    
    if flat_issues:
        top_issue = Counter(flat_issues).most_common(1)[0]
        insights['key_insights'].append(
            f"Most common issue '{top_issue[0]}' affects {(top_issue[1]/total_tickets)*100:.1f}% of tickets."
        )
        insights['recommended_actions'].append(
            f"Create targeted solution for '{top_issue[0]}' to reduce ticket volume."
        )
    
    return insights

def export_dashboard_data(df):
    """Export dashboard data to CSV"""
    
    # Prepare export data
    export_df = df.copy()
    
    # Add processed columns
    if 'sdk_issues' in export_df.columns:
        export_df['sdk_issues_count'] = export_df['sdk_issues'].apply(
            lambda x: len(json.loads(x)) if pd.notna(x) and x else 0
        )
    
    # Convert to CSV
    csv = export_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"sdk_insights_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

def generate_insights_report(df, insights):
    """Generate a comprehensive insights report"""
    
    report = f"""
# SDK Insights Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- Total Tickets Analyzed: {len(df):,}
- Average Priority Score: {df['priority_score'].mean():.1f}/10 if 'priority_score' in df.columns else 'N/A'
- Average Customer Satisfaction: {df['customer_satisfaction'].mean():.1f}/5 if 'customer_satisfaction' in df.columns else 'N/A'
- Average Resolution Time: {df['resolution_time_hours'].mean():.1f} hours if 'resolution_time_hours' in df.columns else 'N/A'

## Key Insights
{chr(10).join([f"• {insight}" for insight in insights['key_insights']])}

## Recommended Actions  
{chr(10).join([f"• {action}" for action in insights['recommended_actions']])}

## Top SDK Issues
{chr(10).join([f"• {issue}" for issue, _ in Counter([item for sublist in extract_sdk_issues(df) for item in sublist]).most_common(10)])}
    """
    
    st.download_button(
        label="📄 Download Report",
        data=report,
        file_name=f"sdk_insights_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )

def extract_sdk_issues(df):
    """Extract SDK issues from the dataframe"""
    
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

def extract_improvement_suggestions(df):
    """Extract improvement suggestions from the dataframe"""
    
    improvements = []
    for _, row in df.iterrows():
        if pd.notna(row.get('improvement_suggestions')):
            try:
                suggestions = json.loads(row['improvement_suggestions']) if isinstance(row['improvement_suggestions'], str) else row['improvement_suggestions']
                if isinstance(suggestions, list):
                    improvements.append(suggestions)
                else:
                    improvements.append([])
            except (json.JSONDecodeError, TypeError):
                improvements.append([])
        else:
            improvements.append([])
    
    return improvements

if __name__ == "__main__":
    main()

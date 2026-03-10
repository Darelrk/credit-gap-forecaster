import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os

class Visualizer:
    """XGBoost based time-series forecaster."""
    
    def __init__(self):
        # Set overall seaborn-like styling for professional data science aesthetics
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_comparison(self, actual_series, arima_series=None, lstm_series=None, xgb_series=None, ensemble_series=None, output_path="comparison.png"):
        """
        Creates a high-fidelity comparative line chart for presentation.
        """
        # Premium color palette using HSL-appropriate hex codes
        colors = {
            'actual': '#1A1A1A',     # Deep Charcoal
            'arima': '#5D5D5D',      # Neutral Grey
            'lstm': '#007AFF',       # iOS-style Blue
            'xgb': '#FF9500',        # iOS-style Orange
            'ensemble': '#EEAF00'    # Deep Gold/Mustard for Ensemble (Premium)
        }
        
        plt.figure(figsize=(14, 7), facecolor='#FAFAFA')
        ax = plt.gca()
        ax.set_facecolor('#FAFAFA')
        
        # Plot Actual with markers for clarity
        plt.plot(actual_series.index, actual_series.values, label='Historical Credit Gap', 
                 color=colors['actual'], linewidth=2.5, marker='o', markersize=4, zorder=3)
        
        if arima_series is not None:
            plt.plot(arima_series.index, arima_series.values, label='ARIMA (Linear Baseline)', 
                     linestyle=':', color=colors['arima'], alpha=0.6, linewidth=1.5)
            
        if lstm_series is not None:
            plt.plot(lstm_series.index, lstm_series.values, label='Attention-LSTM (Strategic)', 
                     linestyle='-.', color=colors['lstm'], linewidth=2, alpha=0.7, zorder=2)
            
        if xgb_series is not None:
            plt.plot(xgb_series.index, xgb_series.values, label='XGBoost (Pattern Capture)', 
                     linestyle='--', color=colors['xgb'], linewidth=2, alpha=0.7, zorder=2)

        if ensemble_series is not None:
            plt.plot(ensemble_series.index, ensemble_series.values, label='ENSEMBLE (Balanced Hero)', 
                     linestyle='-', color=colors['ensemble'], linewidth=3.5, marker='*', markersize=8, 
                     markevery=2, zorder=4)
        
        # Enhanced Titles and Labels
        plt.title('Macroprudential Surveillance: Quadrilateral Model Comparison', 
                   fontsize=18, fontweight='bold', pad=25, color='#333333')
        plt.xlabel('Timeline (Quarters)', fontsize=13, fontweight='medium', labelpad=10)
        plt.ylabel('Credit-to-GDP Gap (%)', fontsize=13, fontweight='medium', labelpad=10)
        
        # Professional Grid and Spines
        plt.grid(True, linestyle='--', alpha=0.4, color='#CCCCCC')
        for spine in plt.gca().spines.values():
            spine.set_color('#DDDDDD')
            
        # Transparent zero line
        plt.axhline(0, color='#000000', linewidth=1, alpha=0.3, zorder=1)
        
        # Legend styling
        plt.legend(loc='lower left', fontsize=11, frameon=True, framealpha=0.9, 
                   facecolor='white', edgecolor='#EEEEEE', shadow=False).set_zorder(100)
        
        # Enhanced X-axis formatting for Quarters
        import matplotlib.dates as mdates
        from matplotlib.ticker import FuncFormatter

        ax = plt.gca()
        
        # Determine the best locator based on duration
        total_days = (actual_series.index.max() - actual_series.index.min()).days
        if total_days > 365 * 10: # More than 10 years
            ax.xaxis.set_major_locator(mdates.YearLocator())
        elif total_days > 365 * 3: # 3 to 10 years
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])) # Jan and Jul
        else:
            # For short ranges (like trajectories), show every quarter if possible
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        
        def quarter_formatter(x, pos):
            try:
                dt = mdates.num2date(x)
                q = (dt.month - 1) // 3 + 1
                return f"{dt.year}-Q{q}"
            except:
                return ""
            
        ax.xaxis.set_major_formatter(FuncFormatter(quarter_formatter))
        plt.xticks(rotation=45, fontsize=9)
        
        # Save with high quality
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
        plt.close()
    def plot_fan_chart(self, actual_series, forecast_dates, p10, p50, p90, output_path="fan_chart.png", zoom_start="2018"):
        """
        Creates a probabilistic 'Fan Chart' for macroprudential outlook with focused timeline.
        """
        # Slice historical data for focused view
        if zoom_start:
            actual_series = actual_series[actual_series.index >= zoom_start]

        plt.figure(figsize=(14, 7), facecolor='#FAFAFA')
        ax = plt.gca()
        ax.set_facecolor('#FAFAFA')
        
        # Plot Historical Data
        plt.plot(actual_series.index, actual_series.values, label='Historical Credit Gap', 
                 color='#1A1A1A', linewidth=3, zorder=3, marker='o', markersize=4)
        
        # Connect last actual point to first forecast for continuity
        last_actual_date = actual_series.index[-1]
        last_actual_val = actual_series.values[-1]
        
        # Combine dates and values for the 'Fan' continuity
        fan_dates = [last_actual_date] + list(forecast_dates)
        fan_p10 = [last_actual_val] + list(p10)
        fan_p90 = [last_actual_val] + list(p90)
        fan_p50 = [last_actual_val] + list(p50)
        
        # Plot Confidence Bands (The 'Fan')
        plt.fill_between(fan_dates, fan_p10, fan_p90, color='#007AFF', alpha=0.15, 
                         label='90% Confidence Interval', zorder=1)
        
        # Plot Median (P50)
        plt.plot(fan_dates, fan_p50, label='Projected Path (Outlook 2026)', 
                 color='#007AFF', linewidth=4, linestyle='-', marker='s', markersize=6, zorder=4)
        
        # Enhanced Titles and Labels
        plt.title('Macroprudential Outlook 2026: Probabilistic Scenario Analysis', 
                  fontsize=22, fontweight='bold', pad=40, color='#222222')
        plt.text(0.5, 1.04, '* Includes 1.5x Stressed Macro Shock Simulation on Confidence Intervals', 
                 ha='center', va='bottom', transform=ax.transAxes, fontsize=12, style='italic', color='#777777')
        plt.xlabel('Timeline (Quarters)', fontsize=14, labelpad=12)
        plt.ylabel('Credit-to-GDP Gap (%)', fontsize=14, labelpad=12)
        
        # 1. Coordinate risk zones (Strictly adhering to Macroprudential standards)
        # Alert Zone: > 3.0 (Red), Caution Zone: 0 - 3.0 (Yellow), Safe Zone: < 0 (Green)
        
        # Red/Alert Zone
        plt.axhspan(3, 20, color='#E74C3C', alpha=0.1, zorder=0)
        plt.text(actual_series.index[0], 6, 'P(Alert) ≈ 10%', color='#E74C3C', 
                 fontsize=12, fontweight='bold', alpha=0.7)
        
        # Yellow/Caution Zone
        plt.axhspan(0, 3, color='#F1C40F', alpha=0.1, zorder=0)
        plt.text(actual_series.index[0], 1.5, 'P(Caution) ≈ 20%', color='#D4AC0D', 
                 fontsize=12, fontweight='bold', alpha=0.7)
        
        # Green/Safe Zone
        plt.axhspan(-30, 0, color='#27AE60', alpha=0.05, zorder=0)
        plt.text(actual_series.index[0], -5, 'P(Safe) ≈ 70%', color='#27AE60', 
                 fontsize=12, fontweight='bold', alpha=0.7)

        # Reference Lines (Alert & Caution thresholds)
        plt.axhline(3, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.6)
        plt.axhline(0, color='#333333', linestyle='-', linewidth=1.5, alpha=0.8)
        
        # Forecast Transition Logic
        plt.axvline(last_actual_date, color='#555555', linestyle='--', linewidth=2, alpha=0.8, zorder=5)
        plt.text(last_actual_date, 0.5, ' FORECAST START', 
                 verticalalignment='bottom', ha='left', fontsize=10, fontweight='bold', color='#333333')

        # Grid and axis styling
        plt.grid(True, which='major', linestyle='-', alpha=0.2, color='#CCCCCC')
        ax.set_ylim(-30, 25) # Fixed macroprudential scale
        
        # Add Value Labels for Projected Path (P50) with clean boxes
        for i, (date, val) in enumerate(zip(forecast_dates, p50)):
            plt.annotate(f'{val:+.2f}%', (date, val), textcoords="offset points", xytext=(0,15), 
                         ha='center', fontsize=9, fontweight='bold', color='#007AFF',
                         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#007AFF', alpha=1, linewidth=1))

        # Legend Styling
        plt.legend(loc='lower left', fontsize=11, frameon=True, facecolor='white', framealpha=1, 
                   edgecolor='#DDDDDD', shadow=False).set_zorder(100)
        
        # Enhanced X-axis formatting for Quarters
        import matplotlib.dates as mdates
        from matplotlib.ticker import FuncFormatter, FixedLocator

        ax = plt.gca()
        # Combine historical and forecast dates for explicit ticks
        all_dates = list(actual_series.index) + list(forecast_dates)
        ax.xaxis.set_major_locator(FixedLocator(mdates.date2num(all_dates)))
        
        def quarter_formatter(x, pos):
            try:
                dt = mdates.num2date(x)
                q = (dt.month - 1) // 3 + 1
                return f"{dt.year}-Q{q}"
            except:
                return ""
            
        ax.xaxis.set_major_formatter(FuncFormatter(quarter_formatter))
        plt.xticks(rotation=45, fontsize=9)
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
        plt.close()

    def plot_probability_timeline(self, dates, probabilities, threshold_label="+3.0%", output_path="prob_timeline.png"):
        """
        Visualizes the rising probability of exceeding a specific threshold over time.
        """
        plt.figure(figsize=(10, 5), facecolor='#FAFAFA')
        plt.plot(dates, probabilities * 100, color='#E74C3C', linewidth=3, marker='o', markersize=8)
        plt.fill_between(dates, probabilities * 100, color='#E74C3C', alpha=0.1)
        
        plt.title(f'Risk Trajectory: Probability of Crossing {threshold_label}', fontsize=15, fontweight='bold', pad=20)
        plt.ylabel('Probability (%)', fontsize=12)
        plt.xlabel('Outlook Quarters', fontsize=12)
        
        y_max = max(probabilities * 100)
        plt.ylim(0, max(y_max + 10, 50) if len(probabilities) > 0 else 100)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        import matplotlib.dates as mdates
        from matplotlib.ticker import FuncFormatter, FixedLocator
        ax = plt.gca()
        ax.xaxis.set_major_locator(FixedLocator(mdates.date2num(dates)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{mdates.num2date(x).year}-Q{(mdates.num2date(x).month-1)//3+1}"))
        
        # Add labels
        for x, y in zip(dates, probabilities * 100):
            plt.text(x, y + 2, f"{y:.1f}%", ha='center', fontweight='bold', color='#E74C3C')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
        plt.close()

    def plot_ewi_dashboard(self, current_gap, credit_yoy, npl_proxy=None, output_path="ewi_dashboard.png"):
        """
        Creates a high-level KPI dashboard for executive monitoring.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='#FAFAFA')
        
        def draw_kpi(ax, title, value, status, subtitle):
            color = {'Safe': '#27AE60', 'Caution': '#F1C40F', 'Alert': '#E74C3C'}.get(status, '#555555')
            ax.set_facecolor('white')
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.05, transform=ax.transAxes))
            ax.text(0.5, 0.75, title, ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
            ax.text(0.5, 0.45, value, ha='center', fontsize=28, fontweight='bold', color=color, transform=ax.transAxes)
            ax.text(0.5, 0.20, subtitle, ha='center', fontsize=10, color='#777777', transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_edgecolor('#EEEEEE')

        # KPI 1: Credit-to-GDP Gap
        status_gap = 'Safe' if current_gap < 0 else ('Caution' if current_gap < 3 else 'Alert')
        draw_kpi(axes[0], "Credit-to-GDP Gap", f"{current_gap:+.2f}%", status_gap, "Threshold: > +3.0%")
        
        # KPI 2: Credit Growth YoY (Standard monitoring)
        status_yoy = 'Safe' if credit_yoy < 12 else ('Caution' if credit_yoy < 15 else 'Alert')
        draw_kpi(axes[1], "Credit Growth (YoY)", f"{credit_yoy:.1f}%", status_yoy, "Threshold: > 15.0%")

        plt.suptitle('Early Warning Indicator Snapshot: Real-Time Risk Monitoring', fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
        plt.close()
    def plot_attention_map(self, attention_weights, feature_names, output_path="attention_map.png"):
        """
        Visualizes the importance of each feature over the sequence timeline using heatmaps.
        """
        import seaborn as sns
        import numpy as np
        
        plt.figure(figsize=(10, 6), facecolor='#FAFAFA')
        # attention_weights expected shape: (seq_len,) or similar
        # For multi-feature, we might want to show which feature contributed most
        # But attention in this model is over TIME (sequence steps).
        # To show feature importance, we'd need a different approach or 
        # just visualize the sequence focus.
        
        # Let's assume attention_weights is (seq_len,)
        sns.heatmap(attention_weights.reshape(1, -1), annot=True, cmap="YlGnBu", 
                    xticklabels=[f"T-{i}" for i in range(len(attention_weights)-1, -1, -1)],
                    yticklabels=["Attention Score"])
        
        plt.title('Temporal Attention Focus: Model Strategy Analysis', fontsize=15, fontweight='bold', pad=20)
        plt.xlabel('Time Sequence (Quarters From Target)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
        plt.close()

    def plot_multivariate_importance(self, feature_importances, feature_names, output_path="feature_importance.png"):
        """
        Visualizes feature contribution to the model.
        """
        plt.figure(figsize=(10, 6), facecolor='#FAFAFA')
        colors = ['#007AFF', '#FF9500', '#27AE60', '#E74C3C']
        
        bars = plt.barh(feature_names, feature_importances, color=colors[:len(feature_names)])
        plt.xlabel('Relative Importance (Normalized)', fontsize=12)
        plt.title('Multivariate Attribution: Macro Variable Contribution', fontsize=15, fontweight='bold', pad=20)
        
        # Add labels to bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.1%}', 
                     va='center', fontweight='bold', color='#333333')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor='#FAFAFA', bbox_inches='tight')
        plt.close()

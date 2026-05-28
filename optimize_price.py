import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
import os
warnings.filterwarnings('ignore')

def run_xgboost_optimization(market_context: dict) -> dict:
    csv_path = os.path.join(os.path.dirname(__file__), "retail_price.csv")
    df = pd.read_csv(csv_path)
    features = [
        'unit_price', 'freight_price', 'product_weight_g', 'product_score',
        'comp_1', 'comp_2', 'comp_3', 'month', 'holiday', 'weekend', 's', 'lag_price'
    ]
    X = df[features]
    y = df['qty']

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)

    # Extract top 3 feature importances (excluding unit_price)
    importances = model.feature_importances_
    feature_imp_dict = {features[i]: importances[i] for i in range(len(features)) if features[i] != 'unit_price'}
    sorted_features = sorted(feature_imp_dict.items(), key=lambda x: x[1], reverse=True)
    top_3_raw = [f[0] for f in sorted_features[:3]]
    
    human_readable_map = {
        'freight_price': "Freight Price",
        'product_weight_g': "Product Weight",
        'comp_1': "Primary Competitor Price",
        'comp_2': "Secondary Competitor Price",
        'comp_3': "Tertiary Competitor Price",
        'product_score': "Our Product Rating",
        'lag_price': "Lag Price (Historical)",
        's': "Seasonality",
        'month': "Month of Year",
        'holiday': "Holiday Flag",
        'weekend': "Weekend Shopping"
    }
    key_drivers = [human_readable_map.get(f, f) for f in top_3_raw]

    current_market = market_context
    COST_OF_GOODS = current_market.get('cogs', 40.0)

    min_allowed_price = COST_OF_GOODS + current_market['freight_price']
    max_allowed_price = current_market['comp_1'] * 1.20

    test_prices = np.arange(min_allowed_price, max_allowed_price, 1)

    best_price = 0
    max_profit = 0
    expected_sales = 0

    for price in test_prices:
        scenario = current_market.copy()
        scenario['unit_price'] = price
        scenario_df = pd.DataFrame([scenario])[features]
        # Predict volume and block negative demand
        predicted_qty = max(0.0, float(model.predict(scenario_df)[0]))
        
        # Commercial Guardrail: Demand destruction for massive premiums
        if price > current_market['comp_1']:
            premium_pct = (price - current_market['comp_1']) / current_market['comp_1']
            # For every 1% above the market leader, destroy 2% of the predicted demand
            elasticity_penalty = 1.0 - (premium_pct * 2.0) 
            predicted_qty = predicted_qty * max(0.1, elasticity_penalty) # Floor demand drop at 90%
            
        # Calculate true commercial margin
        profit = (price - COST_OF_GOODS - current_market['freight_price']) * predicted_qty
        
        if profit > max_profit:
            max_profit = float(profit)
            best_price = float(price)
            expected_sales = float(predicted_qty)
            
    if best_price == 0:
        return {
            "error": "Simulation Failed: No profitable price exists above the break-even floor for this category.",
            "key_drivers": key_drivers
        }

    gross_margin_pct = ((best_price - COST_OF_GOODS - current_market['freight_price']) / best_price) * 100 if best_price > 0 else 0
    comp_premium_pct = ((best_price - current_market['comp_1']) / current_market['comp_1']) * 100 if current_market['comp_1'] > 0 else 0

    return {
        "optimal_price": float(best_price),
        "expected_volume": float(expected_sales),
        "projected_profit": float(max_profit),
        "gross_margin_pct": round(float(gross_margin_pct), 1),
        "comp_premium_pct": round(float(comp_premium_pct), 1),
        "primary_competitor_price": float(current_market['comp_1']),
        "key_drivers": key_drivers
    }

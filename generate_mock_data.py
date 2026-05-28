import pandas as pd
import numpy as np
import random
import os

def generate_mock_data():
    np.random.seed(42)
    random.seed(42)

    num_rows = 600
    categories = ["bed_bath_table", "health_beauty", "computers_accessories"]
    
    data = []
    
    for i in range(num_rows):
        cat = random.choice(categories)
        month = random.randint(1, 12)
        year = random.choice([2017, 2018])
        
        # Product assignments
        if cat == "bed_bath_table":
            product_id = f"bed_{random.randint(1, 10)}"
            base_weight = np.random.uniform(300, 1500)
            base_price = np.random.uniform(30.0, 80.0)
            comp1 = base_price * np.random.uniform(0.9, 1.1)
            comp2 = base_price * np.random.uniform(1.1, 1.5)
            comp3 = base_price * np.random.uniform(0.7, 0.9)
        elif cat == "health_beauty":
            product_id = f"hb_{random.randint(1, 10)}"
            base_weight = np.random.uniform(50, 300)
            base_price = np.random.uniform(10.0, 40.0)
            comp1 = base_price * np.random.uniform(0.9, 1.1)
            comp2 = base_price * np.random.uniform(1.2, 1.4)
            comp3 = base_price * np.random.uniform(0.8, 0.95)
        else: # computers_accessories
            product_id = f"comp_{random.randint(1, 10)}"
            base_weight = np.random.uniform(1000, 5000)
            base_price = np.random.uniform(100.0, 500.0)
            comp1 = base_price * np.random.uniform(0.95, 1.05)
            comp2 = base_price * np.random.uniform(1.2, 1.3)
            comp3 = base_price * np.random.uniform(0.8, 0.9)
            
        # Mathematical correlation for seasonality (month 11 & 12 peak)
        if month in [11, 12]:
            s = np.random.uniform(15.0, 25.0)
            qty = int(np.random.poisson(lam=15))
            holiday = 1
        else:
            s = np.random.uniform(5.0, 12.0)
            qty = int(np.random.poisson(lam=3))
            holiday = 0
            
        if qty == 0:
            qty = 1
            
        # Freight vs weight correlation
        freight_price = (base_weight / 1000) * np.random.uniform(5.0, 10.0) + 5.0
        
        row = {
            "product_id": product_id,
            "product_category_name": cat,
            "month_year": f"01-{month:02d}-{year}",
            "qty": qty,
            "total_price": base_price * qty,
            "freight_price": round(freight_price, 2),
            "unit_price": round(base_price, 2),
            "product_name_lenght": random.randint(20, 60),
            "product_description_lenght": random.randint(100, 500),
            "product_photos_qty": random.randint(1, 5),
            "product_weight_g": round(base_weight, 2),
            "product_score": round(np.random.uniform(3.0, 5.0), 1),
            "customers": random.randint(10, 200),
            "weekday": random.randint(5, 25),
            "weekend": random.randint(2, 10),
            "holiday": holiday,
            "month": month,
            "year": year,
            "s": round(s, 4),
            "volume": random.randint(3000, 5000),
            "comp_1": round(comp1, 2),
            "ps1": round(np.random.uniform(3.5, 5.0), 1),
            "fp1": round(freight_price * np.random.uniform(0.9, 1.1), 2),
            "comp_2": round(comp2, 2),
            "ps2": round(np.random.uniform(4.0, 5.0), 1),
            "fp2": round(freight_price * np.random.uniform(0.8, 1.2), 2),
            "comp_3": round(comp3, 2),
            "ps3": round(np.random.uniform(3.0, 4.5), 1),
            "fp3": round(freight_price * np.random.uniform(0.9, 1.1), 2),
            "lag_price": round(base_price * np.random.uniform(0.95, 1.05), 2)
        }
        data.append(row)

    df = pd.DataFrame(data)
    
    # Ensure exact column order matching the required schema
    columns = [
      "product_id", "product_category_name", "month_year", "qty", 
      "total_price", "freight_price", "unit_price", "product_name_lenght", 
      "product_description_lenght", "product_photos_qty", "product_weight_g", 
      "product_score", "customers", "weekday", "weekend", "holiday", 
      "month", "year", "s", "volume", "comp_1", "ps1", "fp1", 
      "comp_2", "ps2", "fp2", "comp_3", "ps3", "fp3", "lag_price"
    ]
    df = df[columns]
    
    output_path = os.path.join(os.path.dirname(__file__), "retail_price.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {num_rows} rows of Category-Level synthetic data at {output_path}")

if __name__ == "__main__":
    generate_mock_data()

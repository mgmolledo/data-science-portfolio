"""
Retail Data Generator - Professional Quality Dataset
Creates realistic retail data for comprehensive analytics demonstration
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json

class RetailDataGenerator:
    """
    Professional retail data generator for analytics demonstration
    Creates realistic datasets with proper business logic and relationships
    """
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Business parameters
        self.start_date = datetime(2022, 1, 1)
        self.end_date = datetime(2024, 12, 31)
        self.num_customers = 50000
        self.num_products = 1000
        self.num_transactions = 500000
        
        # Product categories and pricing
        self.categories = {
            'Electronics': {'base_price': 200, 'price_range': 0.8, 'seasonality': 0.3},
            'Clothing': {'base_price': 50, 'price_range': 0.6, 'seasonality': 0.5},
            'Home & Garden': {'base_price': 100, 'price_range': 0.7, 'seasonality': 0.4},
            'Sports': {'base_price': 80, 'price_range': 0.5, 'seasonality': 0.6},
            'Books': {'base_price': 20, 'price_range': 0.3, 'seasonality': 0.2},
            'Beauty': {'base_price': 30, 'price_range': 0.4, 'seasonality': 0.3},
            'Food & Beverage': {'base_price': 15, 'price_range': 0.2, 'seasonality': 0.1},
            'Toys': {'base_price': 40, 'price_range': 0.6, 'seasonality': 0.8}
        }
        
        # Customer segments
        self.customer_segments = {
            'Premium': {'size': 0.15, 'avg_order_value': 200, 'frequency': 0.8},
            'Regular': {'size': 0.35, 'avg_order_value': 80, 'frequency': 0.6},
            'Budget': {'size': 0.30, 'avg_order_value': 40, 'frequency': 0.4},
            'Occasional': {'size': 0.20, 'avg_order_value': 60, 'frequency': 0.2}
        }
        
        # Channels
        self.channels = ['Online', 'Physical Store', 'Mobile App', 'Phone']
        
    def generate_customers(self):
        """Generate customer dataset with realistic attributes"""
        print("👥 Generating customer dataset...")
        
        customers = []
        segment_names = list(self.customer_segments.keys())
        
        for i in range(self.num_customers):
            # Assign customer segment
            segment = np.random.choice(segment_names, p=[self.customer_segments[s]['size'] for s in segment_names])
            
            # Generate customer attributes
            customer = {
                'customer_id': f'CUST_{i+1:06d}',
                'age': np.random.normal(35, 12),
                'gender': np.random.choice(['Male', 'Female'], p=[0.48, 0.52]),
                'income': self._generate_income(segment),
                'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']),
                'state': np.random.choice(['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'TX', 'CA', 'TX', 'CA']),
                'customer_segment': segment,
                'registration_date': self._random_date(self.start_date, self.end_date - timedelta(days=365)),
                'loyalty_tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], p=[0.4, 0.3, 0.2, 0.1]),
                'preferred_channel': np.random.choice(self.channels, p=[0.4, 0.3, 0.2, 0.1])
            }
            
            # Ensure age is realistic
            customer['age'] = max(18, min(80, int(customer['age'])))
            
            customers.append(customer)
        
        df_customers = pd.DataFrame(customers)
        print(f"✅ Generated {len(df_customers)} customers")
        return df_customers
    
    def generate_products(self):
        """Generate product dataset with realistic attributes"""
        print("📦 Generating product dataset...")
        
        products = []
        category_names = list(self.categories.keys())
        
        for i in range(self.num_products):
            category = np.random.choice(category_names)
            category_info = self.categories[category]
            
            # Generate product attributes
            base_price = category_info['base_price']
            price_range = category_info['price_range']
            
            product = {
                'product_id': f'PROD_{i+1:06d}',
                'product_name': f'{category} Product {i+1}',
                'category': category,
                'subcategory': self._generate_subcategory(category),
                'brand': np.random.choice(['Brand A', 'Brand B', 'Brand C', 'Brand D', 'Brand E']),
                'price': round(np.random.uniform(base_price * (1 - price_range), base_price * (1 + price_range)), 2),
                'cost': round(np.random.uniform(base_price * 0.3, base_price * 0.7), 2),
                'launch_date': self._random_date(self.start_date - timedelta(days=730), self.end_date),
                'is_active': np.random.choice([True, False], p=[0.85, 0.15]),
                'rating': round(np.random.uniform(3.0, 5.0), 1),
                'inventory_level': np.random.randint(0, 1000)
            }
            
            products.append(product)
        
        df_products = pd.DataFrame(products)
        print(f"✅ Generated {len(df_products)} products")
        return df_products
    
    def generate_transactions(self, customers_df, products_df):
        """Generate transaction dataset with realistic patterns"""
        print("💳 Generating transaction dataset...")
        
        transactions = []
        
        # Create date range
        date_range = pd.date_range(self.start_date, self.end_date, freq='D')
        
        for i in range(self.num_transactions):
            # Select random customer and product
            customer = customers_df.sample(1).iloc[0]
            product = products_df.sample(1).iloc[0]
            
            # Generate transaction date (more recent dates more likely)
            transaction_date = np.random.choice(date_range)
            
            # Calculate quantity based on customer segment and product category
            segment_info = self.customer_segments[customer['customer_segment']]
            base_quantity = 1
            
            # Premium customers buy more
            if customer['customer_segment'] == 'Premium':
                base_quantity = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            elif customer['customer_segment'] == 'Budget':
                base_quantity = np.random.choice([1, 2], p=[0.8, 0.2])
            
            quantity = base_quantity
            
            # Calculate total amount
            unit_price = product['price']
            
            # Apply discounts based on customer segment
            discount_rate = 0
            if customer['loyalty_tier'] == 'Platinum':
                discount_rate = np.random.uniform(0.1, 0.2)
            elif customer['loyalty_tier'] == 'Gold':
                discount_rate = np.random.uniform(0.05, 0.15)
            elif customer['loyalty_tier'] == 'Silver':
                discount_rate = np.random.uniform(0.02, 0.08)
            
            discounted_price = unit_price * (1 - discount_rate)
            total_amount = round(discounted_price * quantity, 2)
            
            # Select channel based on customer preference
            channel = customer['preferred_channel']
            if np.random.random() < 0.2:  # 20% chance to use different channel
                channel = np.random.choice(self.channels)
            
            transaction = {
                'transaction_id': f'TXN_{i+1:08d}',
                'customer_id': customer['customer_id'],
                'product_id': product['product_id'],
                'transaction_date': transaction_date,
                'quantity': quantity,
                'unit_price': unit_price,
                'discount_rate': round(discount_rate, 3),
                'total_amount': total_amount,
                'channel': channel,
                'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Cash', 'Bank Transfer']),
                'is_return': np.random.choice([False, True], p=[0.95, 0.05])
            }
            
            transactions.append(transaction)
        
        df_transactions = pd.DataFrame(transactions)
        print(f"✅ Generated {len(df_transactions)} transactions")
        return df_transactions
    
    def generate_promotions(self):
        """Generate promotion dataset"""
        print("🎯 Generating promotion dataset...")
        
        promotions = []
        promotion_types = ['Percentage Discount', 'Fixed Discount', 'Buy One Get One', 'Free Shipping']
        
        for i in range(100):  # 100 promotions over 3 years
            start_date = self._random_date(self.start_date, self.end_date - timedelta(days=30))
            duration = np.random.randint(1, 30)  # 1-30 days
            end_date = start_date + timedelta(days=duration)
            
            promotion = {
                'promotion_id': f'PROM_{i+1:04d}',
                'promotion_name': f'Promotion {i+1}',
                'promotion_type': np.random.choice(promotion_types),
                'start_date': start_date,
                'end_date': end_date,
                'discount_value': np.random.uniform(0.05, 0.5),
                'min_purchase_amount': np.random.uniform(25, 100),
                'target_customer_segment': np.random.choice(['All', 'Premium', 'Regular', 'Budget']),
                'is_active': end_date > datetime.now()
            }
            
            promotions.append(promotion)
        
        df_promotions = pd.DataFrame(promotions)
        print(f"✅ Generated {len(df_promotions)} promotions")
        return df_promotions
    
    def _generate_income(self, segment):
        """Generate income based on customer segment"""
        if segment == 'Premium':
            return np.random.normal(80000, 20000)
        elif segment == 'Regular':
            return np.random.normal(50000, 15000)
        elif segment == 'Budget':
            return np.random.normal(30000, 10000)
        else:  # Occasional
            return np.random.normal(40000, 12000)
    
    def _generate_subcategory(self, category):
        """Generate subcategory based on main category"""
        subcategories = {
            'Electronics': ['Smartphones', 'Laptops', 'Tablets', 'Audio', 'Cameras'],
            'Clothing': ['Men\'s', 'Women\'s', 'Kids', 'Accessories', 'Shoes'],
            'Home & Garden': ['Furniture', 'Decor', 'Kitchen', 'Garden', 'Tools'],
            'Sports': ['Fitness', 'Outdoor', 'Team Sports', 'Water Sports', 'Winter Sports'],
            'Books': ['Fiction', 'Non-Fiction', 'Children\'s', 'Educational', 'Reference'],
            'Beauty': ['Skincare', 'Makeup', 'Hair Care', 'Fragrance', 'Tools'],
            'Food & Beverage': ['Snacks', 'Beverages', 'Organic', 'International', 'Health'],
            'Toys': ['Action Figures', 'Board Games', 'Educational', 'Outdoor', 'Electronic']
        }
        return np.random.choice(subcategories[category])
    
    def _random_date(self, start, end):
        """Generate random date between start and end"""
        delta = end - start
        random_days = random.randint(0, delta.days)
        return start + timedelta(days=random_days)
    
    def generate_all_data(self):
        """Generate complete retail dataset"""
        print("🚀 Starting comprehensive retail data generation...")
        print("=" * 60)
        
        # Generate all datasets
        customers = self.generate_customers()
        products = self.generate_products()
        transactions = self.generate_transactions(customers, products)
        promotions = self.generate_promotions()
        
        # Save datasets
        customers.to_csv('data/raw/customers.csv', index=False)
        products.to_csv('data/raw/products.csv', index=False)
        transactions.to_csv('data/raw/transactions.csv', index=False)
        promotions.to_csv('data/raw/promotions.csv', index=False)
        
        # Generate data summary
        summary = {
            'generation_date': datetime.now().isoformat(),
            'datasets': {
                'customers': len(customers),
                'products': len(products),
                'transactions': len(transactions),
                'promotions': len(promotions)
            },
            'date_range': {
                'start': self.start_date.isoformat(),
                'end': self.end_date.isoformat()
            },
            'business_metrics': {
                'total_revenue': transactions['total_amount'].sum(),
                'avg_transaction_value': transactions['total_amount'].mean(),
                'unique_customers': transactions['customer_id'].nunique(),
                'unique_products': transactions['product_id'].nunique()
            }
        }
        
        with open('data/raw/data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n📊 Dataset Summary:")
        print(f"   Customers: {len(customers):,}")
        print(f"   Products: {len(products):,}")
        print(f"   Transactions: {len(transactions):,}")
        print(f"   Promotions: {len(promotions):,}")
        print(f"   Total Revenue: ${transactions['total_amount'].sum():,.2f}")
        print(f"   Avg Transaction: ${transactions['total_amount'].mean():.2f}")
        
        print("\n✅ All datasets generated successfully!")
        print("=" * 60)
        
        return customers, products, transactions, promotions

if __name__ == "__main__":
    # Create data directory
    import os
    os.makedirs('data/raw', exist_ok=True)
    
    # Generate data
    generator = RetailDataGenerator()
    customers, products, transactions, promotions = generator.generate_all_data()

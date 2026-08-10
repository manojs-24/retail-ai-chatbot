from sqlalchemy import (
    Boolean,
    Column,
    Date,
    Float,
    Integer,
    String,
    Text,
)

from backend.core.database import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True)

    full_name = Column(String)
    email = Column(String, unique=True, nullable=False)
    password = Column(String)

    role = Column(String)

    phone = Column(String)
    gender = Column(String)
    age = Column(Integer)

    city = Column(String)
    state = Column(String)

    join_date = Column(Date)

    total_orders = Column(Integer)
    total_spent = Column(Float)

    preferred_category = Column(String)
    loyalty_level = Column(String)


class Product(Base):
    __tablename__ = "products"

    product_id = Column(String, primary_key=True)

    product_name = Column(String)
    brand = Column(String)

    category = Column(String)
    sub_category = Column(String)

    description = Column(Text)

    price = Column(Float)
    cost_price = Column(Float)

    discount_percentage = Column(Float)
    final_price = Column(Float)

    stock_quantity = Column(Integer)
    reorder_level = Column(Integer)

    supplier_name = Column(String)
    sku = Column(String, unique=True)

    rating = Column(Float)
    total_reviews = Column(Integer)
    total_sold = Column(Integer)

    warranty_months = Column(Integer)

    weight = Column(Float)
    color = Column(String)

    specifications = Column(Text)

    launch_date = Column(Date)

    status = Column(String)

    image_url = Column(Text)


class Order(Base):
    __tablename__ = "orders"

    order_id = Column(String, primary_key=True)

    user_id = Column(String)

    order_date = Column(Date)

    total_amount = Column(Float)

    payment_method = Column(String)
    payment_status = Column(String)

    delivery_status = Column(String)

    shipping_address = Column(Text)

    delivery_date = Column(Date)


class OrderItem(Base):
    __tablename__ = "order_items"

    order_item_id = Column(String, primary_key=True)

    order_id = Column(String)

    product_id = Column(String)

    quantity = Column(Integer)

    unit_price = Column(Float)

    discount = Column(Float)

    subtotal = Column(Float)


class ProductReview(Base):
    __tablename__ = "product_reviews"

    review_id = Column(String, primary_key=True)

    product_id = Column(String)

    user_id = Column(String)

    rating = Column(Float)

    review_title = Column(String)

    review_text = Column(Text)

    sentiment = Column(String)

    review_date = Column(Date)

    verified_purchase = Column(Boolean)
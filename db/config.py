from pymongo import MongoClient
import os
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client["invoices_db"]
invoices_collection = mongo_db["invoices"]
products_collection = mongo_db["products"]
products_collection.create_index("ref", unique=True)

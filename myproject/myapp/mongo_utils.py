# your_app/mongo_utils.py
from pymongo.mongo_client import MongoClient
import pymongo
from django.conf import settings
from pymongo.server_api import ServerApi

def get_mongo_client():
    """Connects to MongoDB and returns the database object."""
    try:
        client = MongoClient(settings.MONGO_URI, server_api=ServerApi('1'))
        db = client[settings.MONGO_DB_NAME]  # Connects to the database
        return db
    except pymongo.errors.ConnectionError as e:
        raise Exception(f"Failed to connect to MongoDB: {str(e)}")

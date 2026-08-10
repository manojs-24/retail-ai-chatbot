from backend.core.database import Base, engine
import backend.models

print("Creating database...")

Base.metadata.create_all(bind=engine)

print("Database created successfully.")
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/traffic_violations')

# Create engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Violation(Base):
    __tablename__ = "violations"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Violation details
    violation_type = Column(String(50), nullable=False)  # e.g., Over-speed, No Helmet
    severity = Column(String(20), nullable=True)  # low, medium, high, critical
    description = Column(Text, nullable=True)  # extra notes about the violation
    
    # Vehicle details
    number_plate = Column(String(20), nullable=False)
    vehicle_type = Column(String(20), nullable=True)  # car, bike, truck, etc.
    vehicle_color = Column(String(20), nullable=True)
    
    # Speed details (if applicable)
    speed_limit = Column(Float, nullable=True)
    actual_speed = Column(Float, nullable=True)
    speed_unit = Column(String(10), default='km/h')
    
    # Location and camera
    location = Column(String(100), nullable=True)
    camera_id = Column(String(20), nullable=True)
    
    # Evidence
    image_path = Column(Text, nullable=True)
    
    # Metadata
    confidence_score = Column(Float, nullable=True)  # model's prediction confidence
    timestamp = Column(DateTime, default=func.now())



def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def init_database():
    """Initialize database with sample data"""
    db = SessionLocal()
    try:
        # Create tables
        create_tables()
        print("✓ Database initialized successfully")
        
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        db.rollback()
    finally:
        db.close()

class DatabaseManager:
    def __init__(self):
        self.db = SessionLocal()
    
    def add_violation(self, violation_data: dict):
        """Add a new violation to database"""
        try:
            violation = Violation(**violation_data)
            self.db.add(violation)
            self.db.commit()
            self.db.refresh(violation)
            return violation
        except Exception as e:
            self.db.rollback()
            print(f"Error adding violation: {e}")
            return None
    
    def get_violations(self, limit: int = 100, offset: int = 0, violation_type: str = None):
        """Get violations with optional filtering"""
        query = self.db.query(Violation)
        
        if violation_type:
            query = query.filter(Violation.violation_type == violation_type)
        
        return query.order_by(Violation.timestamp.desc()).offset(offset).limit(limit).all()
    
    def get_violation_stats(self):
        """Get violation statistics"""
        total = self.db.query(Violation).count()
        by_type = {}
        by_severity = {}
        
        for violation_type in ['helmet', 'phone', 'speed', 'red_light']:
            count = self.db.query(Violation).filter(Violation.violation_type == violation_type).count()
            by_type[violation_type] = count
        
        for severity in ['low', 'medium', 'high', 'critical']:
            count = self.db.query(Violation).filter(Violation.severity == severity).count()
            by_severity[severity] = count
        
        return {
            'total_violations': total,
            'by_type': by_type,
            'by_severity': by_severity
        }
    

    
    def close(self):
        """Close database connection"""
        self.db.close()

-- Traffic Violation Detection System - Database Setup Script
-- Run this script in PostgreSQL to create the database and tables

-- Create database (run this as superuser)
CREATE DATABASE traffic_violations;

-- Connect to the database
\c traffic_violations;

-- Create violations table
CREATE TABLE violations (
    id SERIAL PRIMARY KEY,
    
    -- Violation details
    violation_type VARCHAR(50) NOT NULL,     -- e.g., Over-speed, No Helmet
    severity VARCHAR(20),                    -- low, medium, high, critical
    description TEXT,                        -- extra notes about the violation
    
    -- Vehicle details
    number_plate VARCHAR(20) NOT NULL,
    vehicle_type VARCHAR(20),                -- car, bike, truck, etc.
    vehicle_color VARCHAR(20),
    
    -- Speed details (if applicable)
    speed_limit FLOAT,
    actual_speed FLOAT,
    speed_unit VARCHAR(10) DEFAULT 'km/h',
    
    -- Location and camera
    location VARCHAR(100),
    camera_id VARCHAR(20),
    
    -- Evidence
    image_path TEXT,
    
    -- Metadata
    confidence_score FLOAT,                  -- model's prediction confidence
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_violations_timestamp ON violations(timestamp);
CREATE INDEX idx_violations_type ON violations(violation_type);
CREATE INDEX idx_violations_plate ON violations(number_plate);
CREATE INDEX idx_violations_camera ON violations(camera_id);

-- Grant permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON DATABASE traffic_violations TO your_user;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_user;

-- Display created tables
\dt

-- Display sample data
SELECT * FROM cameras;

-- Supabase Schema for Satellite Change Detection System
-- Run this SQL in your Supabase SQL editor to set up the required tables

-- Enable the pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Analysis Results Table
-- Stores complete analysis results with metadata
CREATE TABLE IF NOT EXISTS analysis_results (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    image_hash_before TEXT NOT NULL,
    image_hash_after TEXT NOT NULL,
    change_percentage DECIMAL(5,2) NOT NULL DEFAULT 0,
    changed_pixels INTEGER NOT NULL DEFAULT 0,
    total_pixels INTEGER NOT NULL DEFAULT 0,
    contours_count INTEGER NOT NULL DEFAULT 0,
    significance_level TEXT NOT NULL CHECK (significance_level IN ('HIGH', 'MEDIUM', 'LOW', 'MINIMAL')),
    agent_analysis TEXT,
    tools_used TEXT[] DEFAULT '{}',
    processing_time_ms INTEGER DEFAULT 0,
    raw_results JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Image Embeddings Table
-- Stores vector embeddings for similarity search
CREATE TABLE IF NOT EXISTS image_embeddings (
    id TEXT PRIMARY KEY,
    image_hash TEXT UNIQUE NOT NULL,
    embedding vector(512), -- Adjust dimension based on your embedding model
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Users Table (optional - for authentication)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE,
    name TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Analysis Sessions Table (optional - for grouping analyses)
CREATE TABLE IF NOT EXISTS analysis_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    name TEXT,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_analysis_results_user_id ON analysis_results(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_significance ON analysis_results(significance_level);
CREATE INDEX IF NOT EXISTS idx_analysis_results_change_percentage ON analysis_results(change_percentage);
CREATE INDEX IF NOT EXISTS idx_analysis_results_created_at ON analysis_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_results_image_hashes ON analysis_results(image_hash_before, image_hash_after);

CREATE INDEX IF NOT EXISTS idx_image_embeddings_hash ON image_embeddings(image_hash);
CREATE INDEX IF NOT EXISTS idx_image_embeddings_created_at ON image_embeddings(created_at DESC);

-- Vector similarity index for embeddings
CREATE INDEX IF NOT EXISTS idx_image_embeddings_vector ON image_embeddings USING ivfflat (embedding vector_cosine_ops);

-- Triggers to update updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_analysis_results_updated_at 
    BEFORE UPDATE ON analysis_results 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_image_embeddings_updated_at 
    BEFORE UPDATE ON image_embeddings 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_analysis_sessions_updated_at 
    BEFORE UPDATE ON analysis_sessions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Row Level Security (RLS) policies
ALTER TABLE analysis_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE image_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_sessions ENABLE ROW LEVEL SECURITY;

-- Allow anonymous access for demo purposes (adjust as needed)
CREATE POLICY "Allow anonymous read access" ON analysis_results FOR SELECT USING (true);
CREATE POLICY "Allow anonymous insert access" ON analysis_results FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow anonymous read access" ON image_embeddings FOR SELECT USING (true);
CREATE POLICY "Allow anonymous insert access" ON image_embeddings FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow anonymous update access" ON image_embeddings FOR UPDATE USING (true);

-- Sample views for analytics
CREATE VIEW IF NOT EXISTS analysis_summary AS
SELECT 
    significance_level,
    COUNT(*) as count,
    AVG(change_percentage) as avg_change_percentage,
    AVG(processing_time_ms) as avg_processing_time,
    MAX(created_at) as latest_analysis
FROM analysis_results 
GROUP BY significance_level;

CREATE VIEW IF NOT EXISTS daily_analysis_count AS
SELECT 
    DATE(created_at) as analysis_date,
    COUNT(*) as daily_count,
    AVG(change_percentage) as avg_change_percentage
FROM analysis_results 
GROUP BY DATE(created_at)
ORDER BY analysis_date DESC;

-- Functions for advanced queries
CREATE OR REPLACE FUNCTION find_similar_images(
    target_embedding vector(512),
    similarity_threshold float DEFAULT 0.8,
    max_results int DEFAULT 10
)
RETURNS TABLE (
    image_hash text,
    similarity_score float,
    metadata jsonb
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ie.image_hash,
        1 - (ie.embedding <=> target_embedding) as similarity,
        ie.metadata
    FROM image_embeddings ie
    WHERE 1 - (ie.embedding <=> target_embedding) > similarity_threshold
    ORDER BY ie.embedding <=> target_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Sample data for testing (optional)
-- INSERT INTO analysis_results (
--     id, image_hash_before, image_hash_after, 
--     change_percentage, changed_pixels, total_pixels, contours_count,
--     significance_level, agent_analysis, tools_used
-- ) VALUES (
--     'sample_001', 'hash_before_001', 'hash_after_001',
--     12.5, 125000, 1000000, 15,
--     'MEDIUM', 'Sample analysis showing moderate infrastructure changes.',
--     ARRAY['detect_image_changes', 'analyze_images_with_gpt4_vision', 'assess_change_significance']
-- );

COMMENT ON TABLE analysis_results IS 'Stores complete satellite image change analysis results';
COMMENT ON TABLE image_embeddings IS 'Stores vector embeddings for image similarity search';
COMMENT ON TABLE users IS 'User accounts for the satellite analysis system';
COMMENT ON TABLE analysis_sessions IS 'Grouping of related analyses into sessions'; 
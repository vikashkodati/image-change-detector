# Supabase Setup Guide for Satellite Change Detection

This guide will help you set up Supabase to add database functionality, user authentication, and vector storage to your satellite change detection system.

## 🚀 **Quick Setup**

### 1. Create Supabase Project

1. Go to [supabase.com](https://supabase.com) and create an account
2. Create a new project
3. Choose a region close to your users
4. Wait for the project to be ready (~2 minutes)

### 2. Get Your Credentials

From your Supabase dashboard:

1. Go to **Settings** → **API**
2. Copy your:
   - **Project URL** (looks like: `https://xxxxx.supabase.co`)
   - **anon public key** (starts with `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9`)

### 3. Set Environment Variables

Add these to your Railway environment variables (or local `.env` file):

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
```

### 4. Run Database Schema

1. In your Supabase dashboard, go to **SQL Editor**
2. Copy the contents of `backend/supabase_schema.sql`
3. Paste and run the SQL to create tables and indexes

## 📊 **Database Schema Overview**

### Core Tables

**`analysis_results`** - Stores complete analysis results
- `id` - Unique analysis identifier
- `user_id` - Optional user identifier
- `image_hash_before/after` - Unique image identifiers
- `change_percentage` - Percentage of image that changed
- `significance_level` - HIGH/MEDIUM/LOW/MINIMAL
- `agent_analysis` - AI agent's comprehensive analysis
- `tools_used` - Array of tools used in analysis
- `raw_results` - Complete JSON results

**`image_embeddings`** - Vector embeddings for similarity search
- `image_hash` - Links to images in analyses
- `embedding` - 512-dimensional vector
- `metadata` - Additional image metadata

**`users`** (optional) - User accounts
**`analysis_sessions`** (optional) - Grouping analyses

### Vector Search Capabilities

The schema includes **pgvector** support for:
- Image similarity search
- Finding analyses of similar images
- Clustering similar change patterns

## 🔧 **Features Enabled**

### 1. Analysis History
Every analysis is automatically stored with:
- Complete results and metadata
- Processing performance metrics
- Tool usage tracking
- Significance assessment

### 2. Similar Analysis Search
The agent can find previous analyses with:
- Similar change percentages
- Same significance levels
- Historical context for better insights

### 3. System Analytics
Track system-wide statistics:
- Total analyses performed
- Distribution by significance level
- Average processing times
- Usage patterns

### 4. Vector Storage (Future)
Ready for advanced features:
- Image embedding storage
- Semantic similarity search
- Pattern recognition across time

## 🤖 **Agent Enhancement**

With Supabase enabled, the AI agent now:

1. **Stores Results**: Every analysis is saved for future reference
2. **Finds Context**: Searches for similar historical analyses
3. **Learns Patterns**: Can reference past similar situations
4. **Provides Continuity**: Builds institutional knowledge over time

### Enhanced Workflow

```
1. detect_image_changes (OpenCV analysis)
2. analyze_images_with_gpt4_vision (AI semantic analysis)  
3. assess_change_significance (Significance assessment)
4. find_similar_analyses_tool (Historical context)
5. store_analysis_result_tool (Save for future)
6. Comprehensive analysis with historical insights
```

## 🔐 **Security Configuration**

### Row Level Security (RLS)

The schema includes RLS policies for:
- Anonymous access (demo mode)
- User-specific data isolation (when authentication added)
- Read/write permissions control

### Authentication (Optional)

To add user authentication:

1. Enable **Authentication** in Supabase dashboard
2. Configure providers (email, Google, GitHub, etc.)
3. Update RLS policies for user-specific access
4. Add user context to frontend

## 📈 **Database Monitoring**

### Built-in Views

**`analysis_summary`** - Statistics by significance level
```sql
SELECT * FROM analysis_summary;
```

**`daily_analysis_count`** - Daily usage metrics
```sql
SELECT * FROM daily_analysis_count LIMIT 30;
```

### Custom Queries

Find analyses with high change percentages:
```sql
SELECT * FROM analysis_results 
WHERE change_percentage > 10 
ORDER BY created_at DESC;
```

Get agent analysis trends:
```sql
SELECT 
    DATE(created_at) as date,
    COUNT(*) as count,
    AVG(LENGTH(agent_analysis)) as avg_analysis_length
FROM analysis_results 
WHERE agent_analysis IS NOT NULL
GROUP BY DATE(created_at);
```

## 🚧 **Troubleshooting**

### Common Issues

**1. Connection Failed**
- Check `SUPABASE_URL` and `SUPABASE_ANON_KEY`
- Verify project is active in Supabase dashboard

**2. Table Access Denied**
- Run the schema SQL in Supabase SQL Editor
- Check RLS policies allow anonymous access

**3. Vector Extension Missing**
- Ensure `CREATE EXTENSION vector;` ran successfully
- Contact Supabase support if needed

### Health Check

Test your setup:
```bash
curl https://your-railway-app.railway.app/api/health
```

Look for:
```json
{
  "supabase_status": {
    "status": "healthy",
    "connected": true,
    "tables_accessible": true
  },
  "database_features": true
}
```

## 🔮 **Future Enhancements**

With Supabase foundation, you can add:

1. **User Dashboards** - Personal analysis history
2. **Team Collaboration** - Shared analysis sessions  
3. **API Keys** - Rate limiting and usage tracking
4. **Webhooks** - Real-time notifications
5. **Advanced Analytics** - ML pattern detection
6. **Export Features** - PDF reports, data exports

## 📞 **Support**

- **Supabase Docs**: [docs.supabase.com](https://docs.supabase.com)
- **Vector Extension**: [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector)
- **SQL Reference**: [postgresql.org/docs](https://www.postgresql.org/docs/)

---

🎉 **Congratulations!** Your satellite change detection system now has enterprise-grade database capabilities with vector search, analytics, and scalable storage. 
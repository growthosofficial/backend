# 🚀 Setup Guide

## Prerequisites

You need:
1. **Azure OpenAI API** credentials
2. **Supabase** database credentials
3. **Python 3.8+**

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Create Environment File

Create a `.env` file in the root directory with:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key_here

# Optional Settings
DEFAULT_SIMILARITY_THRESHOLD=0.8
DEFAULT_LLM_TYPE=azure_openai
LOG_LEVEL=INFO
```

## Step 3: Get Your Credentials

### Azure OpenAI
1. Go to [Azure Portal](https://portal.azure.com)
2. Create or access your Azure OpenAI resource
3. Get your API key from "Keys and Endpoint"
4. Note your endpoint URL

### Supabase
1. Go to [Supabase](https://supabase.com)
2. Create or access your project
3. Go to Settings → API
4. Copy your project URL and anon key

## Step 4: Run the Application

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Note**: On first startup, the system will automatically:
- Pre-compute embeddings for all 50+ academic categories
- Save them to `main_category_embeddings.pkl`
- Make subsequent user interactions fast (2-3s vs 10-15s)

This only happens once - subsequent startups will be fast.

## Step 5: Test the System

1. **Health Check**: Visit `http://localhost:8000/health`
2. **API Docs**: Visit `http://localhost:8000/docs`
3. **Test Processing**: Use the `/api/process-text` endpoint

## Troubleshooting

### Common Issues:

1. **"AZURE_OPENAI_API_KEY must be set"**
   - Check your `.env` file exists and has the correct API key

2. **"SUPABASE_URL must be set"**
   - Verify your Supabase credentials in `.env`

3. **Import errors**
   - Make sure you're in the correct directory
   - Run `pip install -r requirements.txt`

4. **First startup is slow**
   - This is normal - the system is computing embeddings for all categories
   - Subsequent startups will be fast

### Performance Tips:

- **First Startup**: Will be slower due to embedding computation (~30-60 seconds)
- **Subsequent Starts**: Fast startup with cached embeddings
- **Fallback Mode**: System works even if embedding computation fails
- **Production**: Embeddings are cached and persist across deployments

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /api/process-text` - Main AI processing endpoint
- `GET /api/categories` - Get all categories
- `GET /api/stats` - Database statistics
- `GET /docs` - Interactive API documentation

## Next Steps

1. Test the system with sample text
2. Integrate with your frontend
3. Monitor performance and adjust settings as needed 
# Image Change Detector

A satellite image change detection web application built with Next.js and Python.

Read claude.md to understand current status of project. 

## Features

- Upload before/after satellite images
- Highlight pixel differences 
- AI-powered change descriptions
- Interactive Q&A about changes
- Text-to-speech audio responses

## Architecture

- **Frontend**: Next.js 14 + shadcn/ui + Tailwind CSS
- **Backend**: FastMCP Python server
- **Storage**: Supabase (Auth + Storage + pgvector)
- **AI**: GPT-4 Vision + OpenAI TTS + ElevenLabs TTS
- **Change Detection**: pixelmatch + OpenCV-Python

## Development

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Backend
```bash
cd backend
uv sync
uv run python src/server.py
```

## Project Structure

```
image-change-detector/
├── frontend/          # Next.js 14 app
├── backend/           # FastMCP Python server
├── README.md
└── .gitignore
```
# Helper services to scrape data
Transcribes YouTube videos and extracts text from PDFs

## Endpoints
Transcribe YouTube: ```http://localhost:9500/transcribeYouTube?youtubeURL=<YOUTUBE_VIDEO_URL>```

Extract text from PDF: ```http://localhost:9500/scrapePDF?pdfURL=<PDF_URL>```

## Running the Service
1. docker build .
2. docker run -p 9500:80 <hash>

TODO: fix worker transcription endpoint
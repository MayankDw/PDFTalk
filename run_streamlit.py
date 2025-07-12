#!/usr/bin/env python3
"""
PDFTalk Streamlit Launcher
Run this script to start the PDFTalk web application
"""

import subprocess
import sys
import webbrowser
import time

def main():
    print("🚀 Starting PDFTalk - AI PDF Question Answering App")
    print("=" * 50)
    
    try:
        # Start streamlit
        print("📄 Launching Streamlit app...")
        subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py", 
            "--server.port", "8080",
            "--server.headless", "true"
        ])
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Open browser
        url = "http://localhost:8080"
        print(f"🌐 Opening browser at {url}")
        webbrowser.open(url)
        
        print("\n✅ PDFTalk is now running!")
        print("📖 How to use:")
        print("   1. Upload a PDF file using the sidebar")
        print("   2. Click 'Process PDF' and wait for completion")
        print("   3. Ask questions about the PDF content")
        print("   4. Get AI-powered answers with confidence scores")
        print("\n🛑 Press Ctrl+C to stop the server")
        
        # Keep the script running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Shutting down PDFTalk...")
            
    except Exception as e:
        print(f"❌ Error starting PDFTalk: {e}")
        print("💡 Make sure you have installed the requirements:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
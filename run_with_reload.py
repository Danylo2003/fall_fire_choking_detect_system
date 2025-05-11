import time
import os
import sys
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class RestartHandler(FileSystemEventHandler):
    def __init__(self, app_file):
        self.app_file = app_file
        self.process = None
        self.start_app()
        
    def start_app(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
        print("Starting Gradio app...")
        self.process = subprocess.Popen([sys.executable, self.app_file])
        
    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            print(f"Change detected in {event.src_path}")
            self.start_app()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_with_reload.py your_gradio_app.py")
        sys.exit(1)
        
    app_file = sys.argv[1]
    event_handler = RestartHandler(app_file)
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        if event_handler.process:
            event_handler.process.terminate()
    observer.join()
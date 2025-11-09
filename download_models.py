"""
YOLO Model Downloader
Automatically downloads required YOLOv3 model files
"""

import os
import urllib.request
import sys
import time

class ModelDownloader:
    def __init__(self):
        self.models_dir = "models"
        self.files = {
            "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
            "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", 
            "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights"
        }
        
    def download_file(self, url, filename):
        """Download file with progress indicator"""
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, int(count * block_size * 100 / total_size))
                sys.stdout.write(f"\rDownloading {filename}: {percent}% [{count * block_size / (1024*1024):.1f}MB/{total_size/(1024*1024):.1f}MB]")
                sys.stdout.flush()
        
        try:
            urllib.request.urlretrieve(url, filename, progress_hook)
            print(f"\n✅ {filename} downloaded successfully!")
            return True
        except Exception as e:
            print(f"\n❌ Error downloading {filename}: {e}")
            return False
    
    def check_existing_files(self):
        """Check which files already exist"""
        existing = []
        missing = []
        
        for filename in self.files.keys():
            filepath = os.path.join(self.models_dir, filename)
            if os.path.exists(filepath):
                existing.append(filename)
            else:
                missing.append(filename)
                
        return existing, missing
    
    def download_all(self):
        """Download all missing model files"""
        print("🔍 Checking for YOLO model files...")
        
        # Create models directory
        os.makedirs(self.models_dir, exist_ok=True)
        
        existing, missing = self.check_existing_files()
        
        if existing:
            print("✅ Existing files found:")
            for file in existing:
                print(f"   - {file}")
        
        if missing:
            print("📥 Missing files to download:")
            for file in missing:
                print(f"   - {file}")
            
            print(f"\n🚀 Starting download of {len(missing)} files...")
            print("💡 Note: yolov3.weights is ~250MB and may take several minutes.\n")
            
            success_count = 0
            for filename in missing:
                filepath = os.path.join(self.models_dir, filename)
                url = self.files[filename]
                
                print(f"\n📥 Downloading {filename}...")
                if self.download_file(url, filepath):
                    success_count += 1
                time.sleep(1)  # Brief pause between downloads
            
            print(f"\n🎉 Download complete: {success_count}/{len(missing)} files successfully downloaded.")
            
            if success_count == len(missing):
                print("✅ All model files are ready! You can now run the object detection system.")
                return True
            else:
                print("❌ Some files failed to download. Please check your internet connection and try again.")
                return False
        else:
            print("✅ All model files are already downloaded and ready!")
            return True

def main():
    """Main function for downloading models"""
    print("=" * 60)
    print("        YOLOv3 Model Downloader")
    print("=" * 60)
    
    downloader = ModelDownloader()
    success = downloader.download_all()
    
    if success:
        print("\n🎊 Setup complete! Run 'python main.py' to start object detection.")
    else:
        print("\n💡 Troubleshooting tips:")
        print("   - Check your internet connection")
        print("   - Try running the script again")
        print("   - Download files manually from:")
        print("     https://pjreddie.com/darknet/yolo/")

if __name__ == "__main__":
    main()

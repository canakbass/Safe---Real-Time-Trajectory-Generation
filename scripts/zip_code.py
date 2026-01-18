
import zipfile
import os

def zip_code():
    targets = ['src', 'scripts', 'requirements.txt']
    with zipfile.ZipFile('space_code_v3.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for target in targets:
            if os.path.isfile(target):
                z.write(target)
            elif os.path.isdir(target):
                for root, dirs, files in os.walk(target):
                    # Skip __pycache__
                    if '__pycache__' in root:
                        continue
                    for file in files:
                        if file.endswith('.pyc'): continue
                        z.write(os.path.join(root, file))
    print("Created space_code_v3.zip successfully.")

if __name__ == "__main__":
    zip_code()

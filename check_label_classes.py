import os

label_dir = r"C:\Users\user\Desktop\image\ignition_dataset\labels"

for root, _, files in os.walk(label_dir):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip() == "":
                        continue
                    class_id = int(line.split()[0])
                    if class_id > 0:
                        print(f"[!] {file_path} → 문제 라벨: {line.strip()}")

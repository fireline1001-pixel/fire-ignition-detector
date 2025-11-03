import os

label_dir = r"C:\Users\user\Desktop\image\ignition_dataset\labels"

for root, _, files in os.walk(label_dir):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            fixed_lines = []
            has_invalid = False
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        class_id = int(parts[0])
                        if class_id > 0:
                            parts[0] = "0"
                            has_invalid = True
                        fixed_lines.append(" ".join(parts) + "\n")
                    except ValueError:
                        print(f"[!] 문자열 라벨 제거됨: {file_path} → {line.strip()}")
                        has_invalid = True
            if has_invalid:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(fixed_lines)
                print(f"[수정 완료] {file_path}")

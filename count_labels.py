import os
import glob
import xml.etree.ElementTree as ET
from collections import Counter

xml_folder_path = './data/annotations' 

def analyze_dataset_labels(folder_path):
    xml_files = glob.glob(os.path.join(folder_path, '*.xml'))
    
    if not xml_files:
        print(f"không tìm thấy file xml trong '{folder_path}'")
        return

    print(f"Đang phân tích {len(xml_files)} file XML...")
    
    label_counter = Counter()
    total_objects = 0
    
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                name = obj.find('name').text
                label_counter[name] += 1
                total_objects += 1
                
        except Exception as e:
            print(f"Lỗi khi xử lý file {xml_file}: {e}")
            continue

    print("\n" + "="*40)
    print(f"TỔNG SỐ ĐỐI TƯỢNG (BOXES): {total_objects}")
    print("="*40)
    print(f"{'Tên Label':<25} | {'Số lượng':<10} | {'Tỉ lệ %':<10}")
    print("-" * 50)
    
    for label, count in label_counter.most_common():
        percentage = (count / total_objects) * 100
        print(f"{label:<25} | {count:<10} | {percentage:.2f}%")
    print("="*40)

    # Cảnh báo nếu dữ liệu mất cân bằng
    if len(label_counter) > 0:
        counts = list(label_counter.values())
        max_c = max(counts)
        min_c = min(counts)
        if max_c > min_c * 3: 
            print("\n⚠️ CẢNH BÁO: Dữ liệu đang bị mất cân bằng (Imbalanced Dataset).")
            print("Model sẽ có xu hướng học tốt lớp nhiều ảnh hơn và bỏ qua lớp ít ảnh.")

analyze_dataset_labels(xml_folder_path)
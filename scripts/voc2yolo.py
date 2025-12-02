import os
import xml.etree.ElementTree as ET

CLASSES = ["with_mask", "without_mask"]

img_dir = "./data/images"
ann_dir = "./data/annotations"
output_dir = "./dataset/yolo_annotations"
os.makedirs(output_dir, exist_ok=True)

def convert_bbox(size, box):
    w, h = size
    xmin, ymin, xmax, ymax = box

    # YOLO expects normalized values
    xc = (xmin + xmax) / 2.0 / w
    yc = (ymin + ymax) / 2.0 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h

    return xc, yc, bw, bh

for file in os.listdir(ann_dir):
    if file.endswith(".xml"):
        xml_path = os.path.join(ann_dir, file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        lines = []

        for obj in root.findall("object"):
            cls = obj.find("name").text
            if cls not in CLASSES:
                continue
            cls_id = CLASSES.index(cls)

            xml_box = obj.find("bndbox")
            xmin = float(xml_box.find("xmin").text)
            ymin = float(xml_box.find("ymin").text)
            xmax = float(xml_box.find("xmax").text)
            ymax = float(xml_box.find("ymax").text)

            x, y, bw, bh = convert_bbox((w, h), (xmin, ymin, xmax, ymax))

            # ensure values are clipped between 0-1
            x = max(0, min(1, x))
            y = max(0, min(1, y))
            bw = max(0, min(1, bw))
            bh = max(0, min(1, bh))

            lines.append(f"{cls_id} {x} {y} {bw} {bh}")

        out_file = file.replace(".xml", ".txt")
        with open(os.path.join(output_dir, out_file), "w") as f:
            f.write("\n".join(lines))

print("DONE: Converted VOC XML → YOLO TXT.")
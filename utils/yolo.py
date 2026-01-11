import cv2
import tempfile
import numpy as np

def is_box_inside(inner_box, outer_box, tolerance=0.9):
    
    x1_inner, y1_inner, x2_inner, y2_inner = inner_box
    x1_outer, y1_outer, x2_outer, y2_outer = outer_box
    
    x1_inter = max(x1_inner, x1_outer)
    y1_inter = max(y1_inner, y1_outer)
    x2_inter = min(x2_inner, x2_outer)
    y2_inter = min(y2_inner, y2_outer)
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return False
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    inner_area = (x2_inner - x1_inner) * (y2_inner - y1_inner)
    
    if inner_area == 0:
        return False
    
    return (inter_area / inner_area) >= tolerance

def get_largest_box(boxes, confidences):
    if len(boxes) == 0:
        return None, None
    
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    max_idx = np.argmax(areas)
    return boxes[max_idx], confidences[max_idx]

def detect_object(image_path, model, position):
    img = cv2.imread(image_path)
    
    height, width = img.shape[:2]
    image_area = width * height
    
    results = model(image_path, verbose=False)
    
    class_0_boxes = []  
    class_1_boxes = []  
    class_0_confs = []
    class_1_confs = []
    
    for result in results:
        if result.boxes is not None:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                box_coords = box.cpu().numpy()
                conf_val = conf.cpu().item()
                cls_val = int(cls.cpu().item())
                
                if cls_val == 0:  # Ambing
                    class_0_boxes.append(box_coords)
                    class_0_confs.append(conf_val)
                elif cls_val == 1:  # Sapi
                    class_1_boxes.append(box_coords)
                    class_1_confs.append(conf_val)
                    
    largest_class_0, _ = get_largest_box(class_0_boxes, class_0_confs)
    largest_class_1, _ = get_largest_box(class_1_boxes, class_1_confs)
    
    ambing_to_sapi_ratio = None
    sapi_to_image_ratio = None
    
    if largest_class_0 is not None and largest_class_1 is not None:
        if is_box_inside(largest_class_0, largest_class_1):
            class_0_width = largest_class_0[2] - largest_class_0[0]
            class_1_width = largest_class_1[2] - largest_class_1[0]
            
            if class_1_width > 0:
                ambing_to_sapi_ratio = class_0_width / class_1_width
                
    if largest_class_1 is not None:
        class_1_width = largest_class_1[2] - largest_class_1[0]
        sapi_to_image_ratio = class_1_width / width
        
    if ambing_to_sapi_ratio is None:
        if position == 'back':
            ambing_to_sapi_ratio = 0.061050536996666606
        elif position == 'side':
            ambing_to_sapi_ratio = 0.036839442481699965
    
    if sapi_to_image_ratio is None:
        if position == 'back':
            ambing_to_sapi_ratio = 0.16588849097824268
        elif position == 'side':
            ambing_to_sapi_ratio = 0.07101039914607489
    
    return ambing_to_sapi_ratio, sapi_to_image_ratio
    

def get_ratio_data(uploaded_file, model, position):
    uploaded_file.seek(0)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    return detect_object(tmp_path, model, position)
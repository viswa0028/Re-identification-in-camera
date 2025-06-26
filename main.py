import numpy as np
import torch
import cv2
import faiss
from ultralytics import YOLO
from torchvision import models, transforms

# Feature extractor
resmodel = models.resnet50(pretrained=True)
resmodel.eval()
modules = list(resmodel.children())[:-1]
resmodel = torch.nn.Sequential(*modules)

# Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# YOLO model
model = YOLO('Best (1).pt')

# FAISS setup
feature_dim = 2048
index = faiss.IndexFlatIP(feature_dim)
faiss_id_to_person_id = {}
current_faiss_id = 0
current_person_id = 0
Similarity_threshold = 0.87

# Video input
cap = cv2.VideoCapture("Assignment Materials 720p.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
initialization_frames = int(fps * 5)
frame_number = 0
player_ids = {}

# Minimum bounding box area
min_area = 1000

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Skiping small boxes
            if (x2 - x1) * (y2 - y1) < min_area:
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            transformed = transform(roi).unsqueeze(0)
            with torch.no_grad():
                feature = resmodel(transformed)
                feature = feature.squeeze().numpy()
            norm_feature = feature / np.linalg.norm(feature)

            person_id = None
            if index.ntotal > 0:
                D, I = index.search(norm_feature.reshape(1, -1), k=1)
                similarity = D[0][0]
                matched_faiss_id = I[0][0]

                if similarity > Similarity_threshold:
                    person_id = faiss_id_to_person_id[matched_faiss_id]

            if person_id is None:
                if frame_number < initialization_frames:
                    index.add(norm_feature.reshape(1, -1))
                    faiss_id_to_person_id[current_faiss_id] = current_person_id
                    person_id = current_person_id
                    player_ids[current_person_id] = f"Player {current_person_id}"
                    current_faiss_id += 1
                    current_person_id += 1
                else:
                    person_id = "Unknown"

            label = player_ids.get(person_id, "Unknown") if isinstance(person_id, int) else person_id
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    frame_number += 1
    cv2.imshow("FRAME", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

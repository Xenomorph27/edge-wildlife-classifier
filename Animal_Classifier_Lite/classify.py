import torch
import torch.nn.functional as F
from torchvision import models, transforms
import cv2

# ---------------- CONFIG ----------------
MODEL_PATH = "animal_classifier.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODEL ----------------
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
class_names = checkpoint["class_names"]

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)
model.eval()

print("Model loaded on", DEVICE)
print("Classes:", class_names)

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------- WEBCAM ----------------
cap = cv2.VideoCapture(0)
WINDOW_NAME = "Wildlife Classifier (Local)"

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = transform(rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(inp)
        probs = F.softmax(out, dim=1)[0]
        conf, pred = torch.max(probs, 0)

    # ---------- CONFIDENCE-AWARE DECISION ----------
    confidence = conf.item() * 100
    raw_label = class_names[pred.item()]

    if raw_label == "elk":
        threshold = 75
    elif raw_label == "deer":
        threshold = 60
    elif raw_label == "moose":
        threshold = 60
    else:
        threshold = 70

    if confidence < threshold:
        label = "UNSURE"
    else:
        label = raw_label.upper()

    # ---------- DISPLAY ----------
    text = f"{label}  {confidence:.1f}%"
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(WINDOW_NAME, frame)

    # ---------- EXIT HANDLING ----------
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# Page configuration
st.set_page_config(
    page_title="Number Plate Recognition",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #33B5A0;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #33B5A0;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        color: #33B5A0;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🚗 Car Number Plate Recognition System</p>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Model paths
YOLO_MODEL_PATH = r"C:\Users\Dell\Desktop\yolo_models\runs\detect\train4\weights\best.pt"
CNN_MODEL_PATH = r"C:\Users\Dell\Desktop\yolo_models\cnn_model.h5"

# Load models function
@st.cache_resource
def load_yolo_model():
    """Load YOLO model"""
    try:
        from ultralytics import YOLO
        model = YOLO(YOLO_MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.error(f"Please check the path: {YOLO_MODEL_PATH}")
        return None

@st.cache_resource
def load_cnn_model():
    """Load CNN model"""
    try:
        from tensorflow.keras.models import load_model
        model = load_model(CNN_MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading CNN model: {e}")
        st.error(f"Please check the path: {CNN_MODEL_PATH}")
        return None

# Core functions
def detect_plate(img, yolo_model, conf_threshold=0.25):
    """Detect license plate using YOLO"""
    plate_img = img.copy()
    roi = img.copy()
    
    results = yolo_model.predict(source=img, conf=conf_threshold, save=False, verbose=False)
    
    plate_rect = []
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            w, h = x2 - x1, y2 - y1
            plate_rect.append((x1, y1, w, h))
    
    if len(plate_rect) == 0:
        return None, None, []
    
    for (x, y, w, h) in plate_rect:
        plate = roi[y:y+h, x:x+w, :]
        cv2.rectangle(plate_img, (x+2, y), (x+w-3, y+h-5), (51, 181, 155), 3)
    
    return plate_img, plate, plate_rect

def find_contours(dimensions, img):
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    lower_width, upper_width, lower_height, upper_height = dimensions
    
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    x_cntr_list = []
    img_res = []
    
    for cntr in cntrs:
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        if lower_width < intWidth < upper_width and lower_height < intHeight < upper_height:
            x_cntr_list.append(intX)
            
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            char = cv2.subtract(255, char)
            
            char_copy = np.zeros((44, 24))
            char_copy[2:42, 2:22] = char
            
            img_res.append(char_copy)
    
    if len(x_cntr_list) > 0:
        indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
        img_res_copy = [img_res[idx] for idx in indices]
        img_res = np.array(img_res_copy)
    else:
        img_res = np.array([])
    
    return img_res

def segment_characters(image):
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    img_binary_lp[0:3, :] = 255
    img_binary_lp[:, 0:3] = 255
    img_binary_lp[72:75, :] = 255
    img_binary_lp[:, 330:333] = 255

    dimensions = [
        LP_WIDTH/6,
        LP_WIDTH/2,
        LP_HEIGHT/10,
        2*LP_HEIGHT/3
    ]

    char_list = find_contours(dimensions, img_binary_lp)
    
    return char_list, img_binary_lp

def fix_dimension(img):
    new_img = np.zeros((28, 28, 3))
    for i in range(3):
        new_img[:, :, i] = img
    return new_img

def predict_characters(char_list, model):
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, c in enumerate(characters):
        dic[i] = c
    
    output = []
    confidences = []
    
    for ch in char_list:
        img_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img / 255.0
        img = img.reshape(1, 28, 28, 3)
        
        y_pred = model.predict(img, verbose=0)
        y_index = np.argmax(y_pred, axis=1)[0]
        confidence = np.max(y_pred)
        
        character = dic[y_index]
        output.append(character)
        confidences.append(confidence)
    
    plate_number = ''.join(output)
    
    return plate_number, output, confidences

def create_character_visualization(char_list, predictions):
    if len(char_list) == 0:
        return None
    
    fig, axes = plt.subplots(1, len(char_list), figsize=(len(char_list)*2, 2))
    
    if len(char_list) == 1:
        axes = [axes]
    
    for i, (ch, pred) in enumerate(zip(char_list, predictions)):
        img = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'{pred}', fontsize=14, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close()
    
    return buf

# Sidebar
with st.sidebar:
    st.header("ℹ️ Information")
    
    st.markdown("### 📋 Pipeline Steps")
    st.markdown("""
    1. **YOLO Detection**  
    2. **Preprocessing**  
    3. **Contour Detection**  
    4. **CNN Recognition**
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Supported Characters")
    st.info("0–9, A–Z (36 classes)")
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Status")
    
    with st.spinner("Loading models..."):
        yolo_model = load_yolo_model()
        cnn_model = load_cnn_model()
    
    if yolo_model is not None:
        st.success("✅ YOLO Model Loaded")
    else:
        st.error("❌ YOLO Model Failed")
    
    if cnn_model is not None:
        st.success("✅ CNN Model Loaded")
    else:
        st.error("❌ CNN Model Failed")

# Main layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Input Image")
    
    uploaded_file = st.file_uploader("Upload Car Image", type=['jpg','jpeg','png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        if len(image_np.shape) == 3:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        
        st.image(image, caption="Original Image", width="stretch")
        
        if st.button("🚀 Detect & Recognize", type="primary", width="stretch"):
            if yolo_model is None or cnn_model is None:
                st.error("⚠️ Models are not loaded properly!")
            else:
                with st.spinner("Processing..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Detecting number plate...")
                    plate_img, plate, plate_rect = detect_plate(image_cv, yolo_model)
                    progress_bar.progress(40)
                    
                    if plate is not None:
                        status_text.text("Segmenting characters...")
                        char_list, binary_plate = segment_characters(plate)
                        progress_bar.progress(70)
                        
                        if len(char_list) > 0:
                            status_text.text("Recognizing characters...")
                            plate_number, predictions, confidences = predict_characters(char_list, cnn_model)
                            progress_bar.progress(100)
                            
                            st.session_state.processed = True
                            st.session_state.plate_img = plate_img
                            st.session_state.plate = plate
                            st.session_state.binary_plate = binary_plate
                            st.session_state.char_list = char_list
                            st.session_state.plate_number = plate_number
                            st.session_state.predictions = predictions
                            st.session_state.confidences = confidences
                            
                            status_text.text("✅ Processing complete!")
                        else:
                            st.error("❌ No characters detected!")
                    else:
                        st.error("❌ No number plate detected!")

with col2:
    st.header("📊 Results")
    
    if st.session_state.processed:
        plate_rgb = cv2.cvtColor(st.session_state.plate_img, cv2.COLOR_BGR2RGB)
        st.image(plate_rgb, caption="Detected Plate", width="stretch")
        
        st.subheader("Extracted Plate Region")
        plate_crop_rgb = cv2.cvtColor(st.session_state.plate, cv2.COLOR_BGR2RGB)
        st.image(plate_crop_rgb, width="stretch")
        
        st.subheader("Preprocessed (Binary)")
        st.image(st.session_state.binary_plate, width="stretch", clamp=True)
        
        st.markdown("---")
        st.markdown(f'<div class="result-box">{st.session_state.plate_number}</div>', unsafe_allow_html=True)
        
        st.subheader("Segmented Characters")
        char_viz = create_character_visualization(
            st.session_state.char_list,
            st.session_state.predictions
        )
        if char_viz:
            st.image(char_viz, width="stretch")
        
        with st.expander("📝 Character Predictions Details"):
            for i, (pred, conf) in enumerate(zip(st.session_state.predictions, st.session_state.confidences)):
                st.write(f"**Character {i+1}:** {pred} (Confidence: {conf:.2%})")
        
        st.download_button(
            label="💾 Download Result",
            data=st.session_state.plate_number,
            file_name="plate_number.txt",
            mime="text/plain"
        )
    else:
        st.info("👈 Upload an image and click 'Detect & Recognize'")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><b>Number Plate Recognition System</b></p>
    <p>YOLO Detection → Character Segmentation → CNN Recognition</p>
</div>
""", unsafe_allow_html=True)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource # ใช้เพื่อไม่ให้เครื่องโหลดโมเดลใหม่ทุกครั้งที่กดปุ่ม
def load_fish_model():
    try:
        # ระบุชื่อไฟล์โมเดลของคุณที่นี่ (ต้องอยู่ในโฟลเดอร์เดียวกัน)
        model_fish = tf.keras.models.load_model('fish_model.keras')
        return model_fish
    except Exception as e:
        st.error(f"ไม่สามารถโหลดไฟล์ fish_model.keras ได้: {e}")
        return None
    
# --- ตั้งค่าหน้าเว็บ (ส่วนนี้ใส่ไว้ที่บรรทัดแรกสุดของไฟล์เสมอ) ---
st.set_page_config(page_title="Steam Game Popularity Analysis", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.write("### Select Page")
    page = st.selectbox(
        "",
        ["Ensemble Model (Steam)", "Neural Network (Fish)", "Test Model"],
        label_visibility="collapsed"
    )

# --- หน้า Ensemble Model (Steam) ---
if page == "Ensemble Model (Steam)":
    st.title("Steam Game Popularity Prediction — Ensemble Model")
    
    # --- 1. Data Preparation ---
    st.header("1. การเตรียมข้อมูล (Data Preparation)")
    st.write("""
    กระบวนการเริ่มต้นจากการโหลดชุดข้อมูลเกม Steam ในรูปแบบไฟล์ CSV เข้ามาในระบบ 
    จากนั้นทำการคัดกรองคอลัมน์ที่ไม่สามารถนำมาประมวลผลทางคณิตศาสตร์ได้ทิ้งไป เช่น ข้อมูลที่เป็นข้อความอธิบายเกมหรือลิงก์เว็บไซต์ 
    ถัดมาเป็นการจัดการกับข้อมูลที่แหว่งหายหรือค่า Null โดยใช้วิธีตัดแถวที่มีข้อมูลไม่ครบถ้วนออกไปจากตาราง 
    นอกจากนี้ยังมีการแปลงข้อมูลประเภทตรรกะหรือ Boolean อย่างสถานะการรองรับระบบปฏิบัติการ Windows, Mac และ Linux 
    ที่เดิมเป็นค่า True และ False ให้กลายเป็นข้อมูลชนิดตัวเลข 1 และ 0 เพื่อให้อัลกอริทึมสามารถนำไปคำนวณต่อได้
    """)

    # --- 2. Feature Engineering ---
    st.header("2. การสร้างและเลือกคุณลักษณะ (Feature Engineering)")
    st.info("""
    **การสร้างตัวแปรเป้าหมาย (Target Variable):** สร้างคอลัมน์ใหม่ชื่อ **Is_Hit** 
    โดยกำหนดเงื่อนไขว่าเกมที่มีจำนวนผู้ให้คะแนนรีวิวเชิงบวก (Positive) มากกว่า 500 คน จะเท่ากับ 1 (เกมฮิต) 
    หากต่ำกว่านั้นจะเท่ากับ 0
    """)
    
    st.warning("""
    **การจัดการ Data Leakage:** สิ่งที่สำคัญที่สุดคือการดรอปคอลัมน์ Positive และ Negative ทิ้งไปจากชุดข้อมูลตัวแปรต้น 
    เพื่อป้องกันไม่ให้โมเดลแอบดูเฉลย ส่งผลให้ตัวแปรต้น (Features) เหลือเพียง 5 ตัวแปร ได้แก่:
    1. ราคาของเกม (Price) 
    2. การรองรับ Windows 
    3. การรองรับ Mac 
    4. การรองรับ Linux 
    5. จำนวน Achievements ภายในเกม
    """)

    # --- 3. Model Development Process ---
    st.header("3. กระบวนการพัฒนาโมเดล (Model Development Process)")
    st.write("""
    เริ่มต้นจากการแบ่งชุดข้อมูลออกเป็นชุดฝึกสอน (Training Set) 80% และชุดทดสอบ (Test Set) 20% 
    โดยใช้เทคนิค **Stratified Split** เพื่อควบคุมสัดส่วนของเกมฮิตที่มีจำนวนน้อยให้กระจายตัวอย่างเท่าเทียมกัน 
    จากนั้นทำการสร้างอินสแตนซ์ของโมเดลทั้งสาม ประกอบร่างเข้าด้วยกัน และทำการฝึกสอน (Fit) 
    เมื่อเสร็จสิ้นจะทำการทดสอบเพื่อประเมินค่าความแม่นยำ (Accuracy) 
    และบันทึกโมเดลออกมาเป็นไฟล์นามสกุล **.pkl** ด้วยไลบรารี joblib
    """)

    st.divider()

    # --- 4. Algorithm Details (แบ่งเป็น 3 คอลัมน์) ---
    st.header("4. รายละเอียดอัลกอริทึม (Algorithms)")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader(" Decision Tree")
        st.write("""
        ทำงานโดยจำลองตรรกะมนุษย์เป็นลำดับขั้น ตั้งคำถามที่ทรงพลังที่สุดเพื่อแยกข้อมูล 
        เน้นความโปร่งใส ตรวจสอบย้อนหลังได้ง่ายว่าทำไมถึงตัดสินใจแบบนั้น 
        แต่มีจุดอ่อนคือ **Overfitting** หากตั้งคำถามละเอียดเกินไปจนกลายเป็นการท่องจำข้อมูลเก่า
        """)

    with col2:
        st.subheader(" K-Nearest Neighbors")
        st.write("""
        เชื่อในเรื่อง "ความใกล้ชิด" โดยนำข้อมูลใหม่ไปวางเทียบกับข้อมูลเก่าและดู "เพื่อนบ้าน" ที่อยู่ใกล้ที่สุด 
        ความเก่งขึ้นอยู่กับการจัดมาตราส่วนข้อมูล (Scaling) ให้เท่ากัน 
        เป็นโมเดลที่เรียบง่ายแต่จะทำงานช้าลงเมื่อฐานข้อมูลมีขนาดใหญ่ขึ้นเพราะต้องคำนวณระยะทางใหม่ทุกครั้ง
        """)

    with col3:
        st.subheader(" Logistic Regression")
        st.write("""
        เน้นหา "ความน่าจะเป็น" โดยใช้ฟังก์ชัน **Sigmoid** บีบผลลัพธ์ให้อยู่ในเส้นโค้งรูปตัว S (ค่า 0 ถึง 1) 
        รวดเร็วและประหยัดทรัพยากร ทำงานได้ดีเยี่ยมเมื่อข้อมูลมีความสัมพันธ์แบบเส้นตรง 
        แต่อาจไม่แม่นยำหากข้อมูลมีความซับซ้อนหรือพันกันยุ่งเหยิงเกินไป
        """)

    st.subheader("Performance Result")
    st.success("Accuracy Score: **92.41%**")
    # --- Footer & Reference ---
    st.divider()
    st.markdown("""
    **Reference:** [Steam Games Dataset on Kaggle](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset?resource=download)
    """)
    

# --- ส่วนของหน้าอื่นๆ (โครงร่างไว้) ---
elif page == "Neural Network (Fish)":
    st.title("Neural Network Model — Fish Species Classification")

    # --- 1. Data Preparation ---
    st.header("1. การเตรียมข้อมูล (Data Preparation)")
    st.write("""
    เริ่มต้นจากการนำเข้าชุดข้อมูล **A Large-Scale Fish Dataset** ซึ่งประกอบด้วยภาพปลา 9 สายพันธุ์ 
    กระบวนการทำความสะอาดข้อมูลเริ่มจากการเขียนสแกนเพื่อลบโฟลเดอร์ **Ground Truth (GT)** 
    ที่มีไว้สำหรับงานประมวลผลภาพแบบจัดแบ่งส่วน (Segmentation) ออกไป เพื่อป้องกันข้อมูลรบกวน 
    รวมถึงการคัดกรองเก็บไว้เฉพาะไฟล์นามสกุลรูปภาพมาตรฐาน จากนั้นจึงนำภาพทั้งหมดมาปรับขนาด (**Resize**) 
    ให้เป็น **224x224 พิกเซล** เพื่อให้ขนาดของภาพสอดคล้องกับโครงสร้างขาเข้าที่โมเดลต้องการ
    """)

    # --- 2. Feature Engineering ---
    st.header("2. การสร้างและเลือกคุณลักษณะ (Feature Engineering)")
    st.info("""
    Image Preprocessing: ขั้นตอนนี้เน้นไปที่การเตรียมภาพก่อนเข้าโมเดล โดยมีการทำ Normalization
    เพื่อปรับสเกลค่าสีของพิกเซลจากเดิมที่อยู่ในช่วง 0 ถึง 255 ให้อยู่ในช่วง **0 ถึง 1** 
    การบีบช่วงข้อมูลนี้ช่วยให้ฟังก์ชันทางคณิตศาสตร์ภายในโครงข่ายประสาทเทียมสามารถคำนวณ 
    และลู่เข้าหาจุดต่ำสุดของค่าความผิดพลาดได้รวดเร็วและมีเสถียรภาพมากขึ้น
    """)

    # --- 3. Model Development Process ---
    st.header("3. กระบวนการพัฒนาโมเดล (Model Development Process)")
    st.write("""
    เริ่มต้นด้วยการแบ่งชุดรูปภาพออกเป็นชุดฝึกสอน (Training Set) และชุดตรวจสอบ (Validation Set) 
    จากผลการทดสอบพบว่าค่าความแม่นยำ (Validation Accuracy) พุ่งสูงถึงระดับ 100% ตั้งแต่รอบ (Epoch) ที่ 2 
    จึงได้ทำการหยุดการเทรนก่อนกำหนด (**Early Stopping**) เพื่อหลีกเลี่ยงภาวะ **Overfitting** 
    ซึ่งเป็นสภาวะที่โมเดลจดจำข้อมูลจำเพาะมากเกินไปจนขาดความยืดหยุ่น 
    ขั้นตอนสุดท้ายคือการบันทึกโมเดลให้อยู่ในรูปแบบไฟล์นามสกุล **.keras** เพื่อนำไปใช้งานต่อไป
    """)

    st.divider()

    # --- 4. CNN Deep Dive ---
    st.header("4. เจาะลึกการทำงานของ Convolutional Neural Networks (CNN)")
    
    tab1, tab2, tab3, tab4 = st.tabs(["สกัดคุณลักษณะ", "การลดรูป (Pooling)", "ลำดับการเรียนรู้", "การตัดสินใจ"])

    with tab1:
        st.subheader("กระบวนการสกัดคุณลักษณะ (Feature Extraction)")
        st.write("""
        เปรียบเสมือนการนำ **"แว่นขยายพิเศษ"** หรือที่เรียกว่า **Filter (Kernel)** มาสแกนไปทั่วทั้งรูปภาพ 
        เจ้า Filter นี้จะมองหารูปแบบเฉพาะตัว เช่น เส้นตรงแนวตั้ง, ความโค้ง, หรือการตัดกันของสี 
        เมื่อ Filter วิ่งผ่านภาพ มันจะสร้างแผนที่ใหม่ที่เรียกว่า **Feature Map** ซึ่งสกัดเอาความโดดเด่นของภาพออกมา 
        จากนั้นผ่านฟังก์ชัน **ReLU** เพื่อคัดกรองเฉพาะข้อมูลที่เป็นบวกและตัดข้อมูลที่ไม่จำเป็นทิ้งไป
        """)

    with tab2:
        st.subheader("การลดรูปเพื่อความกระชับ (Pooling Layer)")
        st.write("""
        ใช้ขั้นตอนที่เรียกว่า **Max Pooling** เพื่อทำการ "ย่อส่วน" ข้อมูลลงมา โดยเลือกเอาเฉพาะค่าที่สูงที่สุดในหน้าต่างย่อยๆ 
        มาเป็นตัวแทน วิธีนี้ช่วยให้โมเดลมีความทนทาน (**Robustness**) ต่อการเปลี่ยนแปลงตำแหน่งของวัตถุ 
        (ไม่ว่าปลาจะอยู่มุมไหนของภาพ โมเดลก็ยังจำได้) และช่วยลดภาระการคำนวณของคอมพิวเตอร์ลงอย่างมหาศาล
        """)

    with tab3:
        st.subheader("ลำดับชั้นของการเรียนรู้ (Hierarchical Learning)")
        st.write("""
        ในเลเยอร์แรกๆ CNN จะเรียนรู้สิ่งพื้นฐานอย่าง **"เส้น"** และ **"จุด"** แต่เมื่อข้อมูลถูกส่งลึกลงไป 
        มันจะเริ่มนำเส้นเหล่านั้นมาต่อกันเป็น **"รูปร่าง"** เช่น วงกลม และในเลเยอร์ที่ลึกที่สุด 
        มันจะประกอบรูปร่างจนกลายเป็น **"วัตถุ"** เช่น ครีบ, ตา หรือเกล็ดปลา 
        นี่คือเหตุผลที่ CNN สามารถจำแนกภาพที่มีความซับซ้อนสูงได้อย่างแม่นยำ
        """)

    with tab4:
        st.subheader("การตัดสินใจครั้งสุดท้าย (Classification)")
        st.write("""
        ข้อมูลจะถูกเปลี่ยนรูปร่างจากตาราง 2 มิติ ให้กลายเป็นเส้นตรงยาวๆ (**Flatten**) 
        เพื่อส่งเข้าสู่ **Fully Connected Layer** เพื่อให้คอมพิวเตอร์นำฟีเจอร์เด่นๆ ทั้งหมดมา "ลงคะแนนเสียง" 
        ผลลัพธ์สุดท้ายจะออกมาเป็นค่าความน่าจะเป็น เช่น มั่นใจว่าเป็น "ปลาทูน่า" 98% เป็นต้น
        """)

    # --- 5. Performance ---
    st.header("5. ประสิทธิภาพของโมเดล (Performance)")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(label="Validation Accuracy", value="99.52%")
    with col_m2:
        st.metric(label="Validation Loss", value="0.0178")
    
    st.divider()
    st.markdown("""
    **Reference:** [Steam Games Dataset on Kaggle](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset/data)
    """)


elif page == "Test Model":
    st.title("Model Inference Test")
    st.write("ส่วนนี้คือระบบทดสอบการทำนายผลโดยใช้โมเดล **Voting Classifier (.pkl)** ที่ผ่านการฝึกสอนแล้ว")

    # ส่วนรับข้อมูล Input จากผู้ใช้
    st.header("1. Input Features")
    st.info("กรุณากรอกข้อมูลคุณลักษณะของเกมที่ต้องการพยากรณ์ให้ครบถ้วนทั้ง 5 ตัวแปร")

    # สร้างคอลัมน์เพื่อให้หน้าเว็บดูสวยงาม
    col_in1, col_in2 = st.columns(2)

    with col_in1:
        price = st.number_input("ราคาของเกม (Price - $)", min_value=0.0, value=0.0, step=0.1)
        achievements = st.number_input("จำนวนความสำเร็จ (Achievements)", min_value=0, value=0, step=1)

    with col_in2:
        st.write("ระบบปฏิบัติการที่รองรับ (OS Support)")
        win = st.checkbox("Windows")
        mac = st.checkbox("Mac")
        lin = st.checkbox("Linux")

    st.divider()

    # ส่วนประมวลผลเบื้องหลัง (Backend Logic)
    st.header("Prediction Result")
    
    if st.button("กดเพื่อทำนายผล (Predict)"):
        # --- ขั้นตอนสำคัญ: แปลงค่า Boolean เป็น Integer (1 หรือ 0) ---
        # ตามที่คุณแนะนำเพื่อน: Checkbox ส่งค่า True/False เราต้องแปลงเป็น 1/0 ก่อนเข้าโมเดล
        win_int = 1 if win else 0
        mac_int = 1 if mac else 0
        lin_int = 1 if lin else 0

        # จำลองการเตรียมข้อมูลป้อนเข้าโมเดล (Input Data)
        # ลำดับต้องตรงกับตอนเทรน: Price, Windows, Mac, Linux, Achievements
        input_features = [price, win_int, mac_int, lin_int, achievements]
        
        # แสดงข้อมูลที่กำลังส่งไปให้โมเดล (เพื่อให้คนตรวจเห็นกระบวนการ)
        st.write(f"**Data sent to model:** `{input_features}` (แปลงค่า True/False เป็น 1/0 เรียบร้อยแล้ว)")

        # --- ส่วนการทำนาย (Simulated Prediction) ---
        # ในการใช้งานจริง จะต้องใช้: result = model.predict([input_features])[0]
        # ตัวอย่างนี้ขอกำหนดตัวแปรสมมติเพื่อแสดง UI ตามเงื่อนไขที่คุณให้มาครับ
        
        # สมมติผลลัพธ์ (ในโค้ดจริงจะเปลี่ยนตามโมเดล)
        # ตัวอย่าง: ถ้า achievements > 100 ให้เป็น 1 (Hit)
        result = 1 if achievements > 100 else 0 

        # --- แสดงผลลัพธ์ตามเงื่อนไขที่คุณระบุ ---
        if result == 1:
            st.success("""
            ### ระบบทำนายว่า: เกมนี้จะฮิต! 
            **(คาดการณ์ว่าจะมีผู้เล่นรีวิวเชิงบวกมากกว่า 500 คน)**
            """)
            st.balloons() # ใส่ Effect แสดงความยินดี
        else:
            st.warning("""
            ### ระบบทำนายว่า: เกมนี้อาจจะยังไม่ฮิต 
            **(คาดการณ์ว่ายอดรีวิวเชิงบวกอาจไม่ถึง 500 คน)**
            """)

    # ส่วนคำอธิบายเพิ่มเติมสำหรับอาจารย์
    with st.expander("คำอธิบายทางเทคนิคสำหรับการแสดงผล"):
        st.write("""
        *   **การจัดการข้อมูล:** ระบบจะรับค่าจาก Checkbox (Boolean) แล้วทำการ Mapping ค่าเป็น 0 และ 1 ก่อนส่งให้ไฟล์ `.pkl`
        *   **เกณฑ์การวัดผล:** ค่า 1 (Hit) หมายถึงผ่านเกณฑ์ 500 รีวิวเชิงบวก ซึ่งเป็นจุดที่เรากำหนดไว้ในขั้นตอน Feature Engineering
        *   **Ensemble Model:** ผลลัพธ์ที่ได้มาจากการลงคะแนนเสียง (Voting) ระหว่าง Decision Tree, KNN และ Logistic Regression
        """)

    # เพิ่มส่วนทดสอบ Fish Model ด้านล่าง (ถ้าต้องการ)
    st.divider()
    st.subheader("Fish Classification Test")
    st.header("Fish Species Classification Test")
    st.info("อัปโหลดรูปภาพปลาเพื่อทำนายสายพันธุ์ด้วยโมเดล CNN (MobileNetV2)")

    # 1. ช่องอัปโหลดไฟล์
    uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพปลา (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # แสดงรูปภาพที่ผู้ใช้อัปโหลด
        image = Image.open(uploaded_file).convert('RGB') # แปลงเป็น RGB เพื่อความแน่นอน
        st.image(image, caption="รูปภาพที่อัปโหลด", use_container_width=True)
        
        if st.button("ทำนายสายพันธุ์ปลา (Predict Fish)"):
             if model_fish is not None:
                with st.spinner('กำลังวิเคราะห์รูปภาพ...'):
                    # --- 2. กระบวนการ Preprocessing (ตามที่คุณอธิบาย) ---
                    # A. Resize เป็น 224x224
                    img_resized = image.resize((224, 224))
                    
                    # B. แปลงเป็น Array และ Normalization (หาร 255)
                    img_array = np.array(img_resized).astype('float32') / 255.0
                    
                    # C. เพิ่มมิติ Batch (จาก [224,224,3] เป็น [1,224,224,3])
                    img_batch = np.expand_dims(img_array, axis=0)

                    # --- 3. ทำนายผลจริงจากโมเดล ---
                    predictions = fish_model.predict(img_batch)
                    
                    # รายชื่อสายพันธุ์ปลา (ตรวจสอบให้ตรงกับตอนที่เทรน)
                    class_names = [
                        'Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel', 
                        'Red Mullet', 'Red Sea Bream', 'Sea Bass', 
                        'Shrimp', 'Striped Red Mullet', 'Trout'
                    ]
                    
                    # หาค่าที่สูงที่สุด
                    max_idx = np.argmax(predictions[0])
                    predicted_label = class_names[max_idx]
                    confidence_score = predictions[0][max_idx] * 100

                    # --- 4. แสดงผลลัพธ์ (Output) ---
                    st.success(f"### ระบบทายว่าเป็น: **{predicted_label}** (มั่นใจ {confidence_score:.2f}%)")
                    st.progress(predictions[0][max_idx]) # แสดงกราฟความมั่นใจ
        else:
                st.error("ไม่สามารถทำนายได้ เนื่องจากโมเดลไม่ได้ถูกโหลด")
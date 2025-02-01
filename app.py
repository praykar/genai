from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import face_recognition
import numpy as np
import pandas as pd
import threading
import queue
import random
import time
import io
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from huggingface_hub import InferenceClient

# Initialize HuggingFace client
KEY = os.getenv('HF_TOKEN')
client = InferenceClient("black-forest-labs/FLUX.1-dev", token=KEY)

class MessageType(Enum):
    PROGRESS = auto()
    IMAGE = auto()
    ERROR = auto()
    COMPLETE = auto()

@dataclass
class Message:
    type: MessageType
    data: dict

class AdGenerator:
    def __init__(self):
        # self.taglines = [
        #     "Unlock your dreams, one loan at a time.", "Your goals, our priority—approved in minutes.", "Trust us to fund your tomorrow, today.",
        #     "Build bigger with flexible loans, zero regrets.", "From aspirations to assets—we’ve got your back.", "Breathe easy. Borrow smarter. Live freely.",
        #     "Loans that grow with you, not against you.", "Step forward confidently—your financial freedom starts here.", "Secure your future, one seamless loan at a time.",
        #     "Fuel ambition. Expand possibilities. Repay comfortably.", "Your journey, our commitment—borrow with peace of mind.", "Tailored loans for life’s unpredictable adventures.",
        #     "No hurdles, just hope—apply stress-free.", "Turn plans into action with a single yes.", "Empower progress. Own your path. We’ll fund it."
        # ]
        self.taglines = { 
            "home" : ["Turn keys to your dream home, stress free.", "Build your forever, brick by brick.", "Your home, our promise approved faster.",
                            "From foundations to rooftops, we finance it all.", "Unlock the door to your future today."],
            "personal" : ["Life’s surprises? We’ve got you covered.", "Your plans, our priority funds in minutes.", "Flexible loans for life’s unpredictable chapters.",
                                "Need cash? Say goodbye to compromises.", "Dream bigger. Borrow smarter. Live freely."],
            "jewel" : ["Unlock cash, keep your treasures safe.", "Your gold’s value, instantly in your hands.", "Secure loans, no parting with your heirlooms.",
                             "Turn jewels into liquidity, effortlessly.", "Value preserved, funds accessed zero stress."],
            "car" : ["Drive home your dream car, today.", "Fuel your journey, one affordable EMI at a time.", "New wheels, simpler deals approved fast.",
                           "From commutes to adventures, we finance them all.", "Your road to ownership starts here."],
            "education" : ["Invest in tomorrow’s success, today.", "Your degree, our support no boundaries.", "Learn fearlessly. We’ll handle the fees.",
                                 "Education unlocked, future secured.", "Bright minds deserve brighter opportunities."],
            "credit card" : ["Instant cash, just a swipe away.", "Turn credit into possibilities, effortlessly.", "Need liquidity? Your card’s got more power.",
                                   "Flexible funds, zero collateral just your limit.", "Unlock cash without emptying your wallet."]
        }

    def load_font(self, font_path, font_size):
        try:
            return ImageFont.truetype(font_path, font_size)
        except:
            default_font = ImageFont.load_default()
            return default_font.font_variant(size=font_size)

    def create_banner(self, width=1200, height=120):
        """Create a banner with specified dimensions"""
        image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        background = Image.new('RGBA', (width, height), (242, 101, 34, 255))
        image.paste(background, (0, 0), background)
        
        draw = ImageDraw.Draw(image)
        font_size = height * 0.25
        font = self.load_font("Helvetica.ttc", font_size)
        
        # Add banner text and social media icons
        texts = ["Low Interest Rates", '|', "Hassle Free process", '|', "Flexible tenure"]
        self._add_banner_text(draw, texts, width, height, font)
        self._add_social_media_section(draw, image, width, height, font_size)
        self._add_disclaimer(draw, width, height, font_size)
        
        return image

    def _add_banner_text(self, draw, texts, width, height, font):
        """Add main text to banner"""
        box_height = int(height * 0.4)
        box_width = int(width / len(texts))
        spacing = (width - (box_width * len(texts))) // len(texts) + 1

        for i, text in enumerate(texts):
            x1 = spacing + (i * (box_width + spacing))
            y1 = int(height * 0.02)
            
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            text_x = x1 + (box_width - text_width) // 2
            text_y = y1 + (box_height - text_height) // 2
            
            draw.text((text_x, text_y), text, fill='white', font=font, 
                     stroke_width=0.2, stroke_fill='white')

    def _add_social_media_section(self, draw, image, width, height, font_size):
        """Add social media section to banner"""
        follow_text = "Follow us on:"
        follow_font_size = int(font_size * 0.85)
        follow_font = self.load_font("Helvetica.ttc", follow_font_size)
        
        follow_text_bbox = draw.textbbox((0, 0), follow_text, font=follow_font)
        follow_text_width = follow_text_bbox[2] - follow_text_bbox[0]
        icon_paths = ["facebook.png", "twitter.png", "instagram.png", "linkedin.png", "youtube.png", "whatsapp.png"]
        icons = []
        for path in icon_paths:
            if os.path.exists(path):
                icon = Image.open(path)
                if icon.mode != 'RGBA':
                    icon = icon.convert('RGBA')
                icons.append(icon)

        follow_text_x = int((width) - (follow_text_width * 2.8))
        follow_text_y = int(height * 0.55)
        icon_size = int(follow_font_size*1.2)
        icons = [icon.resize((icon_size, icon_size)) for icon in icons]
    
        icon_x = follow_text_x + follow_text_width + 5
        icon_y = follow_text_y + (follow_font_size - icon_size) // 2
    
        for icon in icons:
            image.paste(icon, (int(icon_x), int(icon_y)), icon)
            icon_x += icon.width + 10
        draw.text((follow_text_x, follow_text_y), follow_text, 
                 fill='white', font=follow_font, stroke_width=0.2, stroke_fill='white')

    def _add_disclaimer(self, draw, width, height, font_size):
        """Add disclaimer text to banner"""
        disclaimer_text = "This image was generated using artificial intelligence and may not depict real people, places, or events. " \
                         "Any resemblance to actual individuals or situations is purely coincidental."
        disclaimer_font_size = int(font_size * 0.6)
        disclaimer_font = self.load_font("Helvetica.ttc", disclaimer_font_size)
        
        disclaimer_text_bbox = draw.textbbox((0, 0), disclaimer_text, font=disclaimer_font)
        disclaimer_text_width = disclaimer_text_bbox[2] - disclaimer_text_bbox[0]
        disclaimer_text_x = (width - disclaimer_text_width) // 2
        disclaimer_text_y = height - disclaimer_font_size - 5
        
        draw.text((disclaimer_text_x, disclaimer_text_y), disclaimer_text, 
                 fill='white', font=disclaimer_font, stroke_width=0.4, stroke_fill='white')

    def generate_base_image(self, product, age, location, gender, profession):
        """Generate base image using AI model"""
        # Standardize product naming
        product = product.strip()  # Remove any whitespace
        
        # Map variations to standard names
        product_mapping = {
            'jewel': 'gold_ornament',
            'jewel loan': 'gold_ornament',
            'personal': 'personal_loan_setup',
            'personal loan': 'personal_loan_setup',
            'home': 'house loan',
            'home loan': 'house loan',
            'education loan': 'education_loan_setup',
            'education': 'education_loan_setup',
            'car': 'new_car',
            'car loan': 'new_car'
        }
        
        # Convert to lowercase for case-insensitive matching
        product_key = product.lower()
        prompt_product = product_mapping.get(product_key, product)
            
        # prompt = f"{age}-year-old happy {gender} {profession}, {location}, India, {product} (hidden logo) in foreground, " \
        #         f"sharp focus, beside person. Realistic lighting, natural daylight, warm tones, soft shadows. " \
        #         f"Lifestyle setting, no text, mid-shot, clean composition, cinematic framing."
        prompt = f"A {prompt_product.split()[0]} positioned dominantly without specifications in the foreground under crisp focus, adjacent to a {age}-year-old cheerful {gender} {profession} in {location}," \
                f"Realistic lighting, natural daylight, warm tones, soft shadows. Lifestyle setting, no text, mid-shot, clean composition, cinematic framing."

        try:
            image = client.text_to_image(prompt)
            caption = prompt
            #caption = f"{prompt_product.split()[0]}: {age}-year {gender} {profession}, {location}" 
            return image, caption
        except Exception as e:
            raise Exception(f"Error generating image: {str(e)}")

    def detect_faces(self, image):
        """Detect faces in image using face_recognition"""
        face_locations = face_recognition.face_locations(np.array(image), model="hog")
        return [{'x': left, 'y': top, 'width': right - left, 'height': bottom - top} 
                for top, right, bottom, left in face_locations]

    def apply_branding(self, img, banner, uploaded_logo, product, logo_position="top_right"):
        """Apply branding elements to the image"""
        # Process logo
        logo = Image.open(uploaded_logo)
        logo_width = int(img.width * 0.2 * 1.5)
        logo_height = int(logo.height * (logo_width / logo.width) * 1.2)
        logo = logo.resize((logo_width, logo_height))
        
        # Apply logo
        logo_x = img.width - logo_width - 10 if logo_position == "top_right" else 10
        logo_y = 10
        
        if logo.mode == 'RGBA':
            mask = logo.split()[3]
            rgb_logo = logo.convert('RGB')
            img.paste(rgb_logo, (logo_x, logo_y), mask)
        else:
            rgb_logo = logo.convert('RGB')
            img.paste(rgb_logo, (logo_x, logo_y))
        
        # Add banner
        banner_width = int(img.width)
        banner_height = int(banner.height * (banner_width / banner.width))
        banner = banner.resize((banner_width, banner_height))
        
        # Create final image with banner
        final_image = Image.new('RGB', (img.width, img.height + banner_height), color=(255, 255, 255))
        final_image.paste(img, (0, 0))
        final_image.paste(banner, (0, img.height))
        
        # Add tagline
        self._add_tagline(final_image, product, img.height)
        
        return final_image

    def _add_tagline(self, image, product, original_height):
    """Add tagline to the image with a translucent banner background"""
    # Create a new RGBA layer for drawing
    txt_layer = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)
    
    # Standardize product name to match taglines dictionary keys
    product_key = product.strip()
    if product_key not in self.taglines:
        raise ValueError(f"No taglines found for product: {product_key}")
        
    tagline = random.choice(self.taglines[product_key])
    font_size = int(original_height * 0.05)
    font = self.load_font("Helvetica.ttc", font_size)
    
    faces = self.detect_faces(image)
    if faces:
        best_face = max(faces, key=lambda f: f['width'] * f['height'])
        self._position_tagline_with_banner(draw, txt_layer, tagline, font, best_face, font_size)
    else:
        self._position_tagline_fallback(draw, txt_layer, tagline, font, font_size)
    
    # Blend the text layer with the original image
    return Image.alpha_composite(image.convert('RGBA'), txt_layer)

def _position_tagline_fallback(self, draw, txt_layer, tagline, font, font_size):
    """Position tagline when no faces are detected"""
    words = tagline.split()
    if len(words) > 4:
        mid = len(words) // 2
        lines = [' '.join(words[:mid]), ' '.join(words[mid:])]
    else:
        lines = [tagline]
    
    margin_top = int(txt_layer.height * 0.25)
    line_spacing = int(font_size * 1.5)
    padding = int(font_size * 0.5)  # Padding for banner
    
    # Calculate total banner dimensions
    max_width = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        width = bbox[2] - bbox[0]
        max_width = max(max_width, width)
    
    banner_height = (len(lines) * line_spacing) + (padding * 2)
    banner_width = max_width + (padding * 2)
    
    # Draw translucent banner
    banner_x = (txt_layer.width - banner_width) // 2
    banner_y = margin_top - padding
    
    # Create banner shape
    banner_shape = [(banner_x, banner_y),
                   (banner_x + banner_width, banner_y + banner_height)]
    draw.rectangle(banner_shape, fill=(242, 101, 34, 128))  # Semi-transparent orange
    
    # Draw text
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (txt_layer.width - text_width) // 2
        text_y = margin_top + (i * line_spacing)
        
        # Draw text
        draw.text(
            (text_x, text_y),
            line,
            fill=(255, 255, 255, 255),  # White text
            font=font
        )

def _position_tagline_with_banner(self, draw, txt_layer, tagline, font, face, font_size):
    """Position tagline with banner avoiding face overlap"""
    words = tagline.split()
    if len(words) > 4:
        mid = len(words) // 2
        lines = [' '.join(words[:mid]), ' '.join(words[mid:])]
    else:
        lines = [tagline]
    
    margin_x = int(txt_layer.width * 0.05)
    margin_y = int(txt_layer.height * 0.05)
    line_spacing = int(font_size * 1.2)
    padding = int(font_size * 0.5)  # Padding for banner
    
    # Calculate total text dimensions
    max_width = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        width = bbox[2] - bbox[0]
        max_width = max(max_width, width)
    
    total_text_height = len(lines) * line_spacing
    banner_height = total_text_height + (padding * 2)
    banner_width = max_width + (padding * 2)
    
    # Define potential positions
    positions = [
        {
            'y': max(margin_y, face['y'] - banner_height - margin_y),
            'space': face['y'] - margin_y
        },
        {
            'y': min(face['y'] + face['height'] + margin_y,
                    txt_layer.height - banner_height - margin_y),
            'space': txt_layer.height - (face['y'] + face['height'] + margin_y) - banner_height
        }
    ]
    
    best_position = max(positions, key=lambda p: p['space'])
    banner_y = best_position['y']
    
    # Choose side with more space
    left_space = face['x'] - margin_x
    right_space = txt_layer.width - (face['x'] + face['width']) - margin_x
    text_align = 'left' if left_space > right_space else 'right'
    
    # Calculate banner position
    if text_align == 'left':
        banner_x = margin_x
    else:
        banner_x = txt_layer.width - banner_width - margin_x
    
    # Draw translucent banner
    banner_shape = [(banner_x, banner_y),
                   (banner_x + banner_width, banner_y + banner_height)]
    draw.rectangle(banner_shape, fill=(242, 101, 34, 128))  # Semi-transparent orange
    
    # Draw text
    for i, line in enumerate(lines):
        text_y = banner_y + padding + (i * line_spacing)
        
        if text_align == 'left':
            text_x = margin_x + padding
        else:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = txt_layer.width - text_width - margin_x - padding
        
        # Draw text
        draw.text(
            (text_x, text_y),
            line,
            fill=(255, 255, 255, 255),  # White text
            font=font
        )
        
    def generate_advertisement(self, data, uploaded_logo, message_queue=None, index=None):
        """Generate a single advertisement"""
        try:
            # Generate base image
            img, caption = self.generate_base_image(
                data['Product'], data['age'], data['location'],
                data['gender'], data['job']
            )
            
            # Create banner and apply branding
            banner = self.create_banner(width=int(img.width), height=int(img.height * 0.08))
            final_image = self.apply_branding(img, banner, uploaded_logo, data['Product'])
            
            # Report progress if in batch mode
            if message_queue and index is not None:
                message_queue.put(Message(
                    type=MessageType.PROGRESS,
                    data={'index': index, 'image': final_image, 'caption': caption}
                ))
            
            return final_image, caption
            
        except Exception as e:
            if message_queue and index is not None:
                message_queue.put(Message(
                    type=MessageType.ERROR,
                    data={'index': index, 'error': str(e)}
                ))
            raise e

def create_streamlit_ui():
    """Create the Streamlit UI"""
    st.set_page_config(page_title="Dynamic ADs Generation", page_icon="🎨")
    st.header("Upload Logo in PNG")
    uploaded_logo = st.file_uploader("Upload a Logo file", type=["png"])
    
    if uploaded_logo:
        generator = AdGenerator()
        tab1, tab2 = st.tabs(["🔣 Single Ad", "🗃 Bulk Generation"])
        
        with tab1:
            st.header("Generate Single Advertisement")
            
            # Create two columns for input fields
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=18, max_value=100, value=30,
                                    help="Enter age between 18-100")
                
                gender = st.selectbox("Gender",
                                    options=["Male", "Female"],
                                    help="Select gender")
                
                profession = st.text_input("Profession",
                                        placeholder="e.g., Software Engineer, Doctor, Teacher",
                                        help="Enter the profession of the person")
            
            with col2:
                location = st.selectbox("Location",
                                    options=["Mumbai", "Delhi", "Bangalore", "Chennai",
                                            "Hyderabad", "Kolkata", "Pune", "Other"],
                                    help="Select or type a location")
                
                if location == "Other":
                    location = st.text_input("Enter Location",
                                           placeholder="Enter city name")
                
                product = st.selectbox("Product Type",
                      options=list(generator.taglines.keys()),
                      help="Select the type of financial product")
            
            # Generation container
            generation_container = st.container()
            
            with generation_container:
                if st.button("Generate Advertisement", type="primary", use_container_width=True):
                    if age and gender and profession and location and product:
                        try:
                            with st.status("Generating your advertisement...", expanded=True) as status:
                                st.write("🎨 Creating advertisement...")
                                
                                data = {
                                    'Product': product,
                                    'age': age,
                                    'gender': gender.lower(),
                                    'job': profession,
                                    'location': location
                                }
                                
                                final_image, caption = generator.generate_advertisement(data, uploaded_logo)
                                
                                st.write("✅ Advertisement generated successfully!")
                                status.update(label="Advertisement generated successfully!", state="complete")
                                
                                # Show the generated image
                                st.subheader("Generated Advertisement")
                                st.image(final_image, caption=caption, use_container_width=True)
                                
                                # Add download button
                                buf = io.BytesIO()
                                final_image.save(buf, format="PNG")
                                byte_im = buf.getvalue()
                                
                                st.download_button(
                                    label="Download Advertisement",
                                    data=byte_im,
                                    file_name=f"ad_{product.lower().replace(' ', '_')}.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                                
                                if st.button("Generate Another Version", use_container_width=True):
                                    st.rerun()
                        
                        except Exception as e:
                            st.error(f"Error generating image: {str(e)}")
                            st.button("Try Again", use_container_width=True, on_click=st.rerun)
                    else:
                        st.error("Please fill in all fields before generating the advertisement.")
            
            # Tips section moved outside generation container
            st.divider()
            st.subheader("Tips for Better Results")
            st.markdown("""
            ### 👉 Optimization Tips:
            1. **Age**: Choose an age that matches your target demographic
            2. **Profession**: Be specific with professions (e.g., 'Senior Software Engineer' instead of just 'Engineer')
            3. **Location**: Using major cities tends to give better results
            4. **Gender**: Select the gender that best represents your target audience
            
            ### 🎯 Best Practices:
            - Ensure your logo is clear and high-quality
            - Consider the target audience for your financial product
            - Test different combinations for optimal results
            """)
        
        with tab2:
            st.header("Generate Bulk Advertisements")
            
            # Create two columns for input fields
            col1, col2 = st.columns(2)
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Upload CSV File",
                    type=['csv'],
                    help="Upload a CSV file with columns: Product, age, gender, job, location"
                )
                
                # CSV format info
                st.subheader("CSV Format Guide")
                st.markdown("""
                Your CSV should have the following columns:
                - Product: Type of financial product
                - age: Age of target customer (18-100)
                - gender: Male/Female
                - job: Customer profession
                - location: City name
                """)
                
                st.markdown("**Sample Data:**")
                st.code("""Product,age,gender,job,location
Home Loan,35,Male,Software Engineer,Mumbai
Personal Loan,28,Female,Doctor,Delhi
Business Loan,45,Male,Entrepreneur,Bangalore""")
            
            df = None
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    with col2:
                        st.subheader("Data Preview")
                        st.dataframe(df.head(3), use_container_width=True)
                        
                        st.subheader("Data Summary")
                        st.write(f"Total Rows: {len(df)}")
                        st.write("Product Distribution:")
                        st.dataframe(df['Product'].value_counts().head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
            
            if df is not None:
                st.divider()
                
                # Generation controls
                with st.container():
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        num_workers = st.slider(
                            "Number of Parallel Workers",
                            min_value=1,
                            max_value=min(8, len(df)),
                            value=min(4, len(df)),
                            help="Adjust the number of parallel processes for image generation"
                        )
                    
                    with col4:
                        batch_size = st.number_input(
                            "Batch Size",
                            min_value=1,
                            max_value=len(df),
                            value=min(10, len(df)),
                            help="Number of images to generate in one batch"
                        )
                    
                    if st.button("Generate Advertisements", type="primary", use_container_width=True):
                        # Validate required columns
                        required_columns = ['Product', 'age', 'gender', 'job', 'location']
                        if all(col in df.columns for col in required_columns):
                            try:
                                # Create message queue for communication
                                message_queue = queue.Queue()
                                
                                # Create UI elements for progress tracking
                                progress_container = st.container()
                                with progress_container.status("Generating advertisements...", expanded=True) as status:
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                
                                # Sample data for batch processing
                                batch_data = df.sample(batch_size).reset_index(drop=True)
                                
                                # Create grid layout for images
                                grid_container = st.container()
                                num_cols = 2
                                
                                with grid_container:
                                    st.markdown("### Generating Advertisements")
                                    num_rows = (batch_size + num_cols - 1) // num_cols
                                    image_placeholders = {}
                                    
                                    for row in range(num_rows):
                                        cols = st.columns(num_cols)
                                        for col in range(num_cols):
                                            idx = row * num_cols + col
                                            if idx < batch_size:
                                                image_placeholders[idx] = cols[col]
                                
                                # Process images in parallel
                                completed = 0
                                generated_images = {}
                                
                                def process_batch_row(index, row):
                                    try:
                                        final_image, caption = generator.generate_advertisement(
                                            row.to_dict(),
                                            uploaded_logo,
                                            message_queue,
                                            index
                                        )
                                        return True
                                    except Exception as e:
                                        message_queue.put(Message(
                                            type=MessageType.ERROR,
                                            data={'index': index, 'error': str(e)}
                                        ))
                                        return False
                                
                                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                                    futures = [
                                        executor.submit(process_batch_row, idx, row)
                                        for idx, row in batch_data.iterrows()
                                    ]
                                    
                                    while completed < batch_size:
                                        try:
                                            message = message_queue.get(timeout=0.1)
                                            
                                            if message.type == MessageType.PROGRESS:
                                                completed += 1
                                                progress_bar.progress(completed / batch_size)
                                                status_text.text(f"Generated {completed} of {batch_size} images")
                                                
                                                # Store generated image
                                                idx = message.data['index']
                                                generated_images[idx] = (
                                                    message.data['image'],
                                                    message.data['caption']
                                                )
                                                
                                                # Display image
                                                with image_placeholders[idx]:
                                                    st.image(message.data['image'],
                                                            caption=message.data['caption'],
                                                            use_container_width=True)
                                                    
                                                    # Add download button
                                                    buf = io.BytesIO()
                                                    message.data['image'].save(buf, format="PNG")
                                                    byte_im = buf.getvalue()
                                                    
                                                    st.download_button(
                                                        label=f"Download Image {idx + 1}",
                                                        data=byte_im,
                                                        file_name=f"ad_batch_{idx + 1}.png",
                                                        mime="image/png",
                                                        use_container_width=True,
                                                        key=f"download_{idx}"
                                                    )
                                            
                                            elif message.type == MessageType.ERROR:
                                                st.error(f"Error generating image {message.data['index'] + 1}: {message.data['error']}")
                                                completed += 1
                                                
                                        except queue.Empty:
                                            continue
                                    
                                    # Wait for all futures to complete
                                    for future in futures:
                                        future.result()
                                
                                if completed == batch_size:
                                    st.success(f"Successfully generated all {batch_size} images!")
                                else:
                                    st.warning(f"Completed with {completed} out of {batch_size} images generated.")
                                
                            except Exception as e:
                                st.error(f"Error during batch generation: {str(e)}")
                        else:
                            st.error("Please ensure your CSV has all required columns: Product, age, gender, job, location")

if __name__ == "__main__":
    create_streamlit_ui()

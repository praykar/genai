from PIL import Image, ImageDraw, ImageFont
import os
import random
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import InferenceClient
import pandas as pd
import numpy as np
import face_recognition
import threading
import queue
from dataclasses import dataclass
from enum import Enum, auto
import time
import io

KEY = os.getenv('HF_TOKEN')
client = InferenceClient("black-forest-labs/FLUX.1-dev", token=KEY)

# Define message types for queue communication
class MessageType(Enum):
    PROGRESS = auto()
    IMAGE = auto()
    ERROR = auto()
    COMPLETE = auto()

@dataclass
class Message:
    type: MessageType
    data: dict
    
# Pre-load fonts to reuse
def load_font(font_path, font_size):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        default_font = ImageFont.load_default()
        font = default_font.font_variant(size=font_size)
    return font

def create_banner(width=1200, height=120):
    image = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    background = Image.new('RGBA', (width, height), (242, 101, 34, 255))
    image.paste(background, (0, 0), background)

    draw = ImageDraw.Draw(image)
    font_size = image.height * 0.25 
    font = load_font("Helvetica.ttc", font_size)

    texts = ["Low Interest Rates",'|', "Hassle Free process",'|', "Flexible tenure"]

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

        draw.text((text_x, text_y), text, fill='white', font=font, stroke_width=0.2, stroke_fill='white')

    follow_text = "Follow us on:"
    follow_font_size = int(font_size * 0.85)
    follow_font = load_font("Helvetica.ttc", follow_font_size)

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

    draw.text((follow_text_x, follow_text_y), follow_text, fill='white', font=follow_font, stroke_width=0.2, stroke_fill='white')

    disclaimer_text = "This image was generated using artificial intelligence and may not depict real people, places, or events.\
    Any resemblance to actual individuals or situations is purely coincidental."
    disclaimer_font_size = int(font_size * 0.6)
    disclaimer_font = load_font("Helvetica.ttc", disclaimer_font_size)

    disclaimer_text_bbox = draw.textbbox((0, 0), disclaimer_text, font=disclaimer_font)
    disclaimer_text_width = disclaimer_text_bbox[2] - disclaimer_text_bbox[0]
    disclaimer_text_x = (width - disclaimer_text_width) // 2
    disclaimer_text_y = height - disclaimer_font_size - 5

    draw.text((disclaimer_text_x, disclaimer_text_y), disclaimer_text, fill='white', font=disclaimer_font, stroke_width=0.4, stroke_fill='white')

    return image

def detect_faces(image):
    face_locations = face_recognition.face_locations(np.array(image), model="hog")
    faces = [{'x': left, 'y': top, 'width': right - left, 'height': bottom - top} for top, right, bottom, left in face_locations]
    return faces

def save_genimage(product, age, location, income, gender, profession):
    if product.lower() == 'jewel':
        product = 'gold jewellery'
    elif product.lower() == 'personal':
        product = 'vacation'
    system_prompt = f"{age}-year-old {gender} {profession}, {location}, India, {product} (hidden logo) in foreground, sharp focus, beside person.\
    Realistic lighting, natural daylight, warm tones, soft shadows. Lifestyle setting, no text, mid-shot, clean composition, cinematic framing."
    try:
        image = client.text_to_image(system_prompt)
    except Exception as e:
        raise st.error(f"Error Generating Image: {str(e)}")
    caption = " ".join(system_prompt.split()[:6])
    return image, caption

def resize_logo(uploaded_logo, target_width=300, target_height=75):
    """
    Resize uploaded logo to standard dimensions while maintaining aspect ratio.
    Adds padding if necessary to meet target dimensions.
    
    Parameters:
    uploaded_logo: Streamlit uploaded file or bytes
    target_width: int, desired width in pixels (default 300)
    target_height: int, desired height in pixels (default 75)
    
    Returns:
    PIL.Image: Resized and padded logo image
    """
    try:
        # Convert uploaded file to PIL Image
        if isinstance(uploaded_logo, bytes):
            image = Image.open(io.BytesIO(uploaded_logo))
        else:
            image = Image.open(uploaded_logo)
            
        # Convert to RGBA to handle transparency
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
            
        # Calculate aspect ratios
        target_ratio = target_width / target_height
        image_ratio = image.width / image.height
        
        # Calculate new dimensions maintaining aspect ratio
        if image_ratio > target_ratio:
            # Image is wider than target ratio
            new_width = target_width
            new_height = int(target_width / image_ratio)
        else:
            # Image is taller than target ratio
            new_height = target_height
            new_width = int(target_height * image_ratio)
            
        # Resize image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with padding
        final_image = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 0))
        
        # Calculate padding to center the image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Paste resized image onto padded background
        final_image.paste(resized_image, (x_offset, y_offset), resized_image)
        
        return final_image
        
    except Exception as e:
        raise Exception(f"Error processing logo: {str(e)}")

def validate_and_resize_logo(uploaded_logo, min_width=180, min_height=45, max_width=500, max_height=125):
    """
    Validates logo dimensions and resizes if necessary.
    
    Parameters:
    uploaded_logo: Streamlit uploaded file
    min_width: int, minimum allowed width
    min_height: int, minimum allowed height
    max_width: int, maximum allowed width
    max_height: int, maximum allowed height
    
    Returns:
    PIL.Image: Processed logo image
    """
    try:
        # Open image to check dimensions
        image = Image.open(uploaded_logo)
        width, height = image.size
        
        # Calculate target size based on original dimensions
        if width < min_width or height < min_height:
            # If image is too small, resize to standard size
            return resize_logo(uploaded_logo, target_width=300, target_height=75)
        elif width > max_width or height > max_height:
            # If image is too large, resize to maximum allowed size
            return resize_logo(uploaded_logo, target_width=max_width, target_height=max_height)
        else:
            # If image is within acceptable range, maintain original size
            return resize_logo(uploaded_logo, target_width=width, target_height=height)
            
    except Exception as e:
        raise Exception(f"Error validating logo: {str(e)}")

def process_logo_upload(uploaded_file):
    """
    Process logo upload in Streamlit app.
    
    Parameters:
    uploaded_file: st.file_uploader result
    
    Returns:
    PIL.Image: Processed logo image
    """
    if uploaded_file is not None:
        try:
            # Validate and resize logo
            processed_logo = validate_and_resize_logo(uploaded_file)
            
            # Preview resized logo
            st.image(processed_logo, caption="Processed Logo", use_column_width=False)
            
            return processed_logo
            
        except Exception as e:
            st.error(f"Error processing logo: {str(e)}")
            return None
    return None
    
def apply_tagline_and_logo(img, banner, logo, logo_position="top_left"):
    """
    Adds a logo and a tagline to the image and the banner.
    The logo is placed according to the `logo_position` argument.
    The tagline is added below the image or banner if provided.
    """
    """Applies tagline and logo to the image based on face locations."""
    # Load and resize the logo
    logo_width = int(img.width * 0.2 * 1.5)  # 20% of image width
    logo_height = int(logo.height * (logo_width / logo.width) * 1.2)
    logo = logo.resize((logo_width, logo_height))
    # Get image dimensions
    img_width, img_height = img.size
    # Calculate logo position
    if logo_position == "top_left":
        logo_x = 10
        logo_y = 10
    elif logo_position == "top_right":
        logo_x = img.width - logo_width - 10
        logo_y = 10
    else:
        raise ValueError("Invalid logo_position. Choose 'top_left' or 'top_right'.")
    
    # Ensure the logo has transparency handling
    if logo.mode == 'RGBA':
        mask = logo.split()[3]
        rgb_logo = logo.convert('RGB')
        img.paste(rgb_logo, (logo_x, logo_y), mask)  # Use mask for transparency if logo has it
    else:
        rgb_logo = logo.convert('RGB')
        img.paste(rgb_logo, (logo_x, logo_y))  # Use RGB if no transparency
    
    # Resize the banner if it's already an Image object
    banner_width = int(img.width)  # 100% of image width
    banner_height = int(banner.height * (banner_width / banner.width))
    banner = banner.resize((banner_width, banner_height))

    # Create a new blank image with the calculated dimensions
    new_image = Image.new('RGB', (img.width, img.height + banner_height), color=(255, 255, 255))
    
    # Paste the first image at the top
    new_image.paste(img, (0, 0))
    
    # Paste the second image (banner) at the bottom
    new_image.paste(banner, (0, img.height))

    # Add tagline to the image
    draw = ImageDraw.Draw(new_image)
    
    # Randomly choose a tagline
    taglines = [
        "Dreams Within Reach", "Empowering Your Goals", "Finance Your Future", 
        "Simplify Your Tomorrow", "Loans Made Easy", "Borrow With Confidence", 
        "Unlock New Possibilities", "Invest In You", "Quick, Easy Loans", 
        "Grow Your Potential", "Solutions That Empower", "Seamless Loan Experience", 
        "Secure Your Dreams", "Freedom Through Finance", "Achieve More Today"
    ]
    tagline = random.choice(taglines)
    font = load_font("Helvetica.ttc", banner_height * 0.6)
    margin = 40
    # # Get text size for positioning
    text_bbox = draw.textbbox((0, 0), tagline, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Detect faces in the image for tagline positioning
    faces = detect_faces(img)
    if faces:  
        # Check if any faces are detected
        # Find face with the largest available space
        best_face = None
        max_space = 0
        for face in faces:
            left_space = face['x'] - margin
            right_space = img_width - (face['x'] + face['width']) - margin
            available_space = max(left_space, right_space)
            
            if available_space > max_space:
                max_space = available_space
                best_face = face

        # Calculate space on left and right
        left_space = best_face['x'] - margin # type: ignore
        right_space = img_width - (best_face['x'] + best_face['width']) - margin

        # Choose side with more space
        text_x = margin if left_space > right_space else img_width - text_width - margin

        # --- Fancy Styling (with transparent background) ---
        # 1. Split tagline into two lines (if needed)
        tagline_lines = tagline.split(' ')  # Split by spaces
        if len(tagline_lines) > 2:  # Adjust the threshold as needed
            tagline_line1 = ' '.join(tagline_lines[:len(tagline_lines)//2])
            tagline_line2 = ' '.join(tagline_lines[len(tagline_lines)//2:])
        else:
            tagline_line1 = tagline
            tagline_line2 = ""

        # 2. Calculate text dimensions for each line
        text_bbox1 = draw.textbbox((0, 0), tagline_line1, font=font)
        text_width1 = text_bbox1[2] - text_bbox1[0]
        text_height1 = text_bbox1[3] - text_bbox1[1]

        text_bbox2 = draw.textbbox((0, 0), tagline_line2, font=font)
        text_width2 = text_bbox2[2] - text_bbox2[0]

        # 3. Choose side with more space
        left_space = best_face['x'] - margin
        right_space = img_width - (best_face['x'] + best_face['width']) - margin
        text_x = margin if left_space > right_space else img_width - max(text_width1, text_width2) - margin

        # 4. Calculate y-coordinates for each line
        text_y1 = best_face['y'] + best_face['height'] // 2 - text_height1 - 5  # Above face, with spacing
        text_y2 = best_face['y'] + best_face['height'] // 2 + 5  # Below face, with spacing

        # 5. Draw text with outline
        outline_color = '#f26522'
        draw.text((text_x, text_y1), tagline_line1, fill='#f26522', font=font, stroke_width=1, stroke_fill=outline_color)
        draw.text((text_x, text_y2), tagline_line2, fill='#f26522', font=font, stroke_width=1, stroke_fill=outline_color)

    return new_image

def process_image_for_row(row, uploaded_logo):
    img, caption = save_genimage(row['Product'], row['age'], row['location'], 0, row['gender'], row['job'])
    width, height = img.size
    banner = create_banner(width=width, height=int(img.height * 0.08))
    image_with_tagline_and_logo = apply_tagline_and_logo(img, banner, uploaded_logo, logo_position="top_right")
    return image_with_tagline_and_logo, caption

def validate_csv_data(df):
    """Validate the CSV data format and content"""
    required_columns = ['Product', 'age', 'gender', 'job', 'location']
    
    # Check if all required columns exist
    if not all(col in df.columns for col in required_columns):
        return False
    
    # Validate data types and ranges
    try:
        # Check age range
        if not all(10 <= age <= 100 for age in df['age']):
            return False
        
        # Check gender values
        if not all(gender in ['male', 'female'] for gender in df['gender']):
            return False
        
        # Check for empty values
        if df[required_columns].isnull().any().any():
            return False
            
        return True
        
    except Exception:
        return False
        
def calculate_optimal_workers(total_images):
    """
    Calculate optimal number of workers based on number of images
    """
    if total_images <= 4:
        return total_images
    elif total_images <= 8:
        return 4
    elif total_images <= 16:
        return 6
    else:
        return 8  # Maximum workers to avoid API rate limits
        
def process_csv_data_with_parallel_progress(data, uploaded_logo, num_workers=None):
    # Reset DataFrame index to ensure continuous integers starting from 0
    data = data.reset_index(drop=True)
    
    # Set number of workers
    total_rows = len(data)
    if num_workers is None:
        num_workers = calculate_optimal_workers(total_rows)
    # Ensure num_workers doesn't exceed total rows
    num_workers = min(num_workers, total_rows)
    
    # Create communication queue
    message_queue = queue.Queue()
    
    # Create UI placeholders for progress tracking
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        worker_info = st.empty()
    
    # Display worker information
    worker_info.info(f"Processing {total_rows} images with {num_workers} parallel workers")
    
    # Create grid layout container
    grid_container = st.container()
    
    # Configure grid layout
    num_cols = 2  # Number of columns in the grid
    with grid_container:
        st.markdown("### Generated Advertisements")
        # Calculate number of rows needed
        num_rows = (total_rows + num_cols - 1) // num_cols
        
        # Create grid layout
        image_placeholders = {}
        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col in range(num_cols):
                idx = row * num_cols + col
                if idx < total_rows:
                    image_placeholders[idx] = cols[col]
    
    # Flag to track if we should stop processing
    stop_processing = threading.Event()
    
    # Dictionary to store all generated images and captions
    generated_images = {}
    
    def process_single_image(index, row):
        if stop_processing.is_set():
            return False
            
        try:
            # Generate image
            img, caption = save_genimage(
                row['Product'], 
                row['age'], 
                row['location'], 
                0, 
                row['gender'], 
                row['job']
            )
            
            # Create and apply banner and logo
            banner = create_banner(width=int(img.width), height=int(img.height * 0.08))
            final_image = apply_tagline_and_logo(img, banner, uploaded_logo, logo_position="top_right")
            
            # Send progress update
            message_queue.put(Message(
                type=MessageType.PROGRESS,
                data={'index': index}
            ))
            
            # Store the generated image and caption
            generated_images[index] = (final_image, caption)
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "Rate limit" in error_msg or "Model is busy" in error_msg:
                message_queue.put(Message(
                    type=MessageType.ERROR,
                    data={'index': index, 'error': "MODEL_BUSY"}
                ))
                stop_processing.set()  # Signal all threads to stop
            else:
                message_queue.put(Message(
                    type=MessageType.ERROR,
                    data={'index': index, 'error': str(e)}
                ))
            return False

    # Start processing in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(process_single_image, idx, row)
            for idx, row in data.iterrows()
        ]
        
        # Initialize tracking variables
        completed = 0
        
        # Process messages from queue until all tasks are complete or stop signal
        while (not all(future.done() for future in futures) or not message_queue.empty()) and not stop_processing.is_set():
            try:
                message = message_queue.get(timeout=0.1)
                
                if message.type == MessageType.ERROR and message.data['error'] == "MODEL_BUSY":
                    # Cancel all pending futures
                    for future in futures:
                        future.cancel()
                    
                    # Clean up UI elements
                    worker_info.empty()
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Show error message
                    st.error("Model too busy for process. Please try again later.")
                    return  # Exit the function immediately
                    
                elif message.type == MessageType.PROGRESS:
                    completed += 1
                    progress_bar.progress(completed / total_rows)
                    status_text.text(f"Generated {completed} of {total_rows} images")
                
            except queue.Empty:
                continue
        
        # Wait for all futures to complete
        for future in futures:
            future.result()  # Ensure all futures are done
        
        # Display all images at once
        for index, (img, caption) in generated_images.items():
            with image_placeholders[index]:
                st.image(img, caption=f"Image {index + 1}", use_container_width=True)
                st.caption(caption)
                
                # Add download button for each image
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label=f"Download Image {index + 1}",
                    data=byte_im,
                    file_name=f"ad_{index + 1}.png",
                    mime="image/png",
                    use_container_width=True,
                    key=f"download_{index}"
                )
                
    # Only show completion message if we didn't stop due to error
    if not stop_processing.is_set():
        worker_info.empty()
        status_text.empty()
        if completed == total_rows:
            st.success(f"Successfully generated all {completed} images!")
        else:
            st.warning(f"Completed with {completed} out of {total_rows} images generated.")
        
# Streamlit UI
st.set_page_config(page_title="Dynamic ADs Generation", page_icon="🎨")

st.header("Upload Logo in PNG")
uploaded_logo = st.file_uploader("Upload a Logo file", type=["png"])
processed_logo = process_logo_upload(uploaded_logo)
if processed_logo:
    tab1, tab2 = st.tabs(["🔣 Input", "🗃 Data"])
    
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
                                 options=["Home", "Personal", "Jewel", 
                                         "Car", "Education", "Credit Card"],
                                 help="Select the type of financial product")
    
        # Create a container for the generation process
        generation_container = st.container()
        
        with generation_container:
            if st.button("Generate Advertisement", type="primary", use_container_width=True):
                if age and gender and profession and location and product:
                    try:
                        # Show generation status
                        with st.status("Generating your advertisement...", expanded=True) as status:
                            st.write("🎨 Creating base image...")
                            img, caption = save_genimage(product, age, location, 0, gender, profession)
                            
                            st.write("✨ Adding banner and branding...")
                            banner = create_banner(width=int(img.width), height=int(img.height * 0.08))
                            final_image = apply_tagline_and_logo(img, banner, processed_logo, logo_position="top_right")
                            
                            st.write("✅ Finalizing advertisement...")
                            status.update(label="Advertisement generated successfully!", state="complete")
    
                        # Show the generated image in an expander
                        with st.expander("Generated Advertisement", expanded=True):
                            st.image(final_image, caption=caption, use_container_width=True)
                            
                            # Add download button for the image
                            # Convert image to bytes
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
                            
                            # Add regenerate button
                            if st.button("Generate Another Version", use_container_width=True):
                                st.rerun()
                    
                    except Exception as e:
                        st.error(f"Error generating image: {str(e)}")
                        st.button("Try Again", use_container_width=True, on_click=st.rerun)
                else:
                    st.error("Please fill in all fields before generating the advertisement.")
        
            # Add helpful information
            with st.expander("Tips for Better Results"):
                st.markdown("""
                ### 👉 Tips for Better Results:
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
        df = None
        # Create two columns for input fields
        col1, col2 = st.columns(2)
        
        with col1:
            # File upload section
            uploaded_file = st.file_uploader(
                "Upload CSV File", 
                type=['csv'],
                help="Upload a CSV file with columns: Product, age, gender, job, location"
            )
            # Show sample format
            with st.expander("View CSV Format"):
                st.markdown("""
                Your CSV should have the following columns:
                - Product: Type of financial product
                - age: Age of target customer (18-100)
                - gender: Male/Female
                - job: Customer profession
                - location: City name
                """)
                
                # Show sample data
                st.markdown("**Sample Data:**")
                st.code("""Product,age,gender,job,location
    Home Loan,35,Male,Software Engineer,Mumbai
    Personal Loan,28,Female,Doctor,Delhi
    Business Loan,45,Male,Entrepreneur,Bangalore""")

        with col2:
            # Preview section (only shown when file is uploaded)
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                try:
                    st.write("Data Preview:")
                    st.dataframe(df.head(3), use_container_width=True)
                    
                    # Show data statistics
                    with st.expander("Data Summary"):
                        st.write(f"Total Rows: {len(df)}")
                        st.write("Product Distribution:")
                        st.dataframe(df['Product'].value_counts().head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
                    
        if uploaded_file is not None:
            # Generation controls in a separate container using full width
            st.divider()
            
            # Create a container for generation controls
            with st.container():
                col3, col4 = st.columns(2)
                
                with col3:
                    num_workers = st.slider(
                        "Number of Parallel Workers",
                        min_value=1,
                        max_value=min(10, len(df)),
                        value=min(4, len(df)),
                        help="Adjust the number of parallel processes for image generation"
                    )
                
                with col4:
                    batch_size = st.number_input(
                        "Batch Size",
                        min_value=1,
                        max_value=len(df),
                        value=min(10, len(df)),
                        help="Number of random images to generate in one batch"
                    )
                
                # Generation button using full width
                if st.button("Generate Advertisements", type="primary", use_container_width=True):
                    # Validate data
                    if validate_csv_data(df):
                        try:
                            # Full width container for generation status and images
                            status_container = st.empty()
                            with status_container.status("Generating advertisements...", expanded=True) as status:
                                st.write(f"🎨 Preparing to generate random {batch_size} advertisements...")
                                
                                # Process the data with parallel execution
                                process_csv_data_with_parallel_progress(
                                    df.sample(batch_size), 
                                    processed_logo,
                                    num_workers=num_workers
                                )
                        
                        except Exception as e:
                            st.error(f"Error during generation: {str(e)}")
                            if st.button("Try Again", use_container_width=True):
                                st.rerun()
                    else:
                        st.error("Please ensure your CSV has all required columns with valid data.")
                        st.info("Check the 'View CSV Format' section for the required format.")

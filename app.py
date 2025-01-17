from gettext import install
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
from datetime import datetime
import face_recognition
import regex as re
from huggingface_hub import InferenceClient
import numpy as np
import pandas as pd
import time, asyncio
import random
import streamlit as st

 
KEY = st.secrets['HF_TOKEN'] 
client = InferenceClient(api_key=KEY)
 
def create_banner(width=1200, height=120):
    # Create a new image with transparent background
    image = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    background = Image.new('RGBA', (width, height), (242, 101, 34, 255))
    image.paste(background, (0, 0), background)

    draw = ImageDraw.Draw(image)
    font_size = image.height * 0.25 

    try:
        font = ImageFont.truetype("Helvetica.ttc", font_size)
    except:
        default_font = ImageFont.load_default()
        font = default_font.font_variant(size=font_size)


    texts = ["Low Interest Rates",'|', "Hassle Free process",'|', "Flexible tenure"]

    # 1. Text Boxes at the top
    box_height = int(height * 0.4)
    box_width = int(width / len(texts))
    spacing = (width - (box_width * len(texts))) // len(texts) + 1

    for i, text in enumerate(texts):
        x1 = spacing + (i * (box_width + spacing))
        y1 = int(height * 0.02) # small padding from top

        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x1 + (box_width - text_width) // 2
        text_y = y1 + (box_height - text_height) // 2

         # Draw text in white
        draw.text((text_x, text_y), text, fill='white', font=font, stroke_width=0.2, stroke_fill='white') #Bold effect using stroke


    # 2. "Follow us on" and Social Media Icons in the middle
    follow_text = "Follow us on:"
    follow_font_size = int(font_size * 0.85)
    try:
        follow_font = ImageFont.truetype("Helvetica.ttc", follow_font_size)
    except:
         default_font = ImageFont.load_default()
         follow_font = default_font.font_variant(size=follow_font_size)

    follow_text_bbox = draw.textbbox((0, 0), follow_text, font=follow_font)
    follow_text_width = follow_text_bbox[2] - follow_text_bbox[0]

    # Middle position
    icon_paths = ["facebook.png", "twitter.png", "instagram.png", "linkedin.png", "youtube.png", "whatsapp.png"]
    icons = []
    for path in icon_paths:
        if os.path.exists(path):
          icon = Image.open(path)
          if icon.mode != 'RGBA':
            icon = icon.convert('RGBA')
          icons.append(icon)
    #icon spacing      
    follow_text_x = int((width) - (follow_text_width * 2.8))
    follow_text_y = int(height * 0.55) # Start slightly below the text boxes
    icon_size = int(follow_font_size*1.2)
    icons = [icon.resize((icon_size, icon_size)) for icon in icons]

    icon_x = follow_text_x + follow_text_width + 5
    icon_y = follow_text_y + (follow_font_size - icon_size) // 2

    for icon in icons:
        image.paste(icon, (int(icon_x), int(icon_y)), icon)
        icon_x += icon.width + 10
    

    draw.text((follow_text_x, follow_text_y), follow_text, fill='white', font=follow_font, stroke_width=0.2, stroke_fill='white')


    # 3. Disclaimer at the bottom
    disclaimer_text = "This image was generated using artificial intelligence and may not depict real people, places, or events.\
    Any resemblance to actual individuals or situations is purely coincidental."
    disclaimer_text = disclaimer_text.replace('  ', ' ')

    disclaimer_font_size = int(font_size * 0.6)
    try:
        disclaimer_font = ImageFont.truetype("Helvetica.ttc", disclaimer_font_size)
    except:
        default_font = ImageFont.load_default()
        disclaimer_font = default_font.font_variant(size=disclaimer_font_size)

    disclaimer_text_bbox = draw.textbbox((0, 0), disclaimer_text, font=disclaimer_font)
    disclaimer_text_width = disclaimer_text_bbox[2] - disclaimer_text_bbox[0]
    disclaimer_text_x = (width - disclaimer_text_width) // 2
    disclaimer_text_y = height - disclaimer_font_size - 5  # 5 pixels padding from bottom

    draw.text((disclaimer_text_x, disclaimer_text_y), disclaimer_text, fill='white', font=disclaimer_font, stroke_width=0.4, stroke_fill='white')


    return image

def detect_faces(image):
        """
        Detect faces in an image and return their coordinates
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            list: List of dictionaries containing face coordinates
        """
        model="hog"
        # Detect faces
        face_locations = face_recognition.face_locations(image, model=model)
        
        # Convert to more readable format
        faces = []
        for face_location in face_locations:
            top, right, bottom, left = face_location
            faces.append({
                'x': left,
                'y': top,
                'width': right - left,
                'height': bottom - top
            })
        
        return faces
 
def save_genimage(product, age, location, income, gender, profession):
    """Create and save the banner"""
    
    if product == 'jewel':
        product = 'jewellery'
    elif product == 'personal':
        product = 'vacation'
  
    system_prompt = f"Photo taken with an iPhone 16 Pro Max of a {age}-year-old {gender} {profession} in {location}, India. The {product} (hidden logo) is in the foreground, fully in focus beside the person.\
  Realistic lighting, natural daylight, warm tones, and soft shadows emphasize the product’s sleek design and the subject’s professionalism. Lifestyle setting, no text, mid-shot, clean composition with cinematic framing."
    system_prompt = system_prompt.replace('  ', ' ')
    system_prompt = system_prompt.replace('  ', ' ')
    print(system_prompt)
 
    # output is a PIL.Image object
    client = InferenceClient("black-forest-labs/FLUX.1-dev", token=KEY)
 
    try:
      
       image = client.text_to_image(system_prompt)
        
    except Exception as e:
            raise st.error(f"Error Generating Image: {str(e)}")
            st.session_state.is_waiting = False
            #raise RuntimeError(f"Error adding image: {str(e)}")
 
    return image
 
def apply_tagline_and_logo(image, banner_output_path, logo_file, logo_position="top_left"):
    """Applies tagline and logo to the image."""
    # Load and resize logo
    logo = Image.open(logo_file)
    logo_width = int(image.width * 0.2 * 1.5)  # 20% of image width
    logo_height = int(logo.height * (logo_width / logo.width) * 1.2)
    logo = logo.resize((logo_width, logo_height))
 
    # Calculate logo position
    if logo_position == "top_left":
        logo_x = 10
        logo_y = 10
    elif logo_position == "top_right":
        logo_x = image.width - logo_width - 10
        logo_y = 10
    else:
        raise ValueError("Invalid logo_position. Choose 'top_left' or 'top_right'.")
    if logo.mode == 'RGBA':
     mask = logo.split()[3]
     rgb_logo = logo.convert('RGB')
     image.paste(rgb_logo, (logo_x,logo_y), mask)# Use mask for transparency if logo has it
    else:
     rgb_logo = logo.convert('RGB')
     # Paste banner onto image
     image.paste(rgb_logo, (logo_x,logo_y)) 
 
    # Load and resize banner
    banner = Image.open(banner_output_path)
    banner_width = int(image.width)  # 100% of image width
    banner_height = int(banner.height * (banner_width / banner.width))
    banner = banner.resize((banner_width, banner_height))
 
    # Calculate banner position
 
    # Create a new blank image with the calculated dimensions
    new_image = Image.new('RGB', (image.width, image.height + banner_height), color=(255, 255, 255))
 
    # Paste the first image at the top
    new_image.paste(image, (0, 0))
 
    # Paste the second image at the bottom
    new_image.paste(banner, (0, image.height))

    # Load image
    img = new_image
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Get image dimensions
    img_width, img_height = img.size
    
    # Create drawing object
    draw = ImageDraw.Draw(img)
    
    # Set font size
    FONT_SIZE = banner_height * 0.6
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except:
        # Create a larger default font
        font = ImageFont.load_default()
        # Scale it up to match the desired size
        font = font.font_variant(size=FONT_SIZE) # type: ignore
    tagline = ["Dreams Within Reach"
              ,"Empowering Your Goals"
              ,"Finance Your Future"
              ,"Simplify Your Tomorrow"
              ,"Loans Made Easy"
              ,"Borrow With Confidence"
              ,"Unlock New Possibilities"
              ,"Invest In You"
              ,"Quick, Easy Loans"
              ,"Grow Your Potential"
              ,"Solutions That Empower"
              ,"Seamless Loan Experience"
              ,"Secure Your Dreams"
              ,"Freedom Through Finance"
              ,"Achieve More Today"]
    tagline = random.choice(tagline)
    margin = 40
    # # Get text size for positioning
    text_bbox = draw.textbbox((0, 0), tagline, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Get face locations 
    faces = detect_faces(np.array(img))
    
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
    
  
    return img
 
if __name__ == "__main__":
 
 
    # Streamlit App
    st.title("Dynamic Image Generation App")
  
    # Logo Upload Section
    st.header("Upload Logo in PNG")
    uploaded_logo = st.file_uploader("Upload a Logo file", type=["png"])

    if uploaded_logo:
     # Input Section
     st.header("Generate Image Based on Inputs")
     age = st.text_input("Age")
     gender = st.text_input("Gender")
     profession = st.text_input("Profession")
     location = st.text_input("Location")
     product = st.text_input("Product")
     income=0
     
     if st.button("Generate Image"):
      
         # Simulate a process with different stages
         stages = ["Generating Image...", "Creating Banner & Applying Logo...", "Placing Tagline & Finalizing results..."]
         if age and gender and profession and location and product:
             st.write(f"🔄 {stages[0]}")
             img = save_genimage(product,age,location,income,gender,profession)
             #img.save(f"imageout\\output_image_data{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
             # img = Image.open("imageout\\output_image.png")
             st.write(f"🔄 {stages[1]}")
             banner_output_path = "banner.png"
             banner = create_banner(height = int(img.height * 0.08), width = int(img.width))
             # banner = create_banner()
             #banner.save(banner_output_path)
             print(f"Banner saved as {banner_output_path}")
             st.write(f"🔄 {stages[2]}")
             image = apply_tagline_and_logo(img, banner_output_path, uploaded_logo, logo_position="top_right")
             output_path = f"final_output\\output_image_data{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
             #image.save(output_path)
             print(f"Image saved as {output_path}")
             st.image(image, caption=f"{product} loan::{age}-year-old {gender} {profession} professional in {location}", use_container_width=True)
             st.success("All done!")
         else:
             st.error("Please fill all fields to generate an image.")
  
    
     # CSV Upload Section
     st.header("Generate Images from CSV")
     uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
  
     if uploaded_file:
         try:
             data = pd.read_csv(uploaded_file)
             if st.button("Generate Images from CSV"):
                 # Ensure required columns are present
                 required_columns = ['age', 'gender', 'job', 'location', 'Product']
                 if all(col in data.columns for col in required_columns):
                     # Pick 5 random rows
                     sampled_data = data.sample(5)
                     st.write("Uploaded CSV Data Preview:")
                     st.write(sampled_data.head(5))
                     for idx, row in sampled_data.iterrows():
                         # Simulate a process with different stages
                         stages = ["Generating Image...", "Creating Banner & Applying Logo...", "Placing Tagline & Finalizing results..."]
                         st.write(f"🔄 {stages[0]}")
                         img = save_genimage(
                             row['Product'], row['age'], row['location'], 0, row['gender'], row['job'],  
                         )
                         #img.save(f"imageout\\output_image_data{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
 
                         # Create and save the banner
                         st.write(f"🔄 {stages[1]}")
                         banner_output_path = "banner.png"
                         banner = create_banner(height = int(img.height * 0.08), width = int(img.width))
                         # banner = create_banner()
                         #banner.save(banner_output_path)
                         print(f"Banner saved as {banner_output_path}")
                         st.write(f"🔄 {stages[2]}")
                         image = apply_tagline_and_logo(img, banner_output_path, uploaded_logo, logo_position="top_right")
                         output_path = f"final_output\\output_image_{idx}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                         st.image(image, caption=f"{row['Product']} loan::{row['age']}-year-old {row['gender']} {row['job']} professional in {row['location']}", use_container_width =True)
                         #image.save(output_path)
                         #print(f"Image saved as {output_path}")
                     st.success("All done!")
                 else:
                     st.error(f"CSV must contain columns: {', '.join(required_columns)}")
         except Exception as e:
             st.error(f"Error reading CSV file: {e}")

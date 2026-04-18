import base64
from io import BytesIO
from wordcloud import WordCloud
from loguru import logger

def generate_wordcloud_base64(text_list: list):
    """
    Industrial Practice: Optimized image generation.
    Generates a word cloud using PIL directly, removing the huge matplotlib 
    dependency (~30MB) from the production image.
    """
    try:
        if not text_list:
            return None
            
        combined_text = " ".join(text_list)
        if not combined_text.strip():
            return None
            
        wc = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(combined_text)
        
        # Convert wordcloud to PIL image and save to buffer
        img = wc.to_image()
        img_buffer = BytesIO()
        img.save(img_buffer, format='PNG')
        
        img_buffer.seek(0)
        base64_img = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_img}"
    except Exception as e:
        logger.error(f"Error generating wordcloud: {e}")
        return None

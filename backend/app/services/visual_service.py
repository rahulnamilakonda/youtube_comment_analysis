import base64
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from loguru import logger

def generate_wordcloud_base64(text_list: list):
    """Generates a word cloud from a list of strings and returns it as a base64 string."""
    try:
        if not text_list:
            return None
            
        combined_text = " ".join(text_list)
        if not combined_text.strip():
            return None
            
        wc = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(combined_text)
        
        # Save to buffer
        img_buffer = BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_buffer, format='png')
        plt.close()
        
        img_buffer.seek(0)
        base64_img = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{base64_img}"
    except Exception as e:
        logger.error(f"Error generating wordcloud: {e}")
        return None

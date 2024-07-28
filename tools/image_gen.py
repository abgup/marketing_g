import os
import json
import requests
from crewai_tools import BaseTool
from dotenv import load_dotenv

load_dotenv()

class OpenAIImageGenTool(BaseTool):
    name: str = "OpenAI Image Generation Tool"
    description: str = """Useful for generating images based on text prompts using OpenAI's DALL-E API."""

    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

    def _run(self, prompt: str, size: str = "1024x1024", num_images: int = 1, save_path: str = None):
        """
        Generate images based on the given prompt.
        
        :param prompt: The text prompt to generate images from.
        :param size: Size of the generated images. Options: "256x256", "512x512", or "1024x1024".
        :param num_images: Number of images to generate (1-10).
        :param save_path: Optional path to save the generated images.
        :return: JSON string containing the URLs of the generated images and local file paths if saved.
        """
        url = "https://api.openai.com/v1/images/generations"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "prompt": prompt,
            "n": num_images,
            "size": size
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Failed to generate images: {response.text}")
        
        result = response.json()
        image_urls = [img['url'] for img in result['data']]
        
        output = {
            "image_urls": image_urls,
            "local_paths": []
        }
        
        if save_path:
            output["local_paths"] = self._save_images(image_urls, save_path)
        
        return json.dumps(output, indent=2)

    def _save_images(self, image_urls, save_path):
        local_paths = []
        for i, url in enumerate(image_urls):
            response = requests.get(url)
            if response.status_code == 200:
                # Create the directory if it doesn't exist
                os.makedirs(save_path, exist_ok=True)
                
                file_path = os.path.join(save_path, f"generated_image_{i+1}.png")
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                local_paths.append(file_path)
            else:
                print(f"Failed to download image {i+1}")
        
        return local_paths
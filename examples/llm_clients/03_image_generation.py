import asyncio
import base64
from kavalai.llm_clients.llm_client import LLMClient


async def basic_image_generation():
    """
    Demonstrates basic image generation using DALL-E 3.
    The result is a base64 encoded string of the image.
    """
    print("\n--- Basic Image Generation (OpenAI) ---")
    client = LLMClient(model="openai/dall-e-3")
    prompt = "A futuristic city with flying cars and neon lights, cyberpunk style."

    try:
        # generate_image returns a tuple of (base64_image, stats)
        img_b64, stats = await client.generate_image(prompt=prompt, size="1024x1024")

        # Save the image to a file
        with open("generated_image_openai.png", "wb") as f:
            f.write(base64.b64decode(img_b64))

        print("Image generated and saved to 'generated_image_openai.png'.")
        print(f"Stats: Cost: ${stats.cost:.4f}, Duration: {stats.duration:.2f}s")
    except Exception as e:
        print(f"OpenAI Image generation failed: {e}")


async def gemini_image_generation():
    """
    Demonstrates image generation using Gemini (Imagen).
    """
    print("\n--- Image Generation (Gemini Imagen) ---")
    client = LLMClient(model="gemini/imagen-3.0-generate-001")
    prompt = "A serene landscape with a mountain lake at sunset, digital art style."

    try:
        img_b64, stats = await client.generate_image(prompt=prompt)

        with open("generated_image_gemini.png", "wb") as f:
            f.write(base64.b64decode(img_b64))

        print("Image generated and saved to 'generated_image_gemini.png'.")
        print(f"Stats: Cost: ${stats.cost:.4f}, Duration: {stats.duration:.2f}s")
    except Exception as e:
        print(f"Gemini Image generation failed: {e}")


async def main():
    # Set your API keys in environment variables:
    # os.environ["OPENAI_API_KEY"] = "..."
    # os.environ["GEMINI_API_KEY"] = "..."

    await basic_image_generation()
    await gemini_image_generation()


if __name__ == "__main__":
    asyncio.run(main())

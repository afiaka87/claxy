import base64
import logging
import os
import random
from io import BytesIO

import discord
import httpx
import openai
import python_weather
import torch
from discord.ext import commands

logging.basicConfig(level=logging.INFO)

import io

import numpy as np
import torch
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
from diffusers.models import UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection

DEVICE = torch.device("cuda:0")


openai.api_key = os.environ["OPENAI_API_KEY"]

bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())


@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")


class ChatGPTDiscordClient(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    async def get_temperature(self):
        async with python_weather.Client(unit=python_weather.IMPERIAL) as weather_client:
            current_weather = await weather_client.get('Fayetteville, AR')
            print(current_weather.current.temperature)
            return current_weather.current.temperature
    
    async def run_krazinsky(
        self,
        prompt,
        negative_prior_prompt="",
    ):
        # random seed
        seed = np.random.randint(0, 100000)
        print(f"Seed: {seed}")
        torch.manual_seed(seed)

        # CLIP Image Encoder
        image_encoder = (
            CLIPVisionModelWithProjection.from_pretrained(
                "kandinsky-community/kandinsky-2-2-prior", subfolder="image_encoder"
            )
            .half()
            .to(DEVICE)
        )

        # Kandinsky Prior (unCLIP)
        unclip_prior = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
        ).to(DEVICE)

        # Predict an image embed from the prompt
        img_emb = unclip_prior(
            prompt=prompt,
            num_inference_steps=50,
            num_images_per_prompt=1,
        )

        # Predict a negative image embed from the negative prompt
        negative_emb = unclip_prior(
            prompt=negative_prior_prompt,
            num_inference_steps=50,
            num_images_per_prompt=1,
        )

        # remove the prior from vram
        image_encoder.to("cpu")
        del image_encoder

        # Kandinsky UNet
        unet = (
            UNet2DConditionModel.from_pretrained(
                "kandinsky-community/kandinsky-2-2-decoder", subfolder="unet"
            )
            .half()
            .to(DEVICE)
        )

        # Kandinsky Decoder 
        decoder = KandinskyV22Pipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder",
            unet=unet,
            torch_dtype=torch.float16,
        ).to(DEVICE)

        # Predict a pixel-space image from the (predicted) CLIP image embeds
        images = decoder(
            image_embeds=img_emb.image_embeds,
            negative_image_embeds=negative_emb.image_embeds,
            num_inference_steps=75,
            height=512,
            width=512,
        )

        # remove the decoder from vram
        unet.to("cpu")
        del unet

        output = images.images[0]  # first image is a PIL image
        # convert to bytes
        buffered = BytesIO()
        output.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return io.BytesIO(base64.b64decode(img_str))

    # TODO refactor to stateless, non-asynchronous string splitting
    async def split_response_and_send(self, message, assistant_response):
        if len(assistant_response) > 2000 and len(assistant_response) < 4000:
            print("Splitting response into 2 2000 character chunks")
            await message.channel.send(assistant_response[:2000])
            await message.channel.send(assistant_response[2000:])
        elif len(assistant_response) > 4000 and len(assistant_response) < 6000:
            print("Splitting response into 3 2000 character chunks")
            await message.channel.send(assistant_response[:2000])
            await message.channel.send(assistant_response[2000:4000])
            await message.channel.send(assistant_response[4000:])
        elif len(assistant_response) > 6000 and len(assistant_response) < 8000:
            print("Splitting response into 4 2000 character chunks")
            await message.channel.send(assistant_response[:2000])
            await message.channel.send(assistant_response[2000:4000])
            await message.channel.send(assistant_response[4000:6000])
            await message.channel.send(assistant_response[6000:])
        else:
            print("Response is less than 2000 characters")
            await message.channel.send(assistant_response)
        return

    async def on_ready(self):
        print("Logged in as")
        print(self.user.name)
        print(self.user.id)
        print("------")
        self.message_history = []

    # TODO re-implement help
    # async def on_help(self, message):
    #     await message.channel.send(HELP_MESSAGE)
    #     return

    async def on_system_message(self, message):
        system_prompt = message.content.replace("!system", "").strip()
        # add the system prompt to the message history
        if (
            len(self.message_history) > 0
            and self.message_history[0]["role"] == "system"
        ):
            self.message_history[0] = {
                "role": "system",
                "content": system_prompt,
            }
        else:
            self.message_history.insert(0, {"role": "system", "content": system_prompt})
        print(f"System prompt: {system_prompt}")
        print(f"Message history: {len(self.message_history)} messages")
        # let the user know we got the prompt
        await message.channel.send("Updated system prompt.")
        return

    async def get_openai_response(self, message):
        if message.content.strip().startswith("%"):
            gpt_model_name = "gpt-4-0613"
        else:
            gpt_model_name = "gpt-3.5-turbo-0613"
        user_prompt = message.content.replace("!", "").strip()

        # prepend the author's name to the message
        self.message_history.append(
            {"role": "user", "content": f"{message.author.name}: {user_prompt}"}
        )

        print(f"User prompt: {user_prompt}")
        print(f"Message history: {len(self.message_history)} messages")

        # Configure your OpenAI API call using httpx
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        }
        data = {"model": gpt_model_name, "messages": self.message_history}

        timeout = httpx.Timeout(connect=5, read=None, write=5, pool=5)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=data, headers=headers)
            response.raise_for_status()
            return response.json()

    async def on_message(self, message: discord.Message):
        if message.content.strip().startswith("!temp"):
            temp = await self.get_temperature()
            await message.channel.send(f"Current temperature in Fayetteville, AR: {temp} F")
            return
        
        if message.content.strip().startswith("!cat"):
            await message.channel.send("https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif")
            return

        if message.content.strip().startswith("!eval"):
            # parse message and calculate
            expression = message.content.strip().replace("!eval", "").strip()
            print(f"Calculating: {expression}")
            try:
                # eval the expression
                result = eval(expression)
                await message.channel.send(f"Result: {result}")
            except Exception as e:
                await message.channel.send(f"Error: {e}")
            return

        if message.content.strip().startswith("!random"):
            # generate a random number
            number = random.randint(0, 1000000)
            await message.channel.send(f"Random number: {number}")
            return


        # if "sd" (whitespace separated) is a word in the message, we send it to the sd api
        if message.content.strip().lower().startswith("kr"):
            await self.on_kr(message)
            return

        # use the message as system role prompt
        if message.content.strip().startswith("!system"):
            await self.on_system_message(message)
            return

        HELP_MESSAGE = """
        **Claxy Help**
        **!system** - set the system prompt
        **!eval** - evaluate a mathematical expression
        **!random** - generate a random number
        **!temp** - get the current temperature in Fayetteville, AR
        **!cat** - get a random cat gif
        **!kr** - generate a Krazinsky image
        **!help** - show this help message
        """
        if message.content.strip().startswith("!help"):
            await message.channel.send(HELP_MESSAGE)
            return

        elif message.content.strip().startswith(
            "!"
        ) or message.content.strip().startswith("%"):
            output = await self.get_openai_response(message)
            assistant_response = output["choices"][0]["message"]["content"]
            self.message_history.append(
                {"role": "assistant", "content": assistant_response}
            )
            print(f"Assistant response: {len(assistant_response)}")
            # split the response into 2000 character chunks
            await self.split_response_and_send(message, assistant_response)
            return

    async def on_kr(self, message: discord.Message):
        # we just want the last line starting with sd
        prompt = message.content.strip()[2:]  # remove kr
        # run krazinsky
        output_image = await self.run_krazinsky(prompt)
        print(output_image)
        # send as
        await message.channel.send(file=discord.File(output_image, "krazinsky.png"))


if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True

    client = ChatGPTDiscordClient(intents=intents)
    client.run(os.environ["DISCORD_API_TOKEN"])

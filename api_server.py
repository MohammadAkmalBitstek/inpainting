"""
Inpainting API Server
Wraps your ComfyUI workflow running on RunPod
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import httpx
import base64
import json
import uuid
import time
import asyncio
import os
from pathlib import Path
import logging
import traceback
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Storybook Inpainting API", version="1.0.0")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
   # used only if your pod has auth enabled
RUNPOD_BASE_URL = os.getenv("RUNPOD_BASE_URL")
RUNPOD_API_KEY  = os.getenv("RUNPOD_API_KEY")

HEADERS = {"Authorization": f"Bearer {RUNPOD_API_KEY}"} if RUNPOD_API_KEY else {}
# ──────────────────────────────────────────────────────────────────────────────


def build_workflow(masked_image_name: str, prompt: str) -> dict:
    """
    Builds the ComfyUI API-format workflow from your inpainting workflow.json
    The key nodes are:
      - Node 20 (LoadImage)  → receives the uploaded masked image
      - Node 21 (CLIPTextEncode) → positive prompt
      - Node 16 (LanPaint_KSampler) → sampler settings
    """
    seed = int(time.time())

    return {
        "prompt": {
            # ── Model Loaders ─────────────────────────────────────────────
            "3": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "qwen_3_4b.safetensors",
                    "type": "lumina2",
                    "device": "default"
                }
            },
            "4": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "ae.safetensors"}
            },
            "22": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": "z_image_turbo_bf16.safetensors",
                    "weight_dtype": "default"
                }
            },
            "12": {
                "class_type": "ModelPatchLoader",
                "inputs": {
                    "name": "Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors"
                }
            },

            # ── Image Input (uploaded image) ──────────────────────────────
            "20": {
                "class_type": "LoadImage",   # standard load image node
                "inputs": {"image": masked_image_name}
            },

            # ── Prompt ────────────────────────────────────────────────────
            "21": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["3", 0]
                }
            },

            # ── Negative Prompt ───────────────────────────────────────────
            "15": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "lowres, blurry, out of focus, noise, grain, jpeg artifacts, distorted, deformed, disfigured, bad anatomy, bad proportions, extra limbs, extra fingers, missing fingers, fused fingers, mutated hands, poorly drawn hands, poorly drawn face, unnatural pose, cropped, out of frame, cut off, watermark, text, logo, signature, username, oversaturated, undersaturated, bad lighting, harsh shadows, duplicate, cloned face, tiling, ugly, messy background, bad composition, inconsistent lighting, mismatched colors, unnatural blending",
                    "clip": ["3", 0]
                }
            },

            # ── Preprocessing ─────────────────────────────────────────────
            "2": {
                "class_type": "ModelSamplingAuraFlow",
                "inputs": {
                    "model": ["22", 0],
                    "shift": 3
                }
            },
            "8": {
                "class_type": "GetImageSize",
                "inputs": {"image": ["20", 0]}
            },
            "9": {
                "class_type": "ImageScaleToTotalPixels",
                "inputs": {
                    "image": ["20", 0],
                    "upscale_method": "lanczos",
                    "megapixels": 1,
                    "resolution_steps": 64
                }
            },
            "10": {
                "class_type": "ResizeMask",
                "inputs": {
                    "mask": ["20", 1],      # mask channel from LoadImage
                    "width": ["8", 0],
                    "height": ["8", 1],
                    "upscale_method": "lanczos",
                    "crop": "center",
                    "keep_proportions": False
                }
            },
            "19": {
                "class_type": "GrowMaskWithBlur",
                "inputs": {
                    "mask": ["10", 0],
                    "expand": 0,
                    "incremental_expandrate": 0,
                    "tapered_corners": True,
                    "flip_input": False,
                    "blur_radius": 14.9,
                    "lerp_alpha": 1,
                    "decay_factor": 1,
                    "fill_holes": True
                }
            },

            # ── VAE Encoding ──────────────────────────────────────────────
            "5": {
                "class_type": "VAEEncode",
                "inputs": {
                    "pixels": ["9", 0],
                    "vae": ["4", 0]
                }
            },
            "6": {
                "class_type": "VAEEncode",
                "inputs": {
                    "pixels": ["14", 0],
                    "vae": ["4", 0]
                }
            },
            "14": {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["20", 0],
                    "upscale_method": "lanczos",
                    "width": 512,
                    "height": 512,
                    "crop": "center"
                }
            },
            "11": {
                "class_type": "SetLatentNoiseMask",
                "inputs": {
                    "samples": ["5", 0],
                    "mask": ["19", 0]
                }
            },

            # ── ControlNet ────────────────────────────────────────────────
            "17": {
                "class_type": "ZImageFunControlnet",
                "inputs": {
                    "model": ["2", 0],
                    "model_patch": ["12", 0],
                    "vae": ["4", 0],
                    "image": ["9", 0],
                    "inpaint_image": ["9", 0],
                    "mask": ["19", 0],
                    "strength": 0.7
                }
            },

            # ── Sampler ───────────────────────────────────────────────────
            "16": {
                "class_type": "LanPaint_KSampler",
                "inputs": {
                    "model": ["17", 0],
                    "positive": ["21", 0],
                    "negative": ["15", 0],
                    "latent_image": ["11", 0],
                    "seed": seed,
                    "control_after_generate": "increment",
                    "steps": 15,
                    "cfg": 1,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1,
                    "preview_method": "Image First",
                    "vae_decode": "",
                    "mode": "🖼️ Image Inpainting",
                    "LanPaint_PromptMode": "Image First",
                    "LanPaint_NumSteps": 15,
                    "Inpainting_mode": "🖼️ Image Inpainting",
                    "LanPaint_Info": ""
                }
            },

            # ── Decode & Save ─────────────────────────────────────────────
            "1": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["16", 0],
                    "vae": ["4", 0]
                }
            },
            "13": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["1", 0],
                    "filename_prefix": "inpaint-result"
                }
            }
        },
        "client_id": str(uuid.uuid4())
    }


async def upload_image(image_bytes: bytes, filename: str) -> str:
    """Upload an image to ComfyUI and return the local filename."""
    async with httpx.AsyncClient(headers=HEADERS, timeout=60) as client:
        # ComfyUI requires the MIME type for the image
        files = {"image": (filename, image_bytes, "image/png")}
        data = {"overwrite": "true"}
        # Create a multipart/form-data request
        resp = await client.post(f"{RUNPOD_BASE_URL}/upload/image", files=files, data=data)
        
        if resp.status_code == 200:
            return resp.json()["name"]
        else:
            raise HTTPException(status_code=502, detail=f"Failed to upload image. Status {resp.status_code}: {resp.text}")


async def wait_for_result(prompt_id: str, timeout: int = 600) -> list:
    """Poll ComfyUI history until job is done, return list of output image filenames."""
    deadline = time.time() + timeout
    async with httpx.AsyncClient(headers=HEADERS, timeout=30) as client:
        while time.time() < deadline:
            try:
                resp = await client.get(f"{RUNPOD_BASE_URL}/history/{prompt_id}")              
                if resp.status_code == 200:
                    try:
                        history = resp.json()
                    except json.JSONDecodeError:
                        logger.warning(f"History polling: Received non-JSON response from {RUNPOD_BASE_URL}")
                        await asyncio.sleep(2)
                        continue

                    if prompt_id in history:
                        outputs = history[prompt_id].get("outputs", {})
                        images = []
                        for node_output in outputs.values():
                            for img in node_output.get("images", []):
                                images.append(img["filename"])
                        return images
                elif resp.status_code == 404:
                    # Job might not be in history yet, this is common
                    pass
                else:
                    logger.warning(f"History polling: Unexpected status {resp.status_code}")
            except Exception as e:
                logger.warning(f"History polling error: {str(e)}")
            
            await asyncio.sleep(2)
    raise TimeoutError("Workflow did not finish in time")


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Storybook Inpainting API is running 🎨"}


@app.post("/inpaint")
async def inpaint(
    image: UploadFile = File(..., description="Masked storybook page image (PNG with transparency for mask)"),
    prompt: str = Form(..., description="What to paint in the masked area")
):
    """
    Main inpainting endpoint.
    Send the masked image and a text prompt.
    Returns the inpainted image as base64.
    """
    import asyncio

    # Read file
    try:
        image_bytes = await image.read()
        logger.info(f"Received inpaint request with prompt: {prompt}")
        
        # Upload the image to ComfyUI
        logger.info("Uploading image to ComfyUI...")
        remote_image_name = await upload_image(image_bytes, f"masked_{uuid.uuid4()}.png")
        logger.info(f"Image uploaded successfully as {remote_image_name}")

        # Build workflow payload
        workflow = build_workflow(remote_image_name, prompt)
        client_id = workflow["client_id"]

        # Submit to ComfyUI on RunPod
        logger.info(f"Submitting workflow with client_id: {client_id}")
        async with httpx.AsyncClient(headers=HEADERS, timeout=60) as client:
            resp = await client.post(f"{RUNPOD_BASE_URL}/prompt", json=workflow)
            if resp.status_code != 200:
                logger.error(f"ComfyUI prompt submission failed: {resp.status_code} - {resp.text}")
                raise HTTPException(status_code=502, detail=f"ComfyUI error: {resp.text}")
            
            resp_data = resp.json()
            prompt_id = resp_data.get("prompt_id")
            if not prompt_id:
                raise HTTPException(status_code=500, detail="Did not receive prompt_id from ComfyUI.")
        
        logger.info(f"Workflow submitted successfully! Prompt ID: {prompt_id}. Polling for results...")

        # Wait for result
        try:
            result_filenames = await wait_for_result(prompt_id)
        except TimeoutError:
            logger.error(f"Workflow {prompt_id} timed out.")
            raise HTTPException(status_code=504, detail="Workflow timed out. Try again.")

        if not result_filenames:
            logger.error(f"Workflow {prompt_id} finished but returned no output images.")
            raise HTTPException(status_code=500, detail="No output images from workflow.")

        logger.info(f"Workflow completed. Output images: {result_filenames}")

        # Fetch the output image and return as base64
        logger.info(f"Fetching result image: {result_filenames[0]}")
        async with httpx.AsyncClient(headers=HEADERS, timeout=60) as client:
            img_url = f"{RUNPOD_BASE_URL}/view?filename={result_filenames[0]}&type=output"
            img_resp = await client.get(img_url)
            if img_resp.status_code != 200:
                logger.error(f"Failed to fetch result image: {img_resp.status_code}")
                raise HTTPException(status_code=502, detail="Failed to fetch result image from ComfyUI")
            result_b64 = base64.b64encode(img_resp.content).decode()

        logger.info("Inpainting request successful.")
        return JSONResponse({
            "status": "success",
            "result_image_base64": result_b64,
            "filename": result_filenames[0],
            "prompt_used": prompt
        })
    except HTTPException:
        # Re-raise HTTPExceptions as they are already handled
        raise
    except Exception as e:
        logger.error(f"Unexpected error during inpainting: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": f"An unexpected error occurred: {str(e)}", "traceback": traceback.format_exc()}
        )


@app.get("/health")
async def health():
    """Check if ComfyUI on RunPod is reachable."""
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=10) as client:
            resp = await client.get(f"{RUNPOD_BASE_URL}/system_stats")
        return {"status": "ok", "comfyui_reachable": resp.status_code == 200}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

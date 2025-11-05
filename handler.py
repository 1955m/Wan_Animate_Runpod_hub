import runpod
from runpod.serverless.utils import rp_upload
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii # Import for Base64 error handling
import subprocess
import time

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())
def save_data_if_base64(data_input, temp_dir, output_filename):
    """
    Check if input data is a Base64 string, and if so, save as file and return path.
    If it's a regular path string, return as is.
    """
    # If input is not a string, return as is
    if not isinstance(data_input, str):
        return data_input

    try:
        # Base64 strings will succeed when attempting to decode
        decoded_data = base64.b64decode(data_input)

        # Create directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)

        # If decoding succeeds, save as temporary file
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f: # Save in binary write mode ('wb')
            f.write(decoded_data)

        # Return the path of the saved file
        print(f"✅ Saved Base64 input to '{file_path}' file.")
        return file_path

    except (binascii.Error, ValueError):
        # If decoding fails, treat as regular path and return original value
        print(f"➡️ Processing '{data_input}' as file path.")
        return data_input
    
def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    logger.info(f"Getting image from: {url}")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}") as response:
        return response.read()

def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def get_videos(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_videos = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
        else:
            continue

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        videos_output = []
        if 'gifs' in node_output:
            for video in node_output['gifs']:
                # fullpath를 이용하여 직접 파일을 읽고 base64로 인코딩
                with open(video['fullpath'], 'rb') as f:
                    video_data = base64.b64encode(f.read()).decode('utf-8')
                videos_output.append(video_data)
        output_videos[node_id] = videos_output

    return output_videos

def load_workflow(workflow_path):
    with open(workflow_path, 'r') as file:
        return json.load(file)


def process_input(input_data, temp_dir, output_filename, input_type):
    """Function to process input data and return file path"""
    if input_type == "path":
        # If path, return as is
        logger.info(f"📁 Processing path input: {input_data}")
        return input_data
    elif input_type == "url":
        # If URL, download
        logger.info(f"🌐 Processing URL input: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        # If Base64, decode and save
        logger.info(f"🔢 Processing Base64 input")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"Unsupported input type: {input_type}")

        
def download_file_from_url(url, output_path):
    """Function to download file from URL"""
    try:
        # Download file using wget
        result = subprocess.run([
            'wget', '-O', output_path, '--no-verbose', url
        ], capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"✅ Successfully downloaded file from URL: {url} -> {output_path}")
            return output_path
        else:
            logger.error(f"❌ wget download failed: {result.stderr}")
            raise Exception(f"URL download failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("❌ Download timeout")
        raise Exception("Download timeout")
    except Exception as e:
        logger.error(f"❌ Error during download: {e}")
        raise Exception(f"Error during download: {e}")


def save_base64_to_file(base64_data, temp_dir, output_filename):
    """Function to save Base64 data to file"""
    try:
        # Decode Base64 string
        decoded_data = base64.b64decode(base64_data)

        # Create directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)

        # Save to file
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)

        logger.info(f"✅ Saved Base64 input to '{file_path}' file.")
        return file_path
    except (binascii.Error, ValueError) as e:
        logger.error(f"❌ Base64 decoding failed: {e}")
        raise Exception(f"Base64 decoding failed: {e}")

def calculate_dimensions_from_preset(video_width, video_height, preset):
    """
    Calculate optimal dimensions based on resolution preset.
    Maintains aspect ratio and rounds to multiples of 8.

    Args:
        video_width: Original video width (or assumed aspect ratio width)
        video_height: Original video height (or assumed aspect ratio height)
        preset: Resolution preset string ("480p", "720p", or "1080p")

    Returns:
        (width, height) tuple, rounded to multiples of 8
    """
    import math

    PRESET_PIXELS = {
        "480p": 399360,    # 832 × 480
        "720p": 921600,    # 1280 × 720
        "1080p": 2073600   # 1920 × 1080
    }

    target_pixels = PRESET_PIXELS.get(preset, 399360)
    aspect_ratio = video_width / video_height

    # Calculate dimensions maintaining aspect ratio
    new_width = math.sqrt(target_pixels * aspect_ratio)
    new_height = math.sqrt(target_pixels / aspect_ratio)

    # Round to multiple of 8 (required for video encoding)
    new_width = round(new_width / 8) * 8
    new_height = round(new_height / 8) * 8

    # Enforce reasonable limits (prevent extremely narrow/wide videos)
    new_width = max(480, min(2560, int(new_width)))
    new_height = max(480, min(1440, int(new_height)))

    return int(new_width), int(new_height)

def handler(job):
    job_input = job.get("input", {})

    logger.info(f"Received job input: {job_input}")
    task_id = f"task_{uuid.uuid4()}"


    image_path = None
    # Process image input (use only one of image_path, image_url, image_base64)
    if "image_path" in job_input:
        image_path = process_input(job_input["image_path"], task_id, "input_image.jpg", "path")
    elif "image_url" in job_input:
        image_path = process_input(job_input["image_url"], task_id, "input_image.jpg", "url")
    elif "image_base64" in job_input:
        image_path = process_input(job_input["image_base64"], task_id, "input_image.jpg", "base64")
    else:
        # Use default value
        image_path = "/examples/image.jpg"
        logger.info("Using default image file: /examples/image.jpg")

    video_path = None
    # Process video input (use only one of video_path, video_url, video_base64)
    if "video_path" in job_input:
        video_path = process_input(job_input["video_path"], task_id, "input_video.mp4", "path")
    elif "video_url" in job_input:
        video_path = process_input(job_input["video_url"], task_id, "input_video.mp4", "url")
    elif "video_base64" in job_input:
        video_path = process_input(job_input["video_base64"], task_id, "input_video.mp4", "base64")
    else:
        # Use default value (use default image when no video is available)
        video_path = "/examples/image.jpg"
        logger.info("Using default image file: /examples/image.jpg")

    check_coord = job_input.get("points_store", None)

    # Determine dimensions based on priority: explicit > preset > default
    if "width" in job_input and "height" in job_input:
        # Priority 1: Explicit dimensions provided (highest priority)
        width = job_input["width"]
        height = job_input["height"]
        logger.info(f"📐 Using explicit dimensions: {width}x{height}")
    elif "resolution_preset" in job_input:
        # Priority 2: Calculate from resolution preset
        preset = job_input["resolution_preset"]
        if preset not in ["480p", "720p", "1080p"]:
            raise Exception(f"Invalid resolution_preset: {preset}. Must be one of: 480p, 720p, 1080p")

        # Assume 16:9 aspect ratio (1920:1080) as default
        # ComfyUI's VHS_LoadVideo will handle actual video scaling based on these target dimensions
        video_width, video_height = 1920, 1080
        width, height = calculate_dimensions_from_preset(video_width, video_height, preset)
        logger.info(f"📐 Using resolution preset '{preset}': {width}x{height}")
    else:
        # Priority 3: Default to 480p
        width = 832
        height = 480
        logger.info(f"📐 Using default dimensions (480p): {width}x{height}")

    if check_coord == None:
        if job_input.get("mode", "replace") == "animate":
            prompt = load_workflow('/newWanAnimate_noSAM_animate_api.json')
        else:
            prompt = load_workflow('/newWanAnimate_noSAM_api.json')

        prompt["57"]["inputs"]["image"] = image_path
        prompt["63"]["inputs"]["video"] = video_path
        prompt["63"]["inputs"]["force_rate"] = job_input["fps"]
        prompt["30"]["inputs"]["frame_rate"] = job_input["fps"]
        prompt["65"]["inputs"]["positive_prompt"] = job_input["prompt"]
        prompt["27"]["inputs"]["seed"] = job_input["seed"]
        prompt["27"]["inputs"]["cfg"] = job_input["cfg"]
        prompt["27"]["inputs"]["steps"] = job_input.get("steps", 4)
        prompt["150"]["inputs"]["value"] = width
        prompt["151"]["inputs"]["value"] = height
    else:
        if job_input.get("mode", "replace") == "animate":
            prompt = load_workflow('/newWanAnimate_point_animate_api.json')
        else:
            prompt = load_workflow('/newWanAnimate_point_api.json')
        
        prompt["57"]["inputs"]["image"] = image_path
        prompt["63"]["inputs"]["video"] = video_path
        prompt["63"]["inputs"]["force_rate"] = job_input["fps"]
        prompt["30"]["inputs"]["frame_rate"] = job_input["fps"]
        prompt["65"]["inputs"]["positive_prompt"] = job_input["prompt"]
        prompt["27"]["inputs"]["seed"] = job_input["seed"]
        prompt["27"]["inputs"]["cfg"] = job_input["cfg"]
        prompt["27"]["inputs"]["steps"] = job_input.get("steps", 4)
        prompt["150"]["inputs"]["value"] = width
        prompt["151"]["inputs"]["value"] = height

        prompt["107"]["inputs"]["points_store"] = job_input["points_store"]
        prompt["107"]["inputs"]["coordinates"] = job_input["coordinates"]
        prompt["107"]["inputs"]["neg_coordinates"] = job_input["neg_coordinates"]
        # prompt["107"]["inputs"]["width"] = job_input["width"]
        # prompt["107"]["inputs"]["height"] = job_input["height"]
    

    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting to WebSocket: {ws_url}")

    # First check if HTTP connection is possible
    http_url = f"http://{server_address}:8188/"
    logger.info(f"Checking HTTP connection to: {http_url}")

    # Check HTTP connection (max 1 minute)
    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            import urllib.request
            response = urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP connection successful (attempt {http_attempt+1})")
            break
        except Exception as e:
            logger.warning(f"HTTP connection failed (attempt {http_attempt+1}/{max_http_attempts}): {e}")
            if http_attempt == max_http_attempts - 1:
                raise Exception("Cannot connect to ComfyUI server. Please check if the server is running.")
            time.sleep(1)
    
    ws = websocket.WebSocket()
    # Attempt WebSocket connection (max 3 minutes)
    max_attempts = int(180/5)  # 3 minutes
    for attempt in range(max_attempts):
        import time
        try:
            ws.connect(ws_url)
            logger.info(f"WebSocket connection successful (attempt {attempt+1})")
            break
        except Exception as e:
            logger.warning(f"WebSocket connection failed (attempt {attempt+1}/{max_attempts}): {e}")
            if attempt == max_attempts - 1:
                raise Exception("WebSocket connection timeout (3 minutes)")
            time.sleep(5)
    videos = get_videos(ws, prompt)
    ws.close()

    # Handle case when no video is found
    for node_id in videos:
        if videos[node_id]:
            return {"video": videos[node_id][0]}

    return {"error": "Video not found."}

runpod.serverless.start({"handler": handler})
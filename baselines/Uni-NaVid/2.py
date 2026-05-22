import requests

url = "[REDACTED_URL]"
sample = {
    "image": "examples/uav_sample/cur_view.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\n## Role\nYou are an expert navigation assistant for a UAV (Unmanned Aerial Vehicle) flight simulator.\n\n## Task Objective\nThe UAV operates in a 3D urban environment with visible roads, buildings, and landmarks.\nYour task is to predict the next sequence of UAV actions based on:\n1. A given natural language navigation instruction,\n2. The current state of the UAV, including its position and heading angle,\n3. The current top-down UAV view image,\n4. Up to four historical top-down view images from previous time steps (if available),\n5. The previously executed UAV actions (if available).\n\n## Text Input\n- **Navigation instruction**: Move forward and keep going straight, watching for a small light‑colored rectangular outbuilding tucked into a narrow garden plot beside a row of houses. Continue moving forward directly toward that little outbuilding, and when you arrive beside it, stop.\n- **Current state of the UAV**: [330.328723607643, 461.34559246293816, 60.02721669169363, -149.79479838175214] (x, y, z in meters; heading in degrees)\n\n## Image Input\n**UAV (Unmanned Aerial Vehicle) Top-Down View Sequence**\n- Historical top-down views (from oldest to newest) show the UAV’s past observations.\n- The last image is the current top-down view of the UAV.\n- In all images, the **top of the image corresponds to the UAV’s forward direction** (its heading).\n\nBased on the navigation instruction, the UAV’s current state, the previously executed actions (which can help infer the UAV’s current orientation and progress), and the provided images, predict how the UAV should move **step by step** to follow the instruction accurately.\n\n## Prediction Rules\n1. Predict no more than **8 future actions** for the UAV to execute.\n2. If the target location is reachable in fewer than 8 actions, output less than 8 actions sequence and end with **\"STOP\"**. Otherwise, it clearly requires more than 8 actions to approach the target, output exactly 8 future actions.\n3. You **must** output **\"STOP\"** if the UAV has already reached the described target.\n\n## Discrete Action Space\n- `STOP`: stop the flight\n- `MOVE_FORWARD`: move straight 5 meters in the current heading\n- `TURN_RIGHT`: rotate right 30°\n- `TURN_LEFT`: rotate left 30°"
        },
        {
            "from": "gpt",
            "value": "['MOVE_FORWARD', 'MOVE_FORWARD', 'MOVE_FORWARD', 'MOVE_FORWARD', 'MOVE_FORWARD', 'STOP']"
        }
    ]
}

prompt = sample["conversations"][0]["value"]
image = sample["image"]

response = requests.post(url, json={"prompt": prompt, "image": image}).json()
print(response)
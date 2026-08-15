# VLA Drone Navigation Pipeline — Technical Handbook

**Date**: August 14, 2026  
**Project**: Vision-Language-Action (VLA) Drone Navigation in AirSim  
**Author**: Auto-generated from project codebase and experiment history

---

## 1. Data Pipeline

### What Data Goes In

Each training sample consists of three things:

1. **Three camera images** (RGB, 224x224 pixels each):
   - Two older frames from recent flight history (what the drone saw a few seconds ago)
   - One current frame (what the drone sees right now)
   - These three images together give the model a sense of motion and depth — similar to how humans use recent visual memory to understand movement

2. **A natural language instruction** (text string):
   - A sentence describing where the drone should go, e.g., "Fly toward the group of houses on the left" or "Ascend and head toward the intersection"

3. **An 8-number action vector** (the "correct answer"):
   - Eight decimal numbers representing what the drone actually did at that moment
   - The dimensions are: `[stop, forward_distance, yaw_left, yaw_right, altitude_up, altitude_down, move_left, move_right]`
   - Example: `[0.0, 3.2, 7.5, 0.0, 0.4, 0.0, 0.0, 0.0]` means "move forward 3.2 meters, turn left 7.5 degrees, climb 0.4 meters"

### How the Data Is Collected

We use a **classical autopilot** (not the AI model) to fly the drone through AirSim's simulated neighborhood. This autopilot uses simple geometry — it knows where the waypoints are and steers toward them. At each step:

1. Capture three camera images from the drone
2. The autopilot decides where to fly next based on waypoint geometry
3. The drone moves
4. We measure the **actual displacement** (where the drone was before vs. after) to compute the true 8-number action vector
5. We save `(images, action_vector, instruction)` as one training sample

This produces a `manifest.jsonl` file (one JSON line per sample) plus folders of PNG images.

### How the Data Is Preprocessed for Training

When a sample is loaded for training, the following happens:

1. **Images** are opened from disk, converted to RGB, and passed through the model's built-in image processor (`PrismaticProcessor`). This resizes them and normalizes pixel values to the range the vision backbone expects. The result is a tensor of shape `(3, channels, 224, 224)` — three images stacked together.

2. **The instruction text** is tokenized into integer token IDs using the model's tokenizer (based on LLaMA's vocabulary). For example, "Fly forward" becomes something like `[1, 383, 6375, 5765]`.

3. **The action vector** is converted into 8 token IDs through a process called **binning**:
   - Each of the 8 numbers is first normalized to a range of [-1, 1] using known min/max bounds
   - The normalized value is then placed into one of 256 equally-spaced bins (like rounding to the nearest bucket)
   - Each bin maps to a specific token ID in the range 31744–31999
   - Example: a forward distance of 3.2m might map to token ID 31871

4. **The final training input** is assembled by concatenating:
   - The tokenized instruction (variable length)
   - A special trigger token (ID 29871)
   - The 8 action token IDs
   
   The training loss is only computed on the 8 action tokens — the instruction tokens are masked out with a special "ignore" value (-100) so the model only learns to predict actions, not repeat the instruction.

---

## 2. The Model

### Base Model: OpenFly-Agent-7B

The model is called **OpenFly-Agent-7B**, published by IPEC-COMMUNITY on HuggingFace. It has approximately **7.5 billion parameters**.

It was originally trained on the **OpenFly-Platform**, a large-scale synthetic environment with 100,000 aerial drone trajectories. The model learned to look at what a drone's camera sees, read a text instruction, and output the next movement the drone should make.

### Architecture (Three Main Parts)

The model has three connected components:

1. **Vision Backbone** (~2.5B parameters):
   Two image-understanding networks that run in parallel:
   - **DINOv2** (ViT-Large): Good at understanding the 3D structure and layout of a scene
   - **SigLIP** (ViT-SO400M): Good at connecting visual content to language concepts
   
   Each of the 3 input images is processed by both networks. The outputs are concatenated, producing a rich set of visual features that capture both spatial structure and semantic meaning.

2. **Projector** (~71M parameters):
   A small neural network (multi-layer perceptron) that translates the vision backbone's output into a format the language model can understand. Think of it as a "translator" between the visual world and the text world.

3. **Language Model** (~4B parameters, based on LLaMA-2-7B):
   A standard large language model that receives the translated visual features and the text instruction, then predicts the next tokens. In our case, those tokens represent the 8-number action vector.
   
   This includes the **lm_head** (~131M parameters) — the final layer that converts the language model's internal representation into a probability distribution over all possible tokens.

### How the Parts Connect

```
Camera Images (3x RGB)
       │
       ▼
 ┌─────────────┐
 │ DINOv2 ViT  │──┐
 └─────────────┘  │
                  ├──► Projector ──► Language Model (LLaMA) ──► lm_head ──► Action Tokens
 ┌─────────────┐  │          (translator)     (reasoning)        (output)
 │ SigLIP ViT  │──┘
 └─────────────┘

 Text Instruction ──────────────────► Language Model (LLaMA) ──►
                                       (same LLM receives both)
```

---

## 3. How the Model Is Used (Frozen vs. Trainable)

### The Problem

The base OpenFly model was trained on a different simulated environment (OpenFly-Platform). When placed in AirSim's Neighborhood environment, the visuals look different enough that the model cannot interpret the images properly. It collapses to outputting the same action every time regardless of what it sees — the drone flies in a straight line.

### The Solution: LoRA Fine-Tuning

Instead of retraining the entire 7.5B-parameter model from scratch (which would require enormous compute), we use a technique called **LoRA (Low-Rank Adaptation)**. LoRA adds small trainable "adapter" matrices to specific layers of the model, allowing it to learn new behavior while keeping most of the original weights unchanged.

### What Is Frozen (Not Updated During Training)

| Component | Parameters | Status | Why |
|-----------|-----------|--------|-----|
| Vision Backbone (DINOv2 + SigLIP) | ~2.5B | **Frozen** | Already knows how to extract image features — these are general-purpose visual representations that transfer well across environments |
| Most of the Language Model | ~3.8B | **Frozen** | Retains general reasoning and language understanding capabilities |

### What Is Trainable (Updated During Training)

| Component | Parameters | Status | Why |
|-----------|-----------|--------|-----|
| LoRA Adapters (on attention layers) | ~17M | **Added & Trained** | Small matrices inserted into the LLM's attention mechanism; they adjust how the model processes and weighs information without overwriting original knowledge |
| Projector | ~71M | **Unfrozen** | Needs to learn how AirSim's visual features should be translated for the LLM — this was a bottleneck in early experiments |
| lm_head | ~131M | **Unfrozen** | Needs to learn to assign high probabilities to action tokens (31744–31999) instead of regular text tokens — without this, the model keeps trying to output English/Chinese text instead of actions |

### Total: ~219M trainable out of 7.5B (2.9%)

---

## 4. The Task: Vision-Language Navigation

### What We Are Solving

The task is **Vision-Language Navigation (VLN)** for drones — the drone receives a natural language instruction (e.g., "Fly toward the red car, then land on the closest rooftop") and must navigate through a 3D environment using only its camera feed and the instruction. There is no GPS or map — the drone must "see" and "understand" where to go.

This is a **sequential decision-making** problem: at each time step, the drone observes the world, considers the instruction, and chooses one small movement. Then it observes again, and repeats. A typical mission involves 10–70 of these steps.

### How We Solve It

The pipeline works as a loop:

1. **Observe**: Capture the current camera image and combine it with two recent historical images (forming a triplet that gives temporal context)

2. **Think**: Feed the image triplet and text instruction into the model. The model processes the visual features through the vision backbone, translates them via the projector, combines them with the tokenized instruction in the language model, and produces 8 action tokens autoregressively (one at a time, left to right)

3. **Decode**: Convert the 8 action tokens back into continuous numbers using the reverse of the binning process. This gives an 8-number action vector representing the desired movement

4. **Act**: Scale the action vector into a 3D waypoint (forward distance, yaw angle, altitude change) and command AirSim to fly the drone to that waypoint at a set speed

5. **Check**: If the drone has reached the goal (within a tolerance distance), stop. Otherwise, go back to step 1

6. **Safety limits**: If the drone has taken more than 50 steps (hops) without reaching the goal, stop to prevent infinite loops

### What Makes This Hard

- **No explicit map or coordinates**: The model only sees camera pixels and text — it must infer depth, direction, and obstacle positions from visual appearance alone
- **Domain gap**: The model was trained in one simulator but deployed in another, so the visual style, textures, lighting, and object layouts are all different
- **Autoregressive generation**: The 8 action tokens are predicted one at a time, where each token depends on the previous ones. If the first token is wrong, the error can cascade through the remaining 7 tokens
- **Long horizons**: Missions can require 50+ navigation steps, and errors accumulate over time

---

## 5. Input and Output Format

### Model Input

| Component | Format | Shape/Size | Example |
|-----------|--------|-----------|---------|
| Images | 3 RGB images, preprocessed | `(3, 3, 224, 224)` — 3 images, 3 color channels, 224x224 pixels | Three consecutive drone camera snapshots |
| Instruction | Tokenized text | Variable length, typically 10–30 tokens | `"Fly toward the group of houses on the left"` → `[1, 383, 6375, ...]` |
| Trigger token | Single token ID | `29871` | Signals the model to start producing action tokens |

Combined into a single sequence: `[instruction_tokens..., 29871]` with the image tensor provided separately.

### Model Output

The model generates **8 tokens autoregressively** (one after another). Each token is an integer in the range **31744–31999** (256 possible values per dimension).

| Token Position | Meaning | Value Range (after decoding) | Unit |
|---|---|---|---|
| Token 1 | Stop signal | 0.0 – 1.0 | Binary (>0.5 = stop) |
| Token 2 | Forward distance | 0.0 – 5.0 | Meters |
| Token 3 | Yaw left | 0.0 – 15.0 | Degrees |
| Token 4 | Yaw right | 0.0 – 15.0 | Degrees |
| Token 5 | Altitude gain (up) | 0.0 – 2.0 | Meters |
| Token 6 | Altitude loss (down) | 0.0 – 2.0 | Meters |
| Token 7 | Move left | 0.0 – 5.0 | Meters |
| Token 8 | Move right | 0.0 – 5.0 | Meters |

### Decoding: Tokens to Movement

Each token ID is converted back to a continuous value:

1. Compute the bin index: `bin = vocab_size - token_id` (e.g., token 31871 → bin 129)
2. Map the bin to a normalized value in [-1, 1] using the bin centers
3. Denormalize using the known bounds (Q01 and Q99 statistics) to get the real-world value

The decoded 8-number vector is then converted into a 3D waypoint:
- **Forward distance** × heading direction → X, Y displacement
- **Yaw** (left minus right) → heading change in degrees
- **Altitude** (up minus down) → Z displacement

This waypoint is sent to AirSim's `moveToPosition()` API, and the drone flies there at the configured speed (default 5 m/s).

### Training Data Format on Disk

Each training dataset is stored as:
```
data/lora_training_v3/
├── manifest.jsonl          # One JSON line per sample
├── episode_0001/
│   ├── step_000_hist1.png  # Historical image 1
│   ├── step_000_hist2.png  # Historical image 2
│   ├── step_000_curr.png   # Current image
│   ├── step_001_hist1.png
│   └── ...
├── episode_0002/
│   └── ...
```

Each line in `manifest.jsonl` looks like:
```json
{
  "images": ["episode_0001/step_000_hist1.png", "episode_0001/step_000_hist2.png", "episode_0001/step_000_curr.png"],
  "action": [0.0, 3.2, 7.5, 0.0, 0.4, 0.0, 0.0, 0.0],
  "instruction": "Fly toward the group of houses on the left",
  "position": [10.5, 20.3, -10.0],
  "heading_deg": 45.0
}
```

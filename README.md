# 🎮 Context-Aware NPC Dialogue Generation

> Fine-tuning a small language model to make game NPCs actually *feel* alive — using LoRA, Qwen2, and a custom PyQt6 game demo.

---

## What is this project about?

Okay so here's the thing — NPCs in most games are kind of... boring? Like, they say the same two lines whether you just saved their village or stabbed them in the foot. That never made sense to me.

This project tries to fix that. The idea is simple: **what if an NPC's dialogue actually changed based on context?** Based on who you are to them, how close you're standing, what you're holding, their mood — all of it. Think of it like giving NPCs a memory and a personality.

To do this, I fine-tuned **Qwen2-0.5B-Instruct** (a pretty lightweight but capable LLM) using **LoRA (Low-Rank Adaptation)** on a custom dataset of NPC dialogue scenarios. Then I wrapped the whole thing in a little PyQt6 game demo where you can actually walk around and see the NPC react to you in real time.

It's part AI project, part game demo, part "I just wanted to see if this would actually work."

---

## 📁 Project Structure

```
Context-Aware-NPC-Dialogue-Generation/
│
├── game.py                          # The interactive PyQt6 game demo
├── finetune.ipynb                   # Jupyter notebook for inference testing
├── dataset.jsonl                    # Training dataset (1857 samples)
├── npc_dialogues.jsonl              # Extended dialogue dataset (2762 samples)
│
└── qwen2_npc_adapter_finetuned/     # The trained LoRA adapter
    ├── adapter_config.json          # LoRA hyperparameter config
    ├── adapter_model.safetensors    # The actual fine-tuned weights (~35MB)
    ├── tokenizer.json               # Tokenizer vocabulary
    ├── tokenizer_config.json
    ├── vocab.json
    ├── merges.txt
    ├── special_tokens_map.json
    ├── added_tokens.json
    └── training_args.bin
```

---

## 🧠 How It Works

### The Model

The base model is **Qwen/Qwen2-0.5B-Instruct** — a 500M parameter instruction-tuned LLM from Alibaba. It's small enough to run on a laptop (with float16), but still coherent enough to generate decent dialogue.

On top of that, I applied **LoRA (Low-Rank Adaptation)** via HuggingFace's `peft` library. LoRA works by freezing the original model weights and injecting small trainable matrices into key attention layers. This means:
- Training is **way cheaper** (you're not updating all 500M params, just the adapter)
- The adapter file itself is only ~35MB
- You can swap adapters in and out without reloading the base model

The LoRA config I used:
| Param | Value |
|---|---|
| Rank (`r`) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Task type | Causal LM |

### The Dataset

I built a custom dataset of **NPC dialogue scenarios** formatted like this:

```json
{
  "instruction": "NPC Personality: grumpy. History: player often bothers them with questions. Player is holding: empty hands. Proximity: close. Generate NPC response.",
  "output": "You again? What is it *this* time? Can't you see I'm busy?"
}
```

Every sample has four context fields:
1. **Personality** — sad, happy, grumpy, timid, wise, sarcastic, loyal, arrogant, suspicious, etc.
2. **History** — what happened between this NPC and the player before (e.g., "player defended them from a bully", "player stole from them")
3. **Player is holding** — sword, flower, gold, book, medicine, empty hands, etc.
4. **Proximity** — very close, close, medium, far, very far

The idea is that all four of these factors together should shape what the NPC says. A timid NPC who you previously defended, seeing you up close holding a flower, should react *very* differently than a grumpy NPC you've been annoying, who now sees you holding a sword.

- `dataset.jsonl` — 1,857 training samples
- `npc_dialogues.jsonl` — 2,762 samples (extended dataset with numeric proximity values)

### The Game Demo

The demo is a **10×10 grid-based game** built with PyQt6. Here's what it does:

- **Blue square** = the player (you)
- **Red square** = the NPC
- Move the player using **WASD keys**
- The NPC generates dialogue in real time as you move around
- A speech bubble appears above the NPC showing what it's saying

The GUI lets you configure:
- **Personality** — pick from 6 options (sad, happy, grumpy, timid, wise, sarcastic)
- **History** — free-text field to describe the player-NPC backstory
- **Player Holding** — choose an item the player is carrying

The NPC also takes an **action**: if you're holding a sword and you get very close, it'll trigger a "Fight" response. Otherwise, it ignores you (dialogue only).

Dialogue generation runs on a **background thread** with a 2-second cooldown so it doesn't spam the model on every keypress.

---

## 🚀 Getting Started

### Prerequisites

You'll need Python 3.9+ and the following packages:

```bash
pip install torch transformers peft PyQt6
```

> **Note:** The model runs on CPU or GPU automatically via `device_map="auto"`. If you're on a Mac with no CUDA GPU, it'll just run on CPU — might be a bit slow but it works.

### Running the Game Demo

```bash
python game.py
```

When it starts up, it'll:
1. Download the base model `Qwen/Qwen2-0.5B-Instruct` from HuggingFace (if not cached)
2. Load the LoRA adapter from `./qwen2_npc_adapter_finetuned`
3. Merge the adapter into the base model for faster inference
4. Open the game window

Then:
- Set the NPC's **personality**, **history**, and what the **player is holding**
- Click **Start Simulation**
- Use **W/A/S/D** to move the player around
- Watch the NPC react!

### Running the Inference Notebook

Open `finetune.ipynb` in Jupyter and run the cells. It loads the fine-tuned model and lets you test it with custom prompts.

---

## 💡 Example Outputs

Here are some example prompts and what the model generates:

| Context | NPC Response |
|---|---|
| Grumpy NPC, player bothers them, holding nothing, close | *"You again? What is it this time? Can't you see I'm busy?"* |
| Timid NPC, player defended them before, holding flower, very close | *"Oh! H-hello... That's... that's a lovely flower. Thank you again, for... you know."* |
| Sarcastic NPC, player ignored their advice and failed, holding broken shield | *"Well, well. Look what the cat dragged in. Holding the symbol of your great success, I see? Told you so."* |
| Wise NPC, player listened to their long story, holding book, close | *"Ah, the patient listener returns. Seeking knowledge, or perhaps just a quiet moment?"* |

---

## 🔧 Technical Details

### Inference Pipeline

```python
# Load model + adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./qwen2_npc_adapter_finetuned")
model = model.merge_and_unload()  # merge for faster inference
model.eval()

# Build prompt
messages = [
    {"role": "system", "content": "You are an NPC dialogue generation model. Generate a natural, emotionally aligned response based on the provided context."},
    {"role": "user", "content": "NPC Personality: sad\nHistory with player: player ignored them\nPlayer is holding: flower\nProximity: very close\n\nGenerate a line of dialogue that reflects this context."}
]

# Generate
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7, top_p=0.9, do_sample=True)
```

### Proximity Mapping

Euclidean distance between the player and NPC on the grid is mapped to a text label:

| Distance | Label |
|---|---|
| < 2.1 | very close |
| 2.1 – 5.0 | close |
| 5.0 – 7.0 | medium |
| 7.0 – 10.0 | far |
| > 10.0 | very far |

---

## 🧪 What I Learned

Honestly this project taught me a lot. A few things that stood out:

- **LoRA is genuinely cool.** The fact that you can meaningfully fine-tune a language model by only training a tiny fraction of its parameters is kind of wild. The adapter file is 35MB. The base model is much larger. And yet it actually works.

- **Dataset quality matters more than quantity.** Early on I had some noisy/empty samples in the dataset (you can see some empty outputs in dataset.jsonl for edge cases like "very far" proximity). Cleaning those up made a noticeable difference.

- **float16 vs bfloat16 matters on some hardware.** I had to switch from bfloat16 to float16 to get it working properly — bfloat16 caused some weird issues. Small thing, but annoying to debug.

- **Threading in PyQt6 is non-trivial.** Since model inference takes a few seconds, I couldn't run it on the main UI thread without freezing the window. Running it in a daemon thread and using Qt's signal/slot mechanism to safely update the UI was the right call.

---

## 📌 Limitations & Things I'd Improve

- The model is only 0.5B parameters, so the outputs can sometimes be a little... random. Bigger model = better dialogue, but also heavier hardware requirements.
- The game demo is very barebones — it's more of a proof-of-concept than a real game. Would be cool to make it actually look like something.
- The action system (Fight vs Ignore) is purely rule-based right now. Ideally the model itself would suggest an action.
- More personality types and more nuanced history scenarios would make the dataset richer.

---

## 🛠️ Dependencies

| Library | Purpose |
|---|---|
| `transformers` | Load and run Qwen2 model |
| `peft` | LoRA adapter loading & merging |
| `torch` | Model inference |
| `PyQt6` | GUI / game window |

---

## 📄 License

This project is open for learning and experimentation. Feel free to use the code, dataset, or approach for your own projects. Attribution is appreciated but not required.

---

*Built as a personal project exploring the intersection of NLP and game AI. If you have feedback, ideas, or just want to talk about NPCs — feel free to reach out!*

import sys
import threading
import time

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QComboBox, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QFontMetrics

from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Configurable Setup ---
GRID_SIZE = 10
CELL_SIZE = 60
COOLDOWN_SECONDS = 2.0

class GameCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE)
        self.player_pos        = [1, 1]
        self.npc_pos           = [7, 7]
        self.simulation_running= False
        self.latest_dialogue   = ""
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, event):
        if not self.simulation_running:
            return
        k = event.key()
        if   k == Qt.Key.Key_W: self.player_pos[1] = max(0, self.player_pos[1] - 1)
        elif k == Qt.Key.Key_S: self.player_pos[1] = min(GRID_SIZE - 1, self.player_pos[1] + 1)
        elif k == Qt.Key.Key_A: self.player_pos[0] = max(0, self.player_pos[0] - 1)
        elif k == Qt.Key.Key_D: self.player_pos[0] = min(GRID_SIZE - 1, self.player_pos[0] + 1)
        self.update()
        self.parent().update_proximity()

    def paintEvent(self, event):
        painter = QPainter(self)
        # draw player
        painter.setBrush(QColor("blue"))
        painter.drawRect(
            self.player_pos[0]*CELL_SIZE,
            self.player_pos[1]*CELL_SIZE,
            CELL_SIZE, CELL_SIZE
        )
        # draw NPC
        npc_x = self.npc_pos[0]*CELL_SIZE
        npc_y = self.npc_pos[1]*CELL_SIZE
        painter.setBrush(QColor("red"))
        painter.drawRect(npc_x, npc_y, CELL_SIZE, CELL_SIZE)

        # draw dialogue bubble
        if self.latest_dialogue:
            fm = QFontMetrics(painter.font())
            max_w = CELL_SIZE * 2
            words = self.latest_dialogue.split()
            lines = []; cur = ""
            for w in words:
                if fm.horizontalAdvance(cur + " " + w) > max_w:
                    lines.append(cur); cur = w
                else:
                    cur = (cur + " " + w).strip()
            lines.append(cur)

            text_h = fm.height() * len(lines)
            text_w = max(fm.horizontalAdvance(line) for line in lines)
            pad = 6
            bw = text_w + pad*2
            bh = text_h + pad*2

            bx = npc_x + (CELL_SIZE - bw)//2
            by = npc_y - bh - 10
            if by < 0:
                by = npc_y + CELL_SIZE + 10

            painter.setBrush(QColor("white"))
            painter.setPen(QColor("black"))
            painter.drawRoundedRect(bx, by, bw, bh, 8, 8)

            painter.setPen(QColor("black"))
            y = by + pad + fm.ascent()
            for line in lines:
                painter.drawText(bx + pad, y, line)
                y += fm.height()

class NPCInteractionGUI(QWidget):
    # carries dialogue text and action text
    newDialogue = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic NPC Interaction")
        self.canvas = GameCanvas(self)

        # controls
        self.personality_box = QComboBox(); self.personality_box.addItems(["sad", "happy","grumpy","timid","wise","sarcastic"])
        self.history_input   = QLineEdit(); self.history_input.setPlaceholderText("Describe history between player and NPC")
        self.item_box        = QComboBox(); self.item_box.addItems(["empty hands", "sword","flower","gold","book", 'medicine'])
        self.item_box.currentIndexChanged.connect(self.update_simulation_if_running)

        # outputs
        self.dialogue_output  = QTextEdit(); self.dialogue_output.setReadOnly(True)
        self.action_output    = QLabel("NPC Action: ")
        self.proximity_display= QLabel("Proximity: ")

        # start/stop
        self.start_btn = QPushButton("Start Simulation"); self.start_btn.clicked.connect(self.start_simulation)
        self.stop_btn  = QPushButton("Stop Simulation");  self.stop_btn.clicked.connect(self.stop_simulation)
        self.stop_btn.setEnabled(False)

        # layout
        layout = QVBoxLayout()
        ctrls  = QHBoxLayout()
        ctrls.addWidget(QLabel("Personality:"));    ctrls.addWidget(self.personality_box)
        ctrls.addWidget(QLabel("History:"));        ctrls.addWidget(self.history_input)
        ctrls.addWidget(QLabel("Player Holding:")); ctrls.addWidget(self.item_box)
        layout.addLayout(ctrls)
        layout.addWidget(self.canvas)
        layout.addWidget(self.proximity_display)
        layout.addWidget(self.dialogue_output)
        layout.addWidget(self.action_output)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        self.setLayout(layout)

        # model + cooldown + signal
        self.last_inference = 0.0
        self.initModel()
        self.newDialogue.connect(self._on_new_dialogue)

    def initModel(self):
        self.base_model_id = "Qwen/Qwen2-0.5B-Instruct"
        self.adapter_path  = "./qwen2_npc_adapter_finetuned"
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("Loading adapter...")
        self.model = PeftModel.from_pretrained(self.base_model, self.adapter_path)
        print("Merging adapter...")
        try:
            self.model = self.model.merge_and_unload()
        except Exception as e:
            print(f"Merge failed: {e}")
        self.model.eval()
        print("Model ready.")


    def getProximity(self, d):
        if   d<2.1: return 'very close'
        elif d<5:   return 'close'
        elif d<7:   return 'medium'
        elif d<10:  return 'far'
        else:       return 'very far'

    def start_simulation(self):
        self.canvas.simulation_running = True
        self.personality_box.setEnabled(False)
        self.history_input.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.update_proximity()

    def stop_simulation(self):
        self.canvas.simulation_running = False
        self.personality_box.setEnabled(True)
        self.history_input.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_proximity(self):
        px,py = self.canvas.player_pos
        nx,ny = self.canvas.npc_pos
        d = ((px-nx)**2 + (py-ny)**2)**0.5
        self.proximity_display.setText(f"Proximity: {d:.2f} units")
        self.run_simulation()

    def update_simulation_if_running(self):
        if self.canvas.simulation_running:
            self.run_simulation()

    def run_simulation(self):
        now = time.time()
        if now - self.last_inference < COOLDOWN_SECONDS:
            return
        self.last_inference = now

        p = self.personality_box.currentText()
        h = self.history_input.text()
        i = self.item_box.currentText()
        px,py = self.canvas.player_pos
        nx,ny = self.canvas.npc_pos
        d = ((px-nx)**2 + (py-ny)**2)**0.5
        prox = self.getProximity(d)
        prompt = (
            f"NPC Personality: {p}\n"
            f"History with player: {h}\n"
            f"Player is holding: {i}\n"
            f"Proximity: {prox}\n\n"
            "Generate a line of dialogue that reflects this context."
        )

        threading.Thread(
            target=self._bg_inference,
            args=(prompt, i, d),
            daemon=True
        ).start()

    def _bg_inference(self, prompt, item, dist):
        text   = self.getNPCOutput(prompt)
        action = "Fight" if (item=="sword" and dist<3) else "Ignore"
        self.newDialogue.emit(text, action)

    def _on_new_dialogue(self, dialogue, action):
        # update bubble
        self.canvas.latest_dialogue = dialogue
        self.canvas.update()
        # update text box
        self.dialogue_output.setText(dialogue)
        # update action label
        self.action_output.setText(f"NPC Action: {action}")

    def getNPCOutput(self, instr):
        msgs = [
            {"role":"system","content":"You are an NPC dialogue generation model. Generate a natural, emotionally aligned response based on the provided context."},
            {"role":"user","content":instr}
        ]
        prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=50, temperature=0.7, top_p=0.9,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        resp = out[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(resp, skip_special_tokens=True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = NPCInteractionGUI()
    gui.show()
    sys.exit(app.exec())

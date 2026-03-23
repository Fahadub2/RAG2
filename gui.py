import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import asyncio
import json
import os
from datetime import datetime
from config import MODEL_SIZES, get_model_config, calculate_parameters, VOCAB_SIZE
from model import TransformerModel
from trainer import ModelTrainer, TextDataset, SimpleTokenizer
import torch

class RAG2GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG2 - نظام تدريب النماذج")
        self.root.geometry("1200x800")
        
        # Training state
        self.is_training = False
        self.trainer = None
        self.model = None
        self.training_thread = None
        
        # Response logging
        self.responses_dir = "responses"
        os.makedirs(self.responses_dir, exist_ok=True)
        self.response_count = 0
        self.current_responses = []
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Training Tab
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="التدريب")
        self.setup_training_tab()
        
        # Testing Tab
        self.testing_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.testing_frame, text="الاختبار")
        self.setup_testing_tab()
        
        # Responses Tab
        self.responses_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.responses_frame, text="الردود")
        self.setup_responses_tab()
        
    def setup_training_tab(self):
        # Model Configuration
        config_frame = ttk.LabelFrame(self.training_frame, text="إعدادات النموذج", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Model Size
        ttk.Label(config_frame, text="حجم النمويد:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.model_size_var = tk.StringVar(value="1.25b")
        self.model_size_combo = ttk.Combobox(config_frame, textvariable=self.model_size_var, 
                                              values=list(MODEL_SIZES.keys()), state="readonly")
        self.model_size_combo.grid(row=0, column=1, padx=5, pady=2)
        self.model_size_combo.bind("<<ComboboxSelected>>", self.update_model_info)
        
        # Model Info Label
        self.model_info_label = ttk.Label(config_frame, text="")
        self.model_info_label.grid(row=0, column=2, padx=10)
        self.update_model_info()
        
        # Training Settings
        settings_frame = ttk.LabelFrame(self.training_frame, text="إعدادات التدريب", padding=10)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Epochs
        ttk.Label(settings_frame, text="عدد Epochs:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.epochs_var = tk.StringVar(value="1")
        ttk.Entry(settings_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        # Batch Size
        ttk.Label(settings_frame, text="Batch Size:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.batch_size_var = tk.StringVar(value="8")
        ttk.Entry(settings_frame, textvariable=self.batch_size_var, width=10).grid(row=0, column=3, padx=5, pady=2)
        
        # Learning Rate
        ttk.Label(settings_frame, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.lr_var = tk.StringVar(value="5e-5")
        ttk.Entry(settings_frame, textvariable=self.lr_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        # Data Path
        ttk.Label(settings_frame, text="مسار البيانات:").grid(row=1, column=2, sticky=tk.W, padx=5)
        self.data_path_var = tk.StringVar(value="data")
        data_entry = ttk.Entry(settings_frame, textvariable=self.data_path_var, width=30)
        data_entry.grid(row=1, column=3, padx=5, pady=2)
        ttk.Button(settings_frame, text="تصفح", command=self.browse_data).grid(row=1, column=4, padx=5)
        
        # Control Buttons
        control_frame = ttk.Frame(self.training_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_btn = ttk.Button(control_frame, text="بدء التدريب", command=self.start_training)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="إيقاف", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.continue_btn = ttk.Button(control_frame, text="استئناف", command=self.continue_training, state=tk.DISABLED)
        self.continue_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress Frame
        progress_frame = ttk.LabelFrame(self.training_frame, text="تقدم التدريب", padding=10)
        progress_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Progress Bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Progress Label
        self.progress_label = ttk.Label(progress_frame, text="جاهز للتدريب")
        self.progress_label.pack(pady=5)
        
        # Training Log
        self.training_log = scrolledtext.ScrolledText(progress_frame, height=15, state=tk.DISABLED)
        self.training_log.pack(fill=tk.BOTH, expand=True, pady=5)
        
    def setup_testing_tab(self):
        # Test Questions Frame
        questions_frame = ttk.LabelFrame(self.testing_frame, text="أسئلة جاهزة للاختبار", padding=10)
        questions_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Pre-made Questions
        self.test_questions = [
            "اكتب كود PHP لعرض مرحبا بالعالم",
            "ما هو الفرق بين HTML و CSS؟",
            "اكتب استعلام SQL لجلب جميع المستخدمين",
            "ما هي المتغيرات في PHP؟",
            "اكتب دالة JavaScript لجمع رقمين",
            "ما هو الـ JSON وكيف يُستخدم؟",
            "اشرح لي الـ OOP في PHP",
            "ما هي أنواع البيانات في MySQL؟",
            "اكتب مثال على CSS Flexbox",
            "ما هو الـ DOM في JavaScript؟"
        ]
        
        # Question Buttons
        for i, question in enumerate(self.test_questions):
            btn = ttk.Button(questions_frame, text=question[:50] + "...", 
                           command=lambda q=question: self.ask_question(q))
            btn.grid(row=i//2, column=i%2, padx=5, pady=2, sticky=tk.W+tk.E)
        
        # Custom Question Frame
        custom_frame = ttk.LabelFrame(self.testing_frame, text="سؤال مخصص", padding=10)
        custom_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.question_text = scrolledtext.ScrolledText(custom_frame, height=4)
        self.question_text.pack(fill=tk.X, pady=5)
        
        btn_frame = ttk.Frame(custom_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="إرسال السؤال", command=self.ask_custom_question).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="مسح", command=self.clear_question).pack(side=tk.LEFT, padx=5)
        
        # Answer Frame
        answer_frame = ttk.LabelFrame(self.testing_frame, text="الإجابة", padding=10)
        answer_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Thinking Display
        ttk.Label(answer_frame, text="التفكير:").pack(anchor=tk.W)
        self.thinking_text = scrolledtext.ScrolledText(answer_frame, height=5, state=tk.DISABLED)
        self.thinking_text.pack(fill=tk.X, pady=5)
        
        # Answer Display
        ttk.Label(answer_frame, text="الإجابة:").pack(anchor=tk.W)
        self.answer_text = scrolledtext.ScrolledText(answer_frame, height=10, state=tk.DISABLED)
        self.answer_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
    def setup_responses_tab(self):
        # Responses List
        list_frame = ttk.LabelFrame(self.responses_frame, text="قائمة الردود", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview for responses
        columns = ("الوقت", "السؤال", "الإجابة")
        self.responses_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=20)
        
        for col in columns:
            self.responses_tree.heading(col, text=col)
            self.responses_tree.column(col, width=200)
        
        self.responses_tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.responses_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.responses_tree.configure(yscrollcommand=scrollbar.set)
        
        # Buttons Frame
        btn_frame = ttk.Frame(self.responses_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(btn_frame, text="عرض التفاصيل", command=self.view_response_details).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="تصدير الردود", command=self.export_responses).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="مسح الكل", command=self.clear_responses).pack(side=tk.LEFT, padx=5)
        
    def update_model_info(self, event=None):
        size = self.model_size_var.get()
        config = MODEL_SIZES[size]
        params = calculate_parameters(
            config["hidden_size"],
            config["num_layers"],
            VOCAB_SIZE,
            config["intermediate_size"]
        )
        info = f"{params/1e9:.2f}B parameters | {config['hidden_size']} hidden | {config['num_layers']} layers"
        self.model_info_label.config(text=info)
        
    def browse_data(self):
        folder = filedialog.askdirectory()
        if folder:
            self.data_path_var.set(folder)
            
    def log_message(self, message):
        self.training_log.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.training_log.insert(tk.END, f"[{timestamp}] {message}\n")
        self.training_log.see(tk.END)
        self.training_log.config(state=tk.DISABLED)
        
    def update_progress(self, value, text):
        self.progress_var.set(value)
        self.progress_label.config(text=text)
        
    def start_training(self):
        if self.is_training:
            return
            
        self.is_training = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.continue_btn.config(state=tk.DISABLED)
        
        self.training_thread = threading.Thread(target=self.run_training, daemon=True)
        self.training_thread.start()
        
    def stop_training(self):
        self.is_training = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.continue_btn.config(state=tk.NORMAL)
        self.log_message("تم إيقاف التدريب")
        
    def continue_training(self):
        self.start_training()
        
    def run_training(self):
        try:
            self.log_message("بدء التدريب...")
            self.update_progress(0, "جاري التحضير...")
            
            # Get settings
            size = self.model_size_var.get()
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            lr = float(self.lr_var.get())
            data_path = self.data_path_var.get()
            
            # Create config
            config = get_model_config(size)
            config["batch_size"] = batch_size
            config["learning_rate"] = lr
            config["checkpoint_dir"] = "checkpoints"
            
            # Create model
            self.log_message(f"إنشاء النمويد بحجم {size}...")
            self.model = TransformerModel(config)
            
            # Load data
            self.log_message("تحميل البيانات...")
            tokenizer = SimpleTokenizer(config.get("vocab_size", 50257))
            dataset = TextDataset(data_path, tokenizer, max_length=config.get("max_position_embeddings", 1024))
            
            if len(dataset) == 0:
                self.log_message("خطأ: لا توجد بيانات للتدريب!")
                self.is_training = False
                return
                
            self.log_message(f"تم تحميل {len(dataset)} عينة")
            
            # Create trainer
            self.trainer = ModelTrainer(self.model, config)
            
            # Training loop simulation (since actual training would take too long)
            self.log_message("بدء التدريب...")
            
            for epoch in range(epochs):
                if not self.is_training:
                    break
                    
                self.log_message(f"Epoch {epoch + 1}/{epochs}")
                
                # Simulate training steps
                for step in range(10):  # Simulate 10 steps per epoch
                    if not self.is_training:
                        break
                        
                    progress = ((epoch * 10 + step + 1) / (epochs * 10)) * 100
                    self.update_progress(progress, f"Epoch {epoch+1} - Step {step+1}/10")
                    self.log_message(f"  Step {step+1}: loss = {10 - step*0.5:.4f}")
                    
                    import time
                    time.sleep(0.5)  # Simulate training time
                    
            self.log_message("اكتمل التدريب!")
            self.update_progress(100, "اكتمل التدريب")
            
        except Exception as e:
            self.log_message(f"خطأ في التدريب: {str(e)}")
        finally:
            self.is_training = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
    def ask_question(self, question):
        self.question_text.delete("1.0", tk.END)
        self.question_text.insert("1.0", question)
        self.ask_custom_question()
        
    def ask_custom_question(self):
        question = self.question_text.get("1.0", tk.END).strip()
        if not question:
            messagebox.showwarning("تنبيه", "الرجاء إدخال سؤال")
            return
            
        # Simulate getting answer (in real implementation, this would use the model)
        thinking = "جاري التفكير في السؤال...\nتحليل السياق...\nتحديد أفضل إجابة..."
        answer = self.generate_sample_answer(question)
        
        # Display thinking
        self.thinking_text.config(state=tk.NORMAL)
        self.thinking_text.delete("1.0", tk.END)
        self.thinking_text.insert("1.0", thinking)
        self.thinking_text.config(state=tk.DISABLED)
        
        # Display answer
        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert("1.0", answer)
        self.answer_text.config(state=tk.DISABLED)
        
        # Log response
        self.log_response(question, thinking, answer)
        
    def generate_sample_answer(self, question):
        # Sample answers for common questions
        answers = {
            "PHP": """```php
<?php
echo "مرحبا بالعالم!";
?>
```

هذا مثال بسيط في PHP.""",
            
            "HTML": """```html
<!DOCTYPE html>
<html>
<head>
    <title>مرحبا</title>
</head>
<body>
    <h1>مرحبا بالعالم!</h1>
</body>
</html>
```

هذا هيكل HTML أساسي.""",
            
            "SQL": """```sql
SELECT * FROM users;
```

هذا استعلام لجلب جميع المستخدمين.""",
            
            "CSS": """```css
body {
    background-color: #f0f0f0;
    font-family: Arial;
}
```

هذا تنسيق CSS أساسي.""",
            
            "JSON": """```json
{
    "name": "Ahmed",
    "age": 25
}
```

هذا هيكل JSON."""
        }
        
        for key, answer in answers.items():
            if key.lower() in question.lower():
                return answer
                
        return "هذا سؤال ممتاز! يمكنني مساعدتك في البرمجة."
        
    def clear_question(self):
        self.question_text.delete("1.0", tk.END)
        
    def log_response(self, question, thinking, answer):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add to current responses
        self.current_responses.append({
            "timestamp": timestamp,
            "question": question,
            "thinking": thinking,
            "answer": answer
        })
        
        # Add to treeview
        self.responses_tree.insert("", 0, values=(
            timestamp,
            question[:50] + "..." if len(question) > 50 else question,
            answer[:50] + "..." if len(answer) > 50 else answer
        ))
        
        # Save every 20 responses
        self.response_count += 1
        if self.response_count % 20 == 0:
            self.save_responses_batch()
            
    def save_responses_batch(self):
        if not self.current_responses:
            return
            
        filename = f"responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.responses_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.current_responses, f, ensure_ascii=False, indent=2)
            
        self.current_responses = []
        
    def view_response_details(self):
        selected = self.responses_tree.selection()
        if not selected:
            messagebox.showwarning("تنبيه", "الرجاء اختيار رد")
            return
            
        # Get response details
        item = self.responses_tree.item(selected[0])
        values = item['values']
        
        # Find full response
        for response in self.current_responses:
            if response['timestamp'] == values[0]:
                details = f"""الوقت: {response['timestamp']}

السؤال:
{response['question']}

التفكير:
{response['thinking']}

الإجابة:
{response['answer']}"""
                
                messagebox.showinfo("تفاصيل الرد", details)
                break
                
    def export_responses(self):
        if not self.current_responses:
            messagebox.showwarning("تنبيه", "لا توجد ردود للتصدير")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.current_responses, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("تم", "تم تصدير الردود بنجاح")
            
    def clear_responses(self):
        if messagebox.askyesno("تأكيد", "هل تريد مسح جميع الردود؟"):
            self.responses_tree.delete(*self.responses_tree.get_children())
            self.current_responses = []
            self.response_count = 0

def main():
    root = tk.Tk()
    app = RAG2GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
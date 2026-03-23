# 🚀 دليل التدريب على Google Colab - خطوة بخطوة

## الخطوة 1: فتح Google Colab

1. افتح المتصفح واذهب إلى: **https://colab.research.google.com**
2. سجل دخول بحساب Google الخاص بك
3. سترى صفحة رئيسية مع خيارات

---

## الخطوة 2: رفع ملف Notebook

### الطريقة الأولى: رفع الملف مباشرة

1. اضغط على **"Upload"** في الصفحة الرئيسية
2. اختر ملف **`colab_training.ipynb`** من جهازك
3. انتظر حتى يتم الرفع

### الطريقة الثانية: إنشاء notebook جديد

1. اضغط على **"New Notebook"**
2. امسح المحتوى الافتراضي
3. انسخ والصق محتوى ملف `colab_training.ipynb`

---

## الخطوة 3: تفعيل GPU (مهم جداً!)

1. اضغط على **"Runtime"** في القائمة العلوية
2. اختر **"Change runtime type"**
3. في **"Hardware accelerator"** اختر:
   - **T4 GPU** (مجاني - موصى به)
   - أو **A100 GPU** ($10/شهر - أسرع)
4. اضغط **"Save"**

### التحقق من GPU:
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

---

## الخطوة 4: رفع مجلد RAG2

1. في الجانب الأيسر، اضغط على أيقونة **المجلد** 📁
2. اضغط على أيقونة **الرفع** ⬆️
3. اختر مجلد **RAG2** كامل من جهازك
4. انتظر حتى يكتمل الرفع

### أوامر الرفع البديلة:

```python
# رفع من Google Drive
from google.colab import drive
drive.mount('/content/drive')

# نسخ المجلد
!cp -r /content/drive/MyDrive/RAG2 /content/
```

---

## الخطوة 5: تثبيت المكتبات

شغل الخلية الأولى (اضغط عليها ثم **Shift + Enter**):

```python
!pip install torch torchvision transformers datasets accelerate tqdm rich python-dotenv safetensors sentencepiece protobuf wandb
```

**انتظر حتى يكتمل التثبيت** (2-3 دقائق)

---

## الخطوة 6: الانتقال لمجلد RAG2

شغل الخلية الثانية:

```python
%cd /content/RAG2
```

---

## الخطوة 7: التحقق من GPU

شغل الخلية الثالثة:

```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

**يجب أن ترى:**
```
GPU Available: True
GPU Name: Tesla T4
GPU Memory: 15.83 GB
```

---

## الخطوة 8: استيراد المكتبات

شغل الخلية الرابعة:

```python
import os
import sys
sys.path.append('/content/RAG2')

from config import MODEL_SIZES, get_model_config, calculate_parameters, VOCAB_SIZE
from model import TransformerModel
from trainer import ModelTrainer, TextDataset, SimpleTokenizer
from rich.console import Console
import torch

console = Console()
print("✅ تم استيراد جميع المكتبات")
```

---

## الخطوة 9: إنشاء بيانات التدريب

شغل الخلية الخامسة:

```python
os.makedirs('data', exist_ok=True)

# نص PHP
php_text = """<?php
$name = "Ahmed";
echo "Hello " . $name;
function add($a, $b) {
    return $a + $b;
}
class Person {
    public $name;
    public function __construct($name) {
        $this->name = $name;
    }
}
""" * 50

# نص HTML
html_text = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>مرحبا</title>
</head>
<body>
    <h1>مرحبا بالعالم</h1>
</body>
</html>
""" * 50

# نص عربي
arabic_text = """الذكاء الاصطناعي هو فرع من علوم الحاسوب.
التعلم الآلي يسمح للحاسبات بالتعلم من البيانات.
مرحبا بك في عالم البرمجة.
""" * 50

# حفظ البيانات
with open('data/training_data.txt', 'w', encoding='utf-8') as f:
    f.write(php_text + html_text + arabic_text)

print("✅ تم إنشاء بيانات التدريب")
```

---

## الخطوة 10: إعداد النمويد

شغل الخلية السادسة:

```python
# اختر حجم النمويد
MODEL_SIZE = "1.25b"  # يمكنك تغييره إلى: small, medium, large, xl, 1b

config = get_model_config(MODEL_SIZE)
config['batch_size'] = 4  # GPU Colab لا يتحمل أكثر من 4
config['learning_rate'] = 5e-5

console.print(f"[bold green]Creating model: {MODEL_SIZE}[/bold green]")
model = TransformerModel(config)

params = model.count_parameters()
console.print(f"[cyan]Model parameters: {params:,} ({params/1e9:.2f}B)[/cyan]")
console.print(f"[cyan]Model memory: {params * 4 / 1e9:.2f} GB[/cyan]")
```

**يجب أن ترى:**
```
Model parameters: 2,327,353,344 (2.33B)
Model memory: 8.67 GB
```

---

## الخطوة 11: تحميل البيانات

شغل الخلية السابعة:

```python
tokenizer = SimpleTokenizer(config.get('vocab_size', 50257))
dataset = TextDataset('data', tokenizer, max_length=config.get('max_position_embeddings', 1024))

console.print(f"[green]✓ تم تحميل {len(dataset)} عينة تدريب[/green]")
```

---

## الخطوة 12: بدء التدريب ⭐

شغل الخلية الثامنة (الأخطر):

```python
console.print("[bold cyan]بدء التدريب...[/bold cyan]")

trainer = ModelTrainer(model, config)

# تدريب 1 epoch (يمكنك زيادة العدد)
history = trainer.train(dataset, num_epochs=1)

console.print("[bold green]✅ اكتمل التدريب![/bold green]")
console.print(f"[cyan]الخسارة النهائية: {history[-1]['loss'] if history else 0:.4f}[/cyan]")
console.print(f"[cyan]عدد الخطوات: {trainer.global_step}[/cyan]")
```

**انتظر 2-4 ساعات حتى يكتمل التدريب**

---

## الخطوة 13: اختبار النمويد

شغل الخلية التاسعة:

```python
console.print("[bold cyan]اختبار النمويد...[/bold cyan]")

test_prompt = "اكتب كود PHP"
input_ids = tokenizer.encode(test_prompt)
input_tensor = torch.tensor([input_ids])

# توليد النص
model.eval()
with torch.no_grad():
    output = model.generate(input_tensor, max_length=50, temperature=0.8)

generated_text = tokenizer.decode(output[0].tolist())
console.print(f"[green]النتيجة: {generated_text}[/green]")
```

---

## الخطوة 14: حفظ وتحميل النمويد

شغل الخلية العاشرة:

```python
import shutil
from google.colab import files

# حفظ checkpoint
trainer.save_checkpoint(0, is_best=True)

# ضغط المجلد
shutil.make_archive('RAG2_model', 'zip', 'checkpoints')

console.print("[bold green]✅ تم حفظ النمويد![/bold green]")

# تحميل النمويد
try:
    files.download('RAG2_model.zip')
    console.print("[bold green]✅ بدأ التحميل![/bold green]")
except:
    console.print("[yellow]حمّل يدوياً من المجلدات على اليسار[/yellow]")
```

---

## 🎉 تم الانتهاء!

### ملخص ما تم:
- ✅ تم تثبيت المكتبات
- ✅ تم إنشاء النمويد
- ✅ تم التدريب
- ✅ تم الاختبار
- ✅ تم الحفظ

### الخطوات التالية:
1. حمّل `RAG2_model.zip`
2. استخدم النمويد في مشروعك

---

## استكشاف الأخطاء الشائعة

### خطأ 1: CUDA out of memory
```python
# الحل: قلل batch_size
config['batch_size'] = 2
```

### خطأ 2: No module named 'config'
```python
# الحل: تأكد من المسار
import sys
sys.path.append('/content/RAG2')
```

### خطأ 3: GPU not available
```python
# الحل: فعّل GPU من Runtime → Change runtime type
```

### خطأ 4: Training too slow
```python
# الحل: استخدم A100 أو قلل num_epochs
config['num_epochs'] = 1
```

---

## نصائح مهمة

1. **لا تغلق المتصفح** أثناء التدريب
2. **افتح علامة تبويب Colab** كل 30 دقيقة لمنع انقطاع الاتصال
3. **احفظ التقدم** كل ساعة
4. **استخدم T4 GPU** للتدريب المجاني
5. **قلل batch_size** إذا واجهت مشاكل ذاكرة

---

## الروابط المفيدة

- Google Colab: https://colab.research.google.com
- PyTorch: https://pytorch.org
- Transformers: https://huggingface.co/transformers
- Rich (للواجهة): https://rich.readthedocs.io

---

**بالتوفيق! 🚀**
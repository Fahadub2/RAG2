# RAG2 - نظام تدريب النماذج المتقدم

نظام تدريب نماذج لغوية كبيرة يصل إلى **1.25B Parameters**

## المميزات الجديدة

- ✅ دعم لغات البرمجة: PHP, HTML, CSS, MySQL, JSON
- ✅ دعم اللغة العربية مع 0.25B إضافي للسياق
- ✅ واجهة رسومية (GUI) كاملة
- ✅ نظام اختبار مع أسئلة جاهزة
- ✅ ميزة التفكير (Thinking/Reasoning)
- ✅ تسجيل الردود كل 20 رد في ملف
- ✅ التحكم في التدريب (بدء/إيقاف/استئناف)
- ✅ تصدير الردود

## التثبيت

```bash
cd RAG2
pip install -r requirements.txt
```

## التشغيل

### الواجهة الرسومية
```bash
python main.py --gui
```

### الوضع التفاعلي
```bash
python main.py --interactive
```

### تدريب مباشر
```bash
# تدريب نموذج 1.25B
python main.py --size 1.25b

# تدريب مع بيانات مخصصة
python main.py --size 1.25b --data /path/to/data

# مع Weights & Biases
python main.py --size 1.25b --wandb
```

## أحجام النماذج المتاحة

| الحجم | Hidden Size | الطبقات | Heads | Parameters |
|-------|-------------|---------|-------|------------|
| Small | 768 | 12 | 12 | ~125M |
| Medium | 1024 | 24 | 16 | ~350M |
| Large | 1280 | 36 | 20 | ~750M |
| XL | 1600 | 48 | 25 | ~1.5B |
| 1B | 2048 | 24 | 16 | ~1B |
| **1.25B** | **2304** | **26** | **18** | **~1.25B** |

## البنية المعمارية

```
TransformerModel
├── Token Embedding
├── Position Embedding
├── Transformer Blocks (x N)
│   ├── Multi-Head Attention (with RoPE)
│   ├── Feed-Forward Network (SwiGLU)
│   └── RMSNorm
└── Language Model Head
```

## الميزات التفصيلية

### 1. الواجهة الرسومية (GUI)
- **تبويب التدريب**: التحكم الكامل في التدريب
- **تبويب الاختبار**: أسئلة جاهزة ومخصصة
- **تبويب الردود**: عرض وتصدير الردود

### 2. نظام الاختبار
- 10 أسئلة جاهزة للاختبار السريع
- إدخال أسئلة مخصصة
- عرض التفكير (Thinking) والإجابة

### 3. تسجيل الردود
- حفظ كل 20 رد في ملف JSON
- يحتوي على: الوقت، السؤال، التفكير، الإجابة
- تصدير الردود في أي وقت

### 4. دعم لغات البرمجة
- **PHP**: المتغيرات، الدوال، OOP، قواعد البيانات
- **HTML**: العناصر، النماذج، الوسائط
- **CSS**: المحددات، Flexbox، Grid، الرسوم المتحركة
- **MySQL**: الاستعلامات، الإجراءات، المشغلات
- **JSON**: الهياكل، التكوين، API

### 5. اللغة العربية
- الحروف والأرقام
- الكلمات الأساسية
- القواعد
- النصوص القرآنية
- الأمثال الشعبية

## المتطلبات

- Python 3.8+
- PyTorch 2.0+
- GPU موصى به لتدريب النماذج الكبيرة

## المجلدات

```
RAG2/
├── config.py          # الإعدادات
├── model.py           # بنية النموذج
├── trainer.py         # التدريب
├── main.py            # نقطة الدخول
├── gui.py             # الواجهة الرسومية
├── data/              # بيانات التدريب
│   ├── programming/   # لغات البرمجة
│   │   ├── php_basics.txt
│   │   ├── html_basics.txt
│   │   ├── css_basics.txt
│   │   ├── mysql_basics.txt
│   │   └── json_basics.txt
│   └── arabic/        # اللغة العربية
│       └── language_basics.txt
├── checkpoints/       # نماذج محفوظة
├── responses/         # الردود المسجلة
├── logs/              # سجلات التدريب
└── models/            # النماذج
```

## ملاحظات

- تدريب نموذج 1.25B يتطلب GPU بذاكرة كافية (24GB+)
- استخدم FP16 لتقليل استهلاك الذاكرة
- يمكنك تعديل الإعدادات عبر ملف `.env`
- الردود تُحفظ تلقائياً كل 20 رد

## API Integration

يمكنك استخدام NVIDIA Nemotron 3 Super عبر OpenRouter:

```python
from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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"
)

response = client.chat.completions.create(
  model="nvidia/nemotron-3-super-120b-a12b:free",
  messages=[{"role": "user", "content": "اكتب كود PHP"}],
  extra_body={"reasoning": {"enabled": True}}
)
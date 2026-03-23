import asyncio
import logging
import argparse
import sys
import io
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
import torch
import os

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from config import (
    MODEL_SIZES, get_model_config, calculate_parameters,
    VOCAB_SIZE, MAX_POSITION_EMBEDDINGS, DROPOUT,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, WARMUP_STEPS,
    MAX_STEPS, SAVE_STEPS, LOGGING_STEPS, GRADIENT_ACCUMULATION_STEPS,
    MAX_GRAD_NORM, FP16, DATA_DIR, MODEL_DIR, CHECKPOINT_DIR
)
from model import TransformerModel
from trainer import ModelTrainer, TextDataset, SimpleTokenizer

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("RAG2")
console = Console()

def print_banner():
    console.print("""
+===========================================================+
|     RRRR    AAA   GGGG   222                             |
|     R   R  A   A G       2   2                            |
|     RRRR   AAAAA G  GGG  2222                             |
|     R  R   A   A G   G   2                                |
|     R   R  A   A  GGGG   22222                            |
|                                                           |
|     Model Training System - 1.25B Parameters              |
|     Supports: PHP, HTML, CSS, MySQL, JSON, Arabic         |
+===========================================================+
""", style="bold cyan")

def select_model_size():
    """Let user select model size"""
    console.print("\n[bold green]اختر حجم النموذج:[/bold green]\n")

    table = Table(title="أحجام النماذج المتاحة")
    table.add_column("الرقم", style="cyan")
    table.add_column("الحجم", style="magenta")
    table.add_column("Hidden Size", style="yellow")
    table.add_column("الطبقات", style="green")
    table.add_column("التقريب Parameters", style="red")

    for i, (name, config) in enumerate(MODEL_SIZES.items(), 1):
        params = calculate_parameters(
            config["hidden_size"],
            config["num_layers"],
            VOCAB_SIZE,
            config["intermediate_size"]
        )
        table.add_row(
            str(i),
            name.upper(),
            str(config["hidden_size"]),
            str(config["num_layers"]),
            f"{params / 1e9:.2f}B" if params >= 1e9 else f"{params / 1e6:.0f}M"
        )

    console.print(table)
    console.print()

    choice = IntPrompt.ask("أدخل رقم الحجم", default=6)
    sizes = list(MODEL_SIZES.keys())

    if 1 <= choice <= len(sizes):
        selected = sizes[choice - 1]
        console.print(f"[green]✓[/green] تم اختيار: {selected.upper()}")
        return selected
    else:
        console.print("[red]اختيار غير صالح، سيتم استخدام 1.25B[/red]")
        return "1.25b"

def create_model_config(size_name):
    """Create model configuration"""
    size_config = MODEL_SIZES[size_name]

    return {
        "vocab_size": VOCAB_SIZE,
        "hidden_size": size_config["hidden_size"],
        "num_layers": size_config["num_layers"],
        "num_heads": size_config["num_heads"],
        "intermediate_size": size_config["intermediate_size"],
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        "dropout": DROPOUT,
        # Training config
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_steps": WARMUP_STEPS,
        "max_steps": MAX_STEPS,
        "save_steps": SAVE_STEPS,
        "logging_steps": LOGGING_STEPS,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "max_grad_norm": MAX_GRAD_NORM,
        "fp16": FP16,
        "checkpoint_dir": CHECKPOINT_DIR
    }

def display_model_info(model, config):
    """Display model information"""
    params = model.count_parameters()

    console.print("\n[bold]معلومات النموذج:[/bold]")
    console.print(f"  Hidden Size: {config['hidden_size']}")
    console.print(f"  عدد الطبقات: {config['num_layers']}")
    console.print(f"  عدد Heads: {config['num_heads']}")
    console.print(f"  Intermediate Size: {config['intermediate_size']}")
    console.print(f"  [bold cyan]عدد Parameters: {params:,} ({params / 1e9:.2f}B)[/bold cyan]")

    # Memory estimate
    param_memory = params * 4 / (1024 ** 3)  # 4 bytes per float32
    console.print(f"  ذاكرة النموذج (تقريبي): {param_memory:.2f} GB")

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        console.print(f"  ذاكرة GPU المتاحة: {gpu_memory:.2f} GB")

async def train_model(size_name="1.25b", data_path=None, use_wandb=False):
    """Train a model"""
    console.print(f"\n[bold green]━━━ تدريب نموذج {size_name.upper()} ━━━[/bold green]\n")

    # Create config
    config = create_model_config(size_name)

    # Create model
    with console.status("[cyan]جاري إنشاء النموذج..."):
        model = TransformerModel(config)

    display_model_info(model, config)

    # Check if data exists
    if data_path is None:
        data_path = DATA_DIR

    if not os.path.exists(data_path):
        console.print(f"[yellow]مجلد البيانات غير موجود: {data_path}[/yellow]")
        console.print("[yellow]سيتم إنشاء بيانات تدريب افتراضية...[/yellow]")

        # Create sample data
        os.makedirs(data_path, exist_ok=True)
        sample_text = """
الذكاء الاصطناعي هو فرع من علوم الحاسوب يهدف إلى إنشاء أنظمة ذكية.
التعلم الآلي هو جزء من الذكاء الاصطناعي يسمح للحاسبات بالتعلم من البيانات.
التعلم العميق يستخدم الشبكات العصبية العميقة لمعالجة البيانات المعقدة.
نماذج اللغة الكبيرة هي نماذج مدربة على كميات ضخمة من النصوص.
RAG تجمع بين البحث في قواعد المعرفة وتوليد الإجابات.
الحوسبة الكمومية تستخدم ظواهر ميكانيكا الكم لحل المسائل المعقدة.
الروبوتات الذكية يمكنها التفاعل مع البيئة واتخاذ القرارات.
الواقع الافتراضي يوفر تجربة غامرة للمستخدمين.
بلوكتشين تكنولوجيا لامركزية لتسجيل المعاملات بأمان.
إنترنت الأشياء يربط الأجهزة الذكية بالإنترنت.
""" * 100  # Repeat to have enough data

        with open(os.path.join(data_path, "sample.txt"), "w", encoding="utf-8") as f:
            f.write(sample_text)

    # Create dataset
    tokenizer = SimpleTokenizer(config["vocab_size"])

    with console.status("[cyan]جاري تحميل البيانات..."):
        train_dataset = TextDataset(data_path, tokenizer, max_length=config["max_position_embeddings"])

    console.print(f"[green]✓[/green] تم تحميل {len(train_dataset)} عينة تدريب")

    if len(train_dataset) == 0:
        console.print("[red]لا توجد بيانات تدريب كافية![/red]")
        return

    # Use default training parameters
    console.print("\n[bold]إعدادات التدريب:[/bold]")
    num_epochs = 1
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]

    console.print(f"  عدد Epochs: {num_epochs}")
    console.print(f"  Batch Size: {batch_size}")
    console.print(f"  Learning Rate: {learning_rate}")

    config["batch_size"] = batch_size
    config["learning_rate"] = learning_rate

    # Create trainer
    trainer = ModelTrainer(model, config, use_wandb=use_wandb)

    # Start training
    console.print("\n[bold cyan]بدء التدريب...[/bold cyan]\n")

    try:
        history = trainer.train(train_dataset, num_epochs=num_epochs)

        # Display results
        console.print("\n[bold green]✓ تم التدريب بنجاح![/bold green]")

        # Show final stats
        final_loss = history[-1]["loss"] if history else 0
        console.print(f"  الخسارة النهائية: {final_loss:.4f}")
        console.print(f"  عدد الخطوات: {trainer.global_step}")
        console.print(f"  المجلد: {CHECKPOINT_DIR}")

    except KeyboardInterrupt:
        console.print("\n[yellow]تم إيقاف التدريب من قبل المستخدم[/yellow]")
        trainer.save_checkpoint(0, is_best=False)
    except Exception as e:
        console.print(f"\n[red]خطأ في التدريب: {e}[/red]")
        logger.error(f"Training error: {e}", exc_info=True)

async def run_gui():
    """Run the GUI interface"""
    try:
        from gui import main as gui_main
        gui_main()
    except ImportError:
        console.print("[red]خطأ: لا يمكن تشغيل الواجهة الرسومية[/red]")
        console.print("[yellow]تأكد من تثبيت tkinter[/yellow]")

async def interactive_mode():
    """Interactive mode for training"""
    console.print("\n[bold green]━━━ الوضع التفاعلي ━━━[/bold green]\n")

    while True:
        console.print("\n[bold]الخيارات:[/bold]")
        console.print("  [1] تدريب نموذج جديد")
        console.print("  [2] عرض أحجام النماذج")
        console.print("  [3] حساب Parameters")
        console.print("  [4] تشغيل الواجهة الرسومية")
        console.print("  [0] خروج\n")

        choice = Prompt.ask("اختيارك", choices=["0", "1", "2", "3", "4"])

        if choice == "0":
            console.print("[yellow]وداعاً![/yellow]")
            break
        elif choice == "1":
            size = select_model_size()
            await train_model(size)
        elif choice == "2":
            console.print("\n[bold]أحجام النماذج:[/bold]")
            table = Table()
            table.add_column("الحجم", style="cyan")
            table.add_column("Hidden", style="magenta")
            table.add_column("طبقات", style="yellow")
            table.add_column("Heads", style="green")
            table.add_column("Parameters", style="red")

            for name, config in MODEL_SIZES.items():
                params = calculate_parameters(
                    config["hidden_size"],
                    config["num_layers"],
                    VOCAB_SIZE,
                    config["intermediate_size"]
                )
                table.add_row(
                    name.upper(),
                    str(config["hidden_size"]),
                    str(config["num_layers"]),
                    str(config["num_heads"]),
                    f"{params / 1e9:.2f}B"
                )

            console.print(table)
        elif choice == "3":
            console.print("\n[bold]حساب Parameters:[/bold]")
            size = select_model_size()
            config = MODEL_SIZES[size]
            params = calculate_parameters(
                config["hidden_size"],
                config["num_layers"],
                VOCAB_SIZE,
                config["intermediate_size"]
            )
            console.print(f"\n[cyan]حجم النموذج: {size.upper()}[/cyan]")
            console.print(f"[green]عدد Parameters: {params:,}[/green]")
            console.print(f"[magenta]تقريباً: {params / 1e9:.2f} Billion[/magenta]")
        elif choice == "4":
            await run_gui()

async def main():
    parser = argparse.ArgumentParser(description="RAG2 - نظام تدريب النماذج")
    parser.add_argument("--size", type=str, choices=list(MODEL_SIZES.keys()),
                        help="حجم النموذج (small, medium, large, xl, 1b, 1.25b)")
    parser.add_argument("--data", type=str, help="مسار بيانات التدريب")
    parser.add_argument("--interactive", action="store_true", help="تشغيل الوضع التفاعلي")
    parser.add_argument("--gui", action="store_true", help="تشغيل الواجهة الرسومية")
    parser.add_argument("--wandb", action="store_true", help="استخدام Weights & Biases")

    args = parser.parse_args()

    print_banner()

    if args.gui:
        await run_gui()
    elif args.interactive:
        await interactive_mode()
    elif args.size:
        await train_model(args.size, args.data, args.wandb)
    else:
        await interactive_mode()

if __name__ == "__main__":
    asyncio.run(main())
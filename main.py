"""
Tourism Recommendation System Using Deep Learning
===================================================
CLI entry point for dataset generation, training, evaluation, and web app.

Usage:
    python main.py generate-data    Generate synthetic tourism dataset
    python main.py train            Train all models (NCF, Content, Hybrid)
    python main.py evaluate         Evaluate models and generate plots
    python main.py run-app          Launch the Streamlit web app
"""

import sys
import argparse

from utils.logger import get_project_logger

logger = get_project_logger(__name__)


def cmd_generate_data(args):
    """Generate synthetic tourism dataset."""
    from data.generate_dataset import generate_all
    generate_all()


def cmd_train(args):
    """Train recommendation models."""
    import torch
    from utils.preprocess import DataPreprocessor
    from utils.config import LEARNING_RATE, BATCH_SIZE
    from models.ncf import NCF
    from models.content_model import ContentModel
    from models.hybrid import HybridRecommender
    from training.train import Trainer

    # Prepare data
    preprocessor = DataPreprocessor()
    train_loader, val_loader, _, user_features, dest_features = preprocessor.prepare_all(
        batch_size=BATCH_SIZE
    )
    num_users = preprocessor.num_users
    num_dests = preprocessor.num_destinations

    histories = []
    epochs = args.epochs

    # ─── Train NCF ───
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Training NCF Model")
    logger.info("=" * 60)
    ncf = NCF(num_users, num_dests)
    ncf_trainer = Trainer(ncf, model_name="ncf", lr=LEARNING_RATE, epochs=epochs)
    ncf_hist = ncf_trainer.train(train_loader, val_loader)
    histories.append(ncf_hist)

    # ─── Train Content Model ───
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Training Content Model")
    logger.info("=" * 60)
    content = ContentModel()
    content_trainer = Trainer(content, model_name="content", lr=LEARNING_RATE, epochs=epochs)
    content_hist = content_trainer.train(train_loader, val_loader)
    histories.append(content_hist)

    # ─── Train Hybrid Model ───
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Training Hybrid Model")
    logger.info("=" * 60)
    hybrid = HybridRecommender(num_users, num_dests)
    hybrid_trainer = Trainer(hybrid, model_name="hybrid", lr=LEARNING_RATE, epochs=epochs)
    hybrid_hist = hybrid_trainer.train(train_loader, val_loader)
    histories.append(hybrid_hist)

    # Save training curves
    from training.evaluate import Evaluator
    Evaluator.plot_training_curves(histories)

    logger.info("\n✅ All models trained successfully!")
    return histories


def cmd_evaluate(args):
    """Evaluate all trained models."""
    import torch
    from utils.preprocess import DataPreprocessor
    from utils.config import BATCH_SIZE, MODEL_DIR
    from models.ncf import NCF
    from models.content_model import ContentModel
    from models.hybrid import HybridRecommender
    from training.evaluate import Evaluator

    # Prepare data
    preprocessor = DataPreprocessor()
    _, _, test_loader, _, _ = preprocessor.prepare_all(batch_size=BATCH_SIZE)
    num_users = preprocessor.num_users
    num_dests = preprocessor.num_destinations

    evaluator = Evaluator()
    results = []

    import os
    models_to_eval = [
        ("ncf", NCF(num_users, num_dests)),
        ("content", ContentModel()),
        ("hybrid", HybridRecommender(num_users, num_dests)),
    ]

    for model_name, model in models_to_eval:
        ckpt_path = os.path.join(MODEL_DIR, f"{model_name}_best.pt")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
            result = evaluator.evaluate_model(model, test_loader, model_name)
            results.append(result)
        else:
            logger.warning(f"  ⚠ Checkpoint not found: {ckpt_path}")

    if results:
        Evaluator.plot_model_comparison(results)
        Evaluator.plot_prediction_distribution(results)
        logger.info("\n✅ Evaluation complete! Plots saved to plots/")


def cmd_run_app(args):
    """Launch the Streamlit web app."""
    import subprocess
    logger.info("🌍 Launching Tourism Recommendation Web App...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "app/streamlit_app.py",
        "--server.headless", "true",
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Tourism Recommendation System Using Deep Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate-data
    subparsers.add_parser("generate-data", help="Generate synthetic tourism dataset")

    # train
    train_parser = subparsers.add_parser("train", help="Train recommendation models")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (default: 50)")

    # evaluate
    subparsers.add_parser("evaluate", help="Evaluate models and generate plots")

    # run-app
    subparsers.add_parser("run-app", help="Launch the Streamlit web app")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "generate-data": cmd_generate_data,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "run-app": cmd_run_app,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()

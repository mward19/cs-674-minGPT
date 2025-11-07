from mingpt.model import GPT
from dataloader import JSONLDataset
from mingpt.trainer import Trainer
import os
from datetime import datetime


if __name__ == '__main__':
    # Handle ablation arguments
    import argparse
    parser = argparse.ArgumentParser(
        prog='MinGPT ablations',
        description='Test some modifications of minGPT',
    )

    parser.add_argument('--swiglu', action='store_true') 
    parser.add_argument('--rotary', action='store_true') 
    parser.add_argument('--lr-linear', action='store_true') 
    parser.add_argument('--lr-cosine', action='store_true') 
    parser.add_argument('--rmsnorm', action='store_true') 

    args = parser.parse_args()

    print(vars(args))

    dataset = JSONLDataset(
        '/nobackup/autodelete/usr/mward19/pile_data_head.jsonl',
        test_size=1,
        window_size=128
    )

    # Initialize model
    print("Preparing model")
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'
    model_config.vocab_size = dataset.get_vocab_size()
    model_config.block_size = dataset.get_block_size()
    # -- Ablations
    model_config.swiglu = args.swiglu
    model_config.rotary = args.rotary
    model_config.rmsnorm = args.rmsnorm
    model = GPT(model_config)

    # create a Trainer object
    print("Preparing trainer")
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4
    train_config.max_iters = 3000
    train_config.num_workers = 1
    train_config.device = 'cuda'
    # -- Ablations
    train_config.lr_linear = args.lr_linear
    train_config.lr_cosine = args.lr_cosine
    trainer = Trainer(train_config, model, dataset)

    # Train the model
    def batch_end_callback(trainer):
        if trainer.iter_num % 1 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    trainer.set_callback('on_batch_end', batch_end_callback)
    print("Training")
    final_loss = trainer.run()


    # Build filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flags = "_".join([f for f in ['swiglu', 'rotary', 'lr-linear', 'lr-cosine', 'rmsnorm'] if getattr(args, f.replace('-', '_'))])
    suffix = f"_{flags}" if flags else ""
    filename = f"{timestamp}{suffix}.out"

    # Ensure output directory exists
    output_dir = os.path.expanduser('~/Documents/cs-674/cs-674-minGPT/outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Write final loss
    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write(f"Final loss: {final_loss}\n")
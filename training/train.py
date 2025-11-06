from mingpt.model import GPT
from dataloader import JSONLDataset
from mingpt.trainer import Trainer


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

    dataset = JSONLDataset(
        '/nobackup/autodelete/usr/mward19/pile_data_head.jsonl',
        test_size=1,
        window_size=16
    )

    # Initialize model
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
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4
    train_config.max_iters = 10
    train_config.num_workers = 0
    train_config.device = 'cpu'
    # -- Ablations
    train_config.lr_linear = args.lr_linear
    train_config.lr_cosine = args.lr_cosine
    trainer = Trainer(train_config, model, dataset)

    # Train the model
    def batch_end_callback(trainer):
        if trainer.iter_num % 1 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()


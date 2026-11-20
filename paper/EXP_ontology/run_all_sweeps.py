import wandb
from baseline_sweep import main


lr_list = [0.0005, 0.005, 0.01]
margin_list = [0, 0.05, 0.1, 0.15]

baseline_config = {
    'method': 'grid',
    'name': 'PLACEHOLDER',
    'metric': {
        'name': 'surrogate',
        'goal': 'minimize'
    },
    'parameters': {
        'dataset': {
            'value': 'PLACEHOLDER'
        },
        'task': {
            'value': 'PLACEHOLDER'
        },
        'model': {
            'value': 'PLACEHOLDER'
        },
        'lr': {
            'values': lr_list
        },
        'margin': {
            'values': margin_list
        },
        'embedding_dim': {
            "value": "PLACEHOLDER"
        },
        'plugin_type': {
            'values': ['tBox', 'RegD']
        }
    }
}

def train_elbe_elem():
    config = baseline_config
    for task in ['prediction', 'inferences']:
        for model in ['elbe', 'elem']:
            for dataset in ['GALEN', 'GO', 'ANATOMY']:
                for emb_dim in [5, 10]:
                    print(f'Starting sweep {model}-{dataset}-{task}')
                    config['name'] = f'{model}-{dataset}-{task}'
                    config['parameters']['dataset']['value'] = dataset
                    config['parameters']['task']['value'] = task
                    config['parameters']['model']['value'] = model
                    config['parameters']['embedding_dim']['value'] = emb_dim
                    sweep_id = wandb.sweep(sweep=config, project='RegD_EL')
                    print(f'Starting agent')
                    wandb.agent(sweep_id, function=main)


if __name__ == '__main__':
    train_elbe_elem()


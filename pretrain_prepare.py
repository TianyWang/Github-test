import paddle


def prep_env():
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    settings = {
        "data_path": "./datasets",

        "train_filename": None,
        "input_len": 144,
        "output_len": 288,
        "label_len": 72,
        "start_col": 3,
        "capacity": 134,
        "patient": 5,
        "day_len": 144,
        "train_size": 155,
        "val_size": 80,
        "aug_p": 0.9,
        "is_debug": True,
        "checkpoints": "./pretrain_checkpoints",
        "logs_path": "./logs",
        "num_workers": 5,
        "train_epochs": 15,
        "batch_size": 128,
        "log_per_steps": 100,
        "lr": 0.0001,
        "lr_adjust": 'type1',
        "gpu": 0,
        "name": "Filter_MSE_Loss",
        "pred_file": "predict.py",
        "framework": "paddlepaddle",
    }
    # Prepare the GPUs
    if paddle.device.is_compiled_with_cuda():
        settings["use_gpu"] = True
        paddle.device.set_device('gpu:{}'.format(settings["gpu"]))
    else:
        settings["use_gpu"] = False
        paddle.device.set_device('cpu')
    return settings

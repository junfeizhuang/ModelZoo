class config:
    arch = 'EfficientNet_B0'
    data_dir = '/local/zjf/tiny-imagenet-200/'
    train_txt = './train.txt'
    val_txt = './val.txt'
    log_dir = './log'
    input_size = 64
    lr = 0.1
    momentum = 0.9
    weight_decay = 0.0001
    epochs = 120
    gpu = 3
    print_frep = 100
    train_batch_size = 128
    eval_batch_size = 32
    workers = 4

cfg = config()
        
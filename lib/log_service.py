from .cfg_holder import cfg_unique_holder as cfguh


def print_log(*console_info):
    console_info = [str(i) for i in console_info]
    console_info = ' '.join(console_info)
    print(console_info)
    try:
        log_file = cfguh().cfg.train.log_file
    except:
        try:
            log_file = cfguh().cfg.eval.log_file
        except:
            return
    # TODO: potential bug on both have train and eval
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(console_info + '\n')

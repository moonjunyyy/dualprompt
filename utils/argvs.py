l2p_argvs = {
    "--batchsize"      : 32,
    "--stepsize"       : 128,
    "--epochs"         : 5,
    "--log-interval"   : 40,
    "--pool-size"      : 10,
    "--selection-size" : 5,
    "--prompt-len"     : 5,

    "--dimention"      : -1,
    "--num-tasks"      : 10,
    
    "--num-class"      : 100,
    "--lr-scheduler"   : None,
    "--use-amp"        : False,

    "--save-path"      : "saved/l2p/CIL",
    "--data-path"      : "/mnt/e/Datasets/CIFAR100/",
    "--backbone-name"  : "vit_base_patch16_224",

    "--debug"          : False,
}